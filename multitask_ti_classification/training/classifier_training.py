import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # Keep DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader # Specifically import for PyG
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import random
import torch_geometric

# Import from local modules
from helper import config
from helper.dataset import MaterialDataset, custom_collate_fn # Corrected path/import
from src.model import MultiModalMaterialClassifier 

# It's good practice to have this global torch.load safety line in main.py
# or any script that is the primary entry point and might load .pt files.
# If it's already in dataset.py, it's fine, but redundant in multiple places won't hurt.
# from torch_geometric.data import DataEdgeAttr # If you want to be super explicit
# torch.serialization.add_safe_globals([DataEdgeAttr])

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr # ADD THIS LINE
])

def compute_metrics(predictions, targets, num_classes, task_name):
    """Computes accuracy and detailed classification report."""
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    acc = accuracy_score(targets_np, preds_np)
    report = classification_report(targets_np, preds_np, output_dict=True, zero_division=0)
    cm = confusion_matrix(targets_np, preds_np)

    print(f"\n--- {task_name} Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    # Use json.dumps for pretty printing the report dictionary
    print(f"Classification Report:\n{json.dumps(report, indent=2)}")
    print(f"Confusion Matrix:\n{cm}")
    
    return acc, report, cm

def train_main_classifier():
    print(f"Using device: {config.DEVICE}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED) # Need to import 'random'
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- 1. Load Data and Create Datasets ---
    print("\nStarting multi-modal material classification training...")
    print("Setting up environment...")
    
    # Instantiate the full dataset to get dimensions for scaler and model init.
    # The __init__ of MaterialDataset will automatically populate relevant config dimensions
    # after its first material is processed (which happens during __init__ for basic checks
    # or upon the first __getitem__ call).
    full_dataset_for_dim_inference = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        scaler=None # Do not pass scaler yet
    )
    
    # Trigger __getitem__[0] to ensure all dynamic config dimensions are set before model init.
    # This is important because the model's __init__ will then read these from config.
    # Use a try-except block here for robustness in case the first item has issues.
    try:
        _ = full_dataset_for_dim_inference[0] 
        print("Dataset dimensions inferred and config updated.")
    except Exception as e:
        print(f"Warning: Could not load first sample to infer dimensions: {e}")
        print("Please ensure config.py has accurate feature dimensions set manually.")
        # If this fails, your config dimensions MUST be manually accurate.

    print("Environment setup complete.")

    # --- IMPORTANT: Scaler Fitting Strategy ---
    # It's best practice to fit the scaler ONLY on the training data to avoid data leakage.
    #
    # Current approach: Fit on ALL data, then split, then re-initialize.
    # This works but means the validation/test sets influence the scaler, which is data leakage.
    #
    # Recommended approach:
    # 1. Split full_dataset_for_dim_inference into train/val/test *indices*.
    # 2. Create a temporary DataLoader for *only the training indices* to collect scalar features.
    # 3. Fit the scaler on these training scalar features.
    # 4. Create final train/val/test DataLoaders, passing the *fitted scaler* to their datasets.

    print("Fitting StandardScaler for scalar features on training data...")
    # Perform train/val/test split using indices first
    dataset_indices = list(range(len(full_dataset_for_dim_inference)))
    train_indices, temp_indices = train_test_split(
        dataset_indices, 
        train_size=config.TRAIN_RATIO, 
        random_state=config.SEED
    )
    val_indices, test_indices = train_test_split(
        temp_indices, 
        train_size=config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO), # Correct ratio for second split
        random_state=config.SEED
    )

    # Create a temporary subset for training data to fit the scaler
    temp_train_dataset_for_scaler = torch.utils.data.Subset(full_dataset_for_dim_inference, train_indices)
    
    scalar_features_list = []
    # Use a proper DataLoader for efficient batching during scaler fitting
    temp_train_loader_for_scaler = DataLoader(
        temp_train_dataset_for_scaler, 
        batch_size=config.BATCH_SIZE, # Use batch size for efficiency
        shuffle=False, # No need to shuffle for scaler fitting
        collate_fn=custom_collate_fn,
        num_workers=config.NUM_WORKERS # Use workers if available
    )

    # Collect scalar features from the training set only
    for batch in temp_train_loader_for_scaler:
        scalar_features_list.append(batch['scalar_features'].cpu().numpy())
    
    if scalar_features_list:
        all_train_scalar_features = np.vstack(scalar_features_list) # Use vstack for batches
        scaler = StandardScaler()
        scaler.fit(all_train_scalar_features)
        print("StandardScaler fitted on training data.")
    else:
        scaler = None
        print("No scalar features collected for scaler fitting. Scaler will not be used.")

    # --- Re-instantiate full dataset with the fitted scaler for final DataLoader creation ---
    # This ensures train, val, and test datasets all use the same scaler, fitted only on train.
    final_full_dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        scaler=scaler # Pass the fitted scaler
    )
    print("Final dataset re-initialized with fitted scaler.")

    # --- Create final DataLoaders using Subset and the final_full_dataset ---
    train_dataset = torch.utils.data.Subset(final_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(final_full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(final_full_dataset, test_indices)

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Use PyGDataLoader (from torch_geometric.loader) for PyG Data objects
    train_loader = PyGDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    val_loader = PyGDataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    test_loader = PyGDataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)

    # --- 3. Initialize Model ---
    model = MultiModalMaterialClassifier(
        crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim=config.ASPH_FEATURE_DIM,
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
        num_topology_classes=config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes=config.NUM_MAGNETISM_CLASSES,
        
        # Pass encoder specific params from config (ensure these exist in config.py)
        egnn_hidden_irreps_str=config.EGNN_HIDDEN_IRREPS_STR,
        egnn_num_layers=config.GNN_NUM_LAYERS, 
        egnn_radius=config.EGNN_RADIUS, 

        kspace_gnn_hidden_channels=config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers=config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads=config.KSPACE_GNN_NUM_HEADS, 

        ffnn_hidden_dims_asph=config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar=config.FFNN_HIDDEN_DIMS_SCALAR,
        
        latent_dim_gnn=config.LATENT_DIM_GNN,
        latent_dim_ffnn=config.LATENT_DIM_FFNN,
        fusion_hidden_dims=config.FUSION_HIDDEN_DIMS
    ).to(config.DEVICE)

    print(f"Model instantiated successfully: \n{model}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- 4. Loss Functions and Optimizer ---
    criterion_magnetism = nn.CrossEntropyLoss()
    criterion_topology = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training loop...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            # PyG Data objects need to be moved to device:
            batch['crystal_graph'] = batch['crystal_graph'].to(config.DEVICE)
            batch['kspace_graph'] = batch['kspace_graph'].to(config.DEVICE)
            
            # Other tensors (non-PyG Data)
            batch['asph_features'] = batch['asph_features'].to(config.DEVICE)
            batch['scalar_features'] = batch['scalar_features'].to(config.DEVICE)
            batch['topology_label'] = batch['topology_label'].to(config.DEVICE)
            batch['magnetism_label'] = batch['magnetism_label'].to(config.DEVICE)
            
            # K-space physics features are a dictionary of tensors
            for sub_key in batch['kspace_physics_features']: 
                 batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 

            optimizer.zero_grad()
            
            # Prepare input dictionary for the model's forward pass
            model_inputs = {
                'crystal_graph': batch['crystal_graph'],
                'kspace_graph': batch['kspace_graph'],
                'asph_features': batch['asph_features'],
                'scalar_features': batch['scalar_features'],
                'kspace_physics_features': batch['kspace_physics_features'] # Pass the entire dict
            }

            outputs = model(model_inputs)
            topology_logits = outputs['topology_logits']
            magnetism_logits = outputs['magnetism_logits']

            loss_topology = criterion_topology(topology_logits, batch['topology_label'])
            loss_magnetism = criterion_magnetism(magnetism_logits, batch['magnetism_label'])

            total_loss = (config.LOSS_WEIGHT_TOPOLOGY * loss_topology +
                          config.LOSS_WEIGHT_MAGNETISM * loss_magnetism)
            
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()
            
            # Optional: print batch-level loss
            if batch_idx % 50 == 0: # Adjusted interval for more frequent updates
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {total_loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        all_topo_preds = []
        all_topo_targets = []
        all_magnetism_preds = []
        all_magnetism_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move data to device (same as train loop)
                batch['crystal_graph'] = batch['crystal_graph'].to(config.DEVICE)
                batch['kspace_graph'] = batch['kspace_graph'].to(config.DEVICE)
                batch['asph_features'] = batch['asph_features'].to(config.DEVICE)
                batch['scalar_features'] = batch['scalar_features'].to(config.DEVICE)
                batch['topology_label'] = batch['topology_label'].to(config.DEVICE)
                batch['magnetism_label'] = batch['magnetism_label'].to(config.DEVICE)
                for sub_key in batch['kspace_physics_features']: 
                     batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 
                
                model_inputs = {
                    'crystal_graph': batch['crystal_graph'],
                    'kspace_graph': batch['kspace_graph'],
                    'asph_features': batch['asph_features'],
                    'scalar_features': batch['scalar_features'],
                    'kspace_physics_features': batch['kspace_physics_features']
                }

                outputs = model(model_inputs)
                topology_logits = outputs['topology_logits']
                magnetism_logits = outputs['magnetism_logits']

                loss_topology = criterion_topology(topology_logits, batch['topology_label'])
                loss_magnetism = criterion_magnetism(magnetism_logits, batch['magnetism_label'])
                
                total_loss = (config.LOSS_WEIGHT_TOPOLOGY * loss_topology +
                              config.LOSS_WEIGHT_MAGNETISM * loss_magnetism)
                total_val_loss += total_loss.item()

                all_topo_preds.extend(torch.argmax(topology_logits, dim=1).cpu().numpy())
                all_topo_targets.extend(batch['topology_label'].cpu().numpy())
                all_magnetism_preds.extend(torch.argmax(magnetism_logits, dim=1).cpu().numpy())
                all_magnetism_targets.extend(batch['magnetism_label'].cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")

        # Compute and print metrics for each task
        # Convert lists back to tensors before calling compute_metrics
        _ = compute_metrics(torch.tensor(all_topo_preds), torch.tensor(all_topo_targets), config.NUM_TOPOLOGY_CLASSES, "Topology Classification")
        _ = compute_metrics(torch.tensor(all_magnetism_preds), torch.tensor(all_magnetism_targets), config.NUM_MAGNETISM_CLASSES, "Magnetism Classification")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_save_path = config.MODEL_SAVE_DIR / "best_multi_task_classifier.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"  Model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

    # --- 6. Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load(config.MODEL_SAVE_DIR / "best_multi_task_classifier.pth"))
    model.eval()

    all_topo_preds_test = []
    all_topo_targets_test = []
    all_magnetism_preds_test = []
    all_magnetism_targets_test = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device (same as train loop)
            batch['crystal_graph'] = batch['crystal_graph'].to(config.DEVICE)
            batch['kspace_graph'] = batch['kspace_graph'].to(config.DEVICE)
            batch['asph_features'] = batch['asph_features'].to(config.DEVICE)
            batch['scalar_features'] = batch['scalar_features'].to(config.DEVICE)
            batch['topology_label'] = batch['topology_label'].to(config.DEVICE)
            batch['magnetism_label'] = batch['magnetism_label'].to(config.DEVICE)
            for sub_key in batch['kspace_physics_features']: 
                 batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 

            model_inputs = {
                'crystal_graph': batch['crystal_graph'],
                'kspace_graph': batch['kspace_graph'],
                'asph_features': batch['asph_features'],
                'scalar_features': batch['scalar_features'],
                'kspace_physics_features': batch['kspace_physics_features']
            }
            outputs = model(model_inputs)
            topology_logits = outputs['topology_logits']
            magnetism_logits = outputs['magnetism_logits']

            all_topo_preds_test.extend(torch.argmax(topology_logits, dim=1).cpu().numpy())
            all_topo_targets_test.extend(batch['topology_label'].cpu().numpy())
            all_magnetism_preds_test.extend(torch.argmax(magnetism_logits, dim=1).cpu().numpy())
            all_magnetism_targets_test.extend(batch['magnetism_label'].cpu().numpy())

    print("\nTest Set Results:")
    _ = compute_metrics(torch.tensor(all_topo_preds_test), torch.tensor(all_topo_targets_test), config.NUM_TOPOLOGY_CLASSES, "Topology Classification")
    _ = compute_metrics(torch.tensor(all_magnetism_preds_test), torch.tensor(all_magnetism_targets_test), config.NUM_MAGNETISM_CLASSES, "Magnetism Classification")

def main_training_loop(): # This is what main.py calls
    train_main_classifier()