import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
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
from helper.dataset import MaterialDataset, custom_collate_fn
from src.model import MultiModalMaterialClassifier 

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
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
    print(f"Classification Report:\n{json.dumps(report, indent=2)}")
    print(f"Confusion Matrix:\n{cm}")
    
    return acc, report, cm

def train_main_classifier():
    print(f"Using device: {config.DEVICE}")

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("\nStarting multi-modal material classification training...")
    print("Setting up environment...")
    
    full_dataset_for_dim_inference = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        scaler=None
    )
    
    try:
        _ = full_dataset_for_dim_inference[0] 
        print("Dataset dimensions inferred and config updated.")
    except Exception as e:
        print(f"Warning: Could not load first sample to infer dimensions: {e}")
        print("Please ensure config.py has accurate feature dimensions set manually.")

    print("Environment setup complete.")

    print("Fitting StandardScaler for scalar features on training data...")
    dataset_indices = list(range(len(full_dataset_for_dim_inference)))
    train_indices, temp_indices = train_test_split(
        dataset_indices, 
        train_size=config.TRAIN_RATIO, 
        random_state=config.SEED
    )
    val_indices, test_indices = train_test_split(
        temp_indices, 
        train_size=config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.SEED
    )

    temp_train_dataset_for_scaler = torch.utils.data.Subset(full_dataset_for_dim_inference, train_indices)
    
    scalar_features_list = []
    temp_train_loader_for_scaler = DataLoader(
        temp_train_dataset_for_scaler, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.NUM_WORKERS
    )

    for batch in temp_train_loader_for_scaler:
        scalar_features_list.append(batch['scalar_features'].cpu().numpy())
    
    if scalar_features_list:
        all_train_scalar_features = np.vstack(scalar_features_list)
        scaler = StandardScaler()
        scaler.fit(all_train_scalar_features)
        print("StandardScaler fitted on training data.")
    else:
        scaler = None
        print("No scalar features collected for scaler fitting. Scaler will not be used.")

    final_full_dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        scaler=scaler
    )
    print("Final dataset re-initialized with fitted scaler.")

    train_dataset = torch.utils.data.Subset(final_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(final_full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(final_full_dataset, test_indices)

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = PyGDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    val_loader = PyGDataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    test_loader = PyGDataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)

    model = MultiModalMaterialClassifier(
        crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim=config.ASPH_FEATURE_DIM,
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
        num_topology_classes=config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes=config.NUM_MAGNETISM_CLASSES,
        
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

    criterion_magnetism = nn.CrossEntropyLoss()
    criterion_topology = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training loop...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            batch['crystal_graph'] = batch['crystal_graph'].to(config.DEVICE)
            batch['kspace_graph'] = batch['kspace_graph'].to(config.DEVICE)
            batch['asph_features'] = batch['asph_features'].to(config.DEVICE)
            batch['scalar_features'] = batch['scalar_features'].to(config.DEVICE)
            batch['topology_label'] = batch['topology_label'].to(config.DEVICE)
            batch['magnetism_label'] = batch['magnetism_label'].to(config.DEVICE)
            for sub_key in batch['kspace_physics_features']: 
                 batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 

            optimizer.zero_grad()
            
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
            
            total_loss.backward()
            
            # --- ADDED: Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM) # Use max_norm from config
            
            optimizer.step()
            total_train_loss += total_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {total_loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        all_topo_preds = []
        all_topo_targets = []
        all_magnetism_preds = []
        all_magnetism_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
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

        _ = compute_metrics(torch.tensor(all_topo_preds), torch.tensor(all_topo_targets), config.NUM_TOPOLOGY_CLASSES, "Topology Classification")
        _ = compute_metrics(torch.tensor(all_magnetism_preds), torch.tensor(all_magnetism_targets), config.NUM_MAGNETISM_CLASSES, "Magnetism Classification")

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

    print("\n--- Evaluating on Test Set ---")
    model.eval()

    all_topo_preds_test = []
    all_topo_targets_test = []
    all_magnetism_preds_test = []
    all_magnetism_targets_test = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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

def main_training_loop(): 
    train_main_classifier()
