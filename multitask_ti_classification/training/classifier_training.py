import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings

from helper import config
from helper.dataset import MaterialDataset, custom_collate_fn
from src.model import MultiModalMaterialClassifier 

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

    # --- 1. Load Dataset and Create Scaler ---
    full_dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR
    )

    # Infer dimensions after loading the first item (dataset.py updates config)
    # This relies on the _getitem_ being called once to populate config
    _ = full_dataset[0] 

    # Fit scaler on scalar features of the training set (or full dataset, then apply)
    # It's better to fit on training data only to avoid data leakage
    # For now, let's fit on a sample of data or the whole, assuming it's pre-split later.
    # A more robust way would be to create a temporary DataLoader for scalar features only.
    print("Fitting StandardScaler for scalar features...")
    scalar_features_list = []
    for i in range(len(full_dataset)):
        # Temporarily suppress the k-space warning for scaler fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample = full_dataset[i] # This will trigger _getitem_ and load all features
            scalar_features_list.append(sample['scalar_features'].cpu().numpy())
    
    all_scalar_features = np.array(scalar_features_list)
    scaler = StandardScaler()
    scaler.fit(all_scalar_features)
    
    # Re-initialize dataset with the fitted scaler
    full_dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        scaler=scaler
    )
    print("StandardScaler fitted and applied to dataset.")

    # --- 2. Split Dataset ---
    total_size = len(full_dataset)
    train_size = int(config.TRAIN_RATIO * total_size)
    val_size = int(config.VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    # --- 3. Initialize Model ---
    model = MultiModalMaterialClassifier(
        crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim=config.ASPH_FEATURE_DIM,
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
        num_topology_classes=config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes=config.NUM_MAGNETISM_CLASSES,
        
        # Pass encoder specific params from config
        egnn_hidden_irreps_str=config.EGNN_HIDDEN_IRREPS_STR, # Add this to config.py
        egnn_num_layers=config.GNN_NUM_LAYERS, # Reuse GNN_NUM_LAYERS for EGNN
        egnn_radius=config.EGNN_RADIUS, # Add this to config.py

        kspace_gnn_hidden_channels=config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers=config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads=config.KSPACE_GNN_NUM_HEADS, # Add this to config.py

        ffnn_hidden_dims_asph=config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar=config.FFNN_HIDDEN_DIMS_SCALAR,
        
        latent_dim_gnn=config.LATENT_DIM_GNN,
        latent_dim_ffnn=config.LATENT_DIM_FFNN,
        fusion_hidden_dims=config.FUSION_HIDDEN_DIMS
    ).to(config.DEVICE)

    # --- 4. Loss Functions and Optimizer ---
    # For Magnetism: It's currently multi-class (4 classes: NM, FM, AFM, FiM)
    criterion_magnetism = nn.CrossEntropyLoss()
    # For Topology: Multi-class (3 classes: TI, TSM, Trivial)
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
            for key in ['crystal_graph', 'kspace_graph']:
                batch[key] = batch[key].to(config.DEVICE)
            for key in ['asph_features', 'scalar_features', 'topology_label', 'magnetism_label']:
                batch[key] = batch[key].to(config.DEVICE)
            for sub_key in batch['kspace_physics_features']: 
                 batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 

            optimizer.zero_grad()
            
            outputs = model(batch)
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
            # if batch_idx % 100 == 0:
            #     print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}")

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
                for key in ['crystal_graph', 'kspace_graph']:
                    batch[key] = batch[key].to(config.DEVICE)
                for key in ['asph_features', 'scalar_features', 'topology_label', 'magnetism_label']:
                    batch[key] = batch[key].to(config.DEVICE)

                outputs = model(batch)
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
            if patience_counter >= config.PATIENCE: # Add PATIENCE to config.py
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
            for key in ['crystal_graph', 'kspace_graph']:
                batch[key] = batch[key].to(config.DEVICE)
            for key in ['asph_features', 'scalar_features', 'topology_label', 'magnetism_label']:
                batch[key] = batch[key].to(config.DEVICE)

            outputs = model(batch)
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