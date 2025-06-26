import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import json
import pickle
from pathlib import Path
import tqdm
import warnings
import random
import torch_geometric
import shutil
from collections import Counter
from torch_geometric.data import Data as PyGData

# Import from local modules
import helper.config as config
from helper.enhanced_topological_loss import EnhancedTopologicalLoss
# Assuming MaterialDataset and custom_collate_fn are in helper.dataset
from helper.dataset import MaterialDataset, custom_collate_fn 
from src.model import MultiModalMaterialClassifier
from helper.data_processing import ImprovedDataPreprocessor, StratifiedDataSplitter # NEW IMPORT

# Ensure PyG's global settings are safe for serialization
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
    labels = list(range(num_classes))
    report = classification_report(targets_np, preds_np, output_dict=True, zero_division=0, labels=labels)
    cm = confusion_matrix(targets_np, preds_np, labels=labels)

    print(f"\n--- {task_name} Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Classification Report:\n{json.dumps(report, indent=2)}")
    print(f"Confusion Matrix:\n{cm}")
    
    return acc, report, cm

def train_main_classifier():
    print(f"Using device: {config.DEVICE}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("\nStarting multi-modal material classification training...")
    print("Setting up environment...")
    
    # --- Data Loading and Preprocessing ---
    # Load the full dataset initially without a scaler
    full_dataset_raw = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        scaler=None # Scaler is handled by preprocessor
    )
    
    # Access a dummy sample to infer dimensions and trigger config updates in MaterialDataset's __getitem__
    # This is a hack; ideally, your config would be fully populated before dataset initialization.
    try:
        if len(full_dataset_raw) > 0:
            _ = full_dataset_raw[0] 
            print("Dataset dimensions inferred and config updated (if dynamic).")
        else:
            warnings.warn("Dataset is empty. Cannot infer dimensions. Ensure config.py has accurate feature dimensions.")
    except Exception as e:
        warnings.warn(f"Could not load first sample to infer dimensions: {e}. Ensure config.py has accurate feature dimensions set manually.")

    print("Environment setup complete.")

    print("\nStarting data preprocessing with ImprovedDataPreprocessor...")
    preprocessor = ImprovedDataPreprocessor()
    
    all_raw_data_dicts = []
    # Iterate through the raw dataset to collect data for preprocessing
    for i in tqdm(range(len(full_dataset_raw)), desc="Collecting raw data for preprocessing"):
        try:
            # When calling MaterialDataset[i], it returns a dict with tensors.
            # We need to collect these dicts before fitting the preprocessor.
            item_data = full_dataset_raw[i]
            # Ensure tensors are on CPU for numpy conversion in preprocessor
            # This is important if MaterialDataset returns CUDA tensors by default
            item_data_cpu = {}
            for k, v in item_data.items():
                if isinstance(v, torch.Tensor):
                    item_data_cpu[k] = v.cpu()
                elif isinstance(v, dict): # Handle kspace_physics_features
                    item_data_cpu[k] = {sk: sv.cpu() if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
                else:
                    item_data_cpu[k] = v
            all_raw_data_dicts.append(item_data_cpu)
        except Exception as e:
            warnings.warn(f"Skipping material at index {i} due to loading error during raw data collection: {e}")
            continue

    # Fit scalers and transform all data
    # The preprocessor will update the scalar features and ensure consistent tensor types.
    processed_data_list = preprocessor.fit_transform(all_raw_data_dicts)
    print(f"Data preprocessing complete. Processed {len(processed_data_list)} samples.")
    
    print("\nCreating stratified data splits with StratifiedDataSplitter...")
    # Stratify by combined_label (which is generated in collate_fn, not directly present here)
    # So, we stratify by primary labels (topology + magnetism) available in processed_data_list
    # Note: If your splitter expects a single 'y', you might need to combine them for splitting
    # For now, let's assume `split` can handle a list of dicts directly with a specified label key
    
    # To stratify correctly, the splitter needs access to labels.
    # If the splitter takes `y` as a list of labels, we must provide it.
    # Let's extract the combined label as the primary stratification target from `processed_data_list`
    # You will need to make sure your `MaterialDataset` also returns the raw strings `topological_class` 
    # and `magnetic_type` in `__getitem__` for the preprocessor and splitter to access.
    # Alternatively, the splitter could use the integer `topology_label` and `magnetism_label`
    # and you define a combined target *before* splitting for stratification.

    # Re-extract labels for stratification from the processed_data_list directly
    # This implies that `processed_data_list` elements (dicts) contain 'topology_label' and 'magnetism_label'
    stratify_labels_topo = [d['topology_label'].item() for d in processed_data_list]
    stratify_labels_mag = [d['magnetism_label'].item() for d in processed_data_list]

    # Combine topology and magnetism labels for stratification if a single target is needed.
    # A simple way to combine: (topo_label, mag_label) tuple as a stratification key.
    stratify_labels_combined = [
        config.get_combined_label_from_ints(topo, mag)
        for topo, mag in zip(stratify_labels_topo, stratify_labels_mag)
    ]
    
    splitter = StratifiedDataSplitter(
        test_size=config.TEST_RATIO,
        val_size=config.VAL_RATIO,
        random_state=config.SEED,
        # The `split` method needs to know which key in `processed_data_list` to stratify by.
        # Assuming `split` method is designed to take the `stratify_labels_combined` list directly.
    )
    
    train_data_list, val_data_list, test_data_list = splitter.split(
        data_list=processed_data_list,
        stratify_labels=stratify_labels_combined # Pass the combined labels for stratification
    )
    
    print(f"Dataset split: Train={len(train_data_list)}, Val={len(val_data_list)}, Test={len(test_data_list)}")

    # Create PyGDataLoader from the lists of processed data dictionaries
    train_loader = PyGDataLoader(train_data_list, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    val_loader = PyGDataLoader(val_data_list, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    test_loader = PyGDataLoader(test_data_list, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)

    # --- Initialize Model ---
    model = MultiModalMaterialClassifier(
        crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim=config.ASPH_FEATURE_DIM,
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
        
        # Pass all task output dimensions
        num_topology_classes=config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes=config.NUM_MAGNETISM_CLASSES,
        num_combined_classes=config.NUM_COMBINED_CLASSES, # Keep this for the primary task
        
        egnn_hidden_irreps_str=config.EGNN_HIDDEN_IRREPS_STR,
        egnn_num_layers=config.GNN_NUM_LAYERS, 
        egnn_radius=config.EGNN_RADIUS, 

        kspace_gnn_hidden_channels=config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers=config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads=config.KSPACE_GNN_NUM_HEADS, 

        ffnn_hidden_dims_asph=config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar=config.FFNN_HIDDEN_DIMS_SCALAR,
        
        latent_dim_gnn=config.LATENT_DIM_GNN,
        latent_dim_asph=config.LATENT_DIM_ASPH,
        latent_dim_other_ffnn=config.LATENT_DIM_OTHER_FFNN,
        fusion_hidden_dims=config.FUSION_HIDDEN_DIMS, 

        crystal_encoder_hidden_dim=config.crystal_encoder_hidden_dim, # From config
        crystal_encoder_num_layers=config.crystal_encoder_num_layers,
        crystal_encoder_output_dim=config.crystal_encoder_output_dim,
        crystal_encoder_radius=config.crystal_encoder_radius,
        crystal_encoder_num_scales=config.crystal_encoder_num_scales,
        crystal_encoder_use_topological_features=config.crystal_encoder_use_topological_features
    ).to(config.DEVICE)

    print(f"Model instantiated successfully: \n{model}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Calculate Class Weights (for CrossEntropyLoss) ---
    # These weights are for the CrossEntropyLoss component of EnhancedTopologicalLoss
    # and the standalone Magnetism CrossEntropyLoss.
    
    # Combined task (Primary) class weights
    train_combined_labels = [data['combined_label'].item() for data in train_data_list]
    combined_class_counts = Counter(train_combined_labels)
    total_combined_samples = sum(combined_class_counts.values())
    combined_num_classes = config.NUM_COMBINED_CLASSES

    combined_class_weights_raw = torch.zeros(combined_num_classes, dtype=torch.float32)
    print("\n--- Combined Class Distribution (Training Set) ---")
    for i in range(combined_num_classes):
        class_name_tuple = None # Get string name for readability
        for k, v in config.COMBINED_CLASS_MAPPING.items():
            if v == i:
                class_name_tuple = k
                break
        count = combined_class_counts.get(i, 0)
        print(f"Class {i} ('{class_name_tuple}'): {count} samples")
        if count > 0:
            combined_class_weights_raw[i] = total_combined_samples / (count * combined_num_classes)
        else:
            combined_class_weights_raw[i] = 1.0 # Give a default weight for unseen classes to avoid div by zero issues
    combined_class_weights = combined_class_weights_raw / combined_class_weights_raw.sum() * combined_num_classes
    print(f"Calculated Combined Class Weights: {combined_class_weights.tolist()}")
    print("---------------------------------------------------\n")

    # Topology task (Auxiliary) class weights
    train_topology_labels = [data['topology_label'].item() for data in train_data_list]
    topology_class_counts = Counter(train_topology_labels)
    total_topology_samples = sum(topology_class_counts.values())
    topology_num_classes = config.NUM_TOPOLOGY_CLASSES

    topology_class_weights_raw = torch.zeros(topology_num_classes, dtype=torch.float32)
    print("\n--- Topology Class Distribution (Training Set) ---")
    for i in range(topology_num_classes):
        class_name = None
        for name, idx in config.TOPOLOGY_CLASS_MAPPING.items():
            if idx == i:
                class_name = name
                break
        count = topology_class_counts.get(i, 0)
        print(f"Class {i} ('{class_name}'): {count} samples")
        if count > 0:
            topology_class_weights_raw[i] = total_topology_samples / (count * topology_num_classes)
        else:
            topology_class_weights_raw[i] = 1.0 # Default for unseen classes
    topology_class_weights = topology_class_weights_raw / topology_class_weights_raw.sum() * topology_num_classes
    print(f"Calculated Topology Class Weights: {topology_class_weights.tolist()}")
    print("---------------------------------------------------\n")

    # Magnetism task (Auxiliary) class weights
    train_magnetism_labels = [data['magnetism_label'].item() for data in train_data_list]
    magnetism_class_counts = Counter(train_magnetism_labels)
    total_magnetism_samples = sum(magnetism_class_counts.values())
    magnetism_num_classes = config.NUM_MAGNETISM_CLASSES

    magnetism_class_weights_raw = torch.zeros(magnetism_num_classes, dtype=torch.float32)
    print("--- Magnetism Class Distribution (Training Set) ---")
    for i in range(magnetism_num_classes):
        class_name = None
        for name, idx in config.MAGNETISM_CLASS_MAPPING.items():
            if idx == i:
                class_name = name
                break
        count = magnetism_class_counts.get(i, 0)
        print(f"Class {i} ('{class_name}'): {count} samples")
        if count > 0:
            magnetism_class_weights_raw[i] = total_magnetism_samples / (count * magnetism_num_classes)
        else:
            magnetism_class_weights_raw[i] = 1.0 # Default for unseen classes
    magnetism_class_weights = magnetism_class_weights_raw / magnetism_class_weights_raw.sum() * magnetism_num_classes
    print(f"Calculated Magnetism Class Weights: {magnetism_class_weights.tolist()}")
    print("---------------------------------------------------\n")


    # --- Loss Functions ---
    # Primary loss for the main 6-class combined task
    criterion_combined = nn.CrossEntropyLoss(weight=combined_class_weights.to(config.DEVICE)).to(config.DEVICE)
    
    # Auxiliary topology loss with topological regularization
    criterion_topology_aux = EnhancedTopologicalLoss(
        alpha=1.0, # This is the weight for classification loss within EnhancedTopologicalLoss
        beta=config.LOSS_WEIGHT_TOPO_CONSISTENCY,
        gamma=config.LOSS_WEIGHT_REGULARIZATION
    ).to(config.DEVICE)
    # Set the classification loss for the auxiliary topology task, with weights
    criterion_topology_aux.classification_loss = nn.CrossEntropyLoss(
        weight=topology_class_weights.to(config.DEVICE)
    )

    # Auxiliary magnetism loss
    criterion_magnetism_aux = nn.CrossEntropyLoss(
        weight=magnetism_class_weights.to(config.DEVICE)
    ).to(config.DEVICE)


    # --- Optimizer & Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_DECAY_STEP, gamma=config.LR_DECAY_FACTOR)

    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training loop...")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            # Move data to device (data from custom_collate_fn is already on device, but PyG batches might need it)
            batch['crystal_graph'] = batch['crystal_graph'].to(config.DEVICE)
            batch['kspace_graph'] = batch['kspace_graph'].to(config.DEVICE)
            
            # Ensure other tensors are on the correct device if not already set by collate_fn
            for key in ['asph_features', 'scalar_features', 'topology_label', 'magnetism_label', 'combined_label']:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.DEVICE)
            
            # Move kspace_physics_features sub-tensors to device
            if 'kspace_physics_features' in batch and isinstance(batch['kspace_physics_features'], dict):
                for sub_key in batch['kspace_physics_features']: 
                    if isinstance(batch['kspace_physics_features'][sub_key], torch.Tensor):
                        batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 
            else:
                 warnings.warn("kspace_physics_features not found or not a dict in batch. Using default values for model_inputs (may cause errors if not handled by model).")
                 # If kspace_physics_features is missing, provide dummy zeros to model to avoid errors
                 batch['kspace_physics_features'] = {
                     'decomposition_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DECOMPOSITION_FEATURE_DIM, device=config.DEVICE),
                     'gap_features': torch.zeros(batch['crystal_graph'].num_graphs, config.BAND_GAP_SCALAR_DIM, device=config.DEVICE),
                     'dos_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DOS_FEATURE_DIM, device=config.DEVICE),
                     'fermi_features': torch.zeros(batch['crystal_graph'].num_graphs, config.FERMI_FEATURE_DIM, device=config.DEVICE),
                 }

            optimizer.zero_grad()
            
            model_inputs = {
                'crystal_graph': batch['crystal_graph'],
                'kspace_graph': batch['kspace_graph'],
                'asph_features': batch['asph_features'],
                'scalar_features': batch['scalar_features'],
                'kspace_physics_features': batch['kspace_physics_features']
            }

            outputs = model(model_inputs)
            
            # Retrieve all logits and topological features from model outputs
            combined_logits = outputs['combined_logits']
            topology_logits_aux = outputs['topology_logits_aux']
            magnetism_logits_aux = outputs['magnetism_logits_aux']
            topological_features = outputs.get('topological_features', None) # Get, in case crystal encoder doesn't return it

            # Calculate individual loss components
            loss_combined = criterion_combined(combined_logits, batch['combined_label'])
            loss_topology_aux = criterion_topology_aux(
                topology_logits_aux,
                batch['topology_label'],
                topological_features=topological_features # Pass topo features to EnhancedTopologicalLoss
            )
            loss_magnetism_aux = criterion_magnetism_aux(magnetism_logits_aux, batch['magnetism_label'])

            # Combine all losses with their respective weights from config
            total_loss = (
                config.LOSS_WEIGHT_PRIMARY_COMBINED * loss_combined +
                config.LOSS_WEIGHT_AUX_TOPOLOGY * loss_topology_aux + # EnhancedTopologicalLoss includes its own beta/gamma
                config.LOSS_WEIGHT_AUX_MAGNETISM * loss_magnetism_aux
                # The topological regularization part is now internal to loss_topology_aux (EnhancedTopologicalLoss)
            )
            
            total_train_loss += total_loss.item()

            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
            
            optimizer.step()
            # if scheduler: scheduler.step() # Uncomment if you use a scheduler

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        all_val_combined_preds = []
        all_val_combined_labels = []
        all_val_topo_preds = []
        all_val_topo_labels = []
        all_val_mag_preds = []
        all_val_mag_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                # Move data to device (similar to train loop)
                batch['crystal_graph'] = batch['crystal_graph'].to(config.DEVICE)
                batch['kspace_graph'] = batch['kspace_graph'].to(config.DEVICE)
                for key in ['asph_features', 'scalar_features', 'topology_label', 'magnetism_label', 'combined_label']:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(config.DEVICE)
                if 'kspace_physics_features' in batch and isinstance(batch['kspace_physics_features'], dict):
                    for sub_key in batch['kspace_physics_features']: 
                        if isinstance(batch['kspace_physics_features'][sub_key], torch.Tensor):
                            batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 
                else:
                    warnings.warn("kspace_physics_features not found or not a dict in batch during validation.")
                    batch['kspace_physics_features'] = { # Provide dummy if missing for consistent model input
                        'decomposition_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DECOMPOSITION_FEATURE_DIM, device=config.DEVICE),
                        'gap_features': torch.zeros(batch['crystal_graph'].num_graphs, config.BAND_GAP_SCALAR_DIM, device=config.DEVICE),
                        'dos_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DOS_FEATURE_DIM, device=config.DEVICE),
                        'fermi_features': torch.zeros(batch['crystal_graph'].num_graphs, config.FERMI_FEATURE_DIM, device=config.DEVICE),
                    }
                
                model_inputs = {
                    'crystal_graph': batch['crystal_graph'],
                    'kspace_graph': batch['kspace_graph'],
                    'asph_features': batch['asph_features'],
                    'scalar_features': batch['scalar_features'],
                    'kspace_physics_features': batch['kspace_physics_features']
                }

                outputs = model(model_inputs)
                
                combined_logits = outputs['combined_logits']
                topology_logits_aux = outputs['topology_logits_aux']
                magnetism_logits_aux = outputs['magnetism_logits_aux']
                topological_features = outputs.get('topological_features', None)

                loss_combined = criterion_combined(combined_logits, batch['combined_label'])
                loss_topology_aux = criterion_topology_aux(
                    topology_logits_aux,
                    batch['topology_label'],
                    topological_features=topological_features
                )
                loss_magnetism_aux = criterion_magnetism_aux(magnetism_logits_aux, batch['magnetism_label'])
                
                val_loss = (
                    config.LOSS_WEIGHT_PRIMARY_COMBINED * loss_combined +
                    config.LOSS_WEIGHT_AUX_TOPOLOGY * loss_topology_aux +
                    config.LOSS_WEIGHT_AUX_MAGNETISM * loss_magnetism_aux
                )
                total_val_loss += val_loss.item()

                _, predicted_combined = torch.max(combined_logits, 1)
                all_val_combined_preds.extend(predicted_combined.cpu().numpy())
                all_val_combined_labels.extend(batch['combined_label'].cpu().numpy())

                _, predicted_topo_aux = torch.max(topology_logits_aux, 1)
                all_val_topo_preds.extend(predicted_topo_aux.cpu().numpy())
                all_val_topo_labels.extend(batch['topology_label'].cpu().numpy())

                _, predicted_mag_aux = torch.max(magnetism_logits_aux, 1)
                all_val_mag_preds.extend(predicted_mag_aux.cpu().numpy())
                all_val_mag_labels.extend(batch['magnetism_label'].cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")

        _ = compute_metrics(torch.tensor(all_val_combined_preds), torch.tensor(all_val_combined_labels), config.NUM_COMBINED_CLASSES, "Combined Classification (Validation)")
        _ = compute_metrics(torch.tensor(all_val_topo_preds), torch.tensor(all_val_topo_labels), config.NUM_TOPOLOGY_CLASSES, "Topology Classification (Validation)")
        _ = compute_metrics(torch.tensor(all_val_mag_preds), torch.tensor(all_val_mag_labels), config.NUM_MAGNETISM_CLASSES, "Magnetism Classification (Validation)")

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
    # Load the best model's state_dict before testing
    if (config.MODEL_SAVE_DIR / "best_multi_task_classifier.pth").exists():
        model.load_state_dict(torch.load(config.MODEL_SAVE_DIR / "best_multi_task_classifier.pth", map_location=config.DEVICE))
    else:
        warnings.warn(f"Best model not found at {config.MODEL_SAVE_DIR / 'best_multi_task_classifier.pth'}. Testing with current model state.")

    model.eval()

    all_test_combined_preds = []
    all_test_combined_labels = []
    all_test_topo_preds = []
    all_test_topo_labels = []
    all_test_mag_preds = []
    all_test_mag_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation"):
            batch['crystal_graph'] = batch['crystal_graph'].to(config.DEVICE)
            batch['kspace_graph'] = batch['kspace_graph'].to(config.DEVICE)
            for key in ['asph_features', 'scalar_features', 'topology_label', 'magnetism_label', 'combined_label']:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.DEVICE)
            if 'kspace_physics_features' in batch and isinstance(batch['kspace_physics_features'], dict):
                for sub_key in batch['kspace_physics_features']: 
                    if isinstance(batch['kspace_physics_features'][sub_key], torch.Tensor):
                        batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 
            else:
                batch['kspace_physics_features'] = { # Provide dummy if missing for consistent model input
                    'decomposition_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DECOMPOSITION_FEATURE_DIM, device=config.DEVICE),
                    'gap_features': torch.zeros(batch['crystal_graph'].num_graphs, config.BAND_GAP_SCALAR_DIM, device=config.DEVICE),
                    'dos_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DOS_FEATURE_DIM, device=config.DEVICE),
                    'fermi_features': torch.zeros(batch['crystal_graph'].num_graphs, config.FERMI_FEATURE_DIM, device=config.DEVICE),
                }

            model_inputs = {
                'crystal_graph': batch['crystal_graph'],
                'kspace_graph': batch['kspace_graph'],
                'asph_features': batch['asph_features'],
                'scalar_features': batch['scalar_features'],
                'kspace_physics_features': batch['kspace_physics_features']
            }
            outputs = model(model_inputs)
            
            combined_logits = outputs['combined_logits']
            topology_logits_aux = outputs['topology_logits_aux']
            magnetism_logits_aux = outputs['magnetism_logits_aux']

            _, predicted_combined = torch.max(combined_logits, 1)
            all_test_combined_preds.extend(predicted_combined.cpu().numpy())
            all_test_combined_labels.extend(batch['combined_label'].cpu().numpy())

            _, predicted_topo_aux = torch.max(topology_logits_aux, 1)
            all_test_topo_preds.extend(predicted_topo_aux.cpu().numpy())
            all_test_topo_labels.extend(batch['topology_label'].cpu().numpy())

            _, predicted_mag_aux = torch.max(magnetism_logits_aux, 1)
            all_test_mag_preds.extend(predicted_mag_aux.cpu().numpy())
            all_test_mag_labels.extend(batch['magnetism_label'].cpu().numpy())

    print("\nTest Set Results:")
    _ = compute_metrics(torch.tensor(all_test_combined_preds), torch.tensor(all_test_combined_labels), config.NUM_COMBINED_CLASSES, "Combined Classification")
    _ = compute_metrics(torch.tensor(all_test_topo_preds), torch.tensor(all_test_topo_labels), config.NUM_TOPOLOGY_CLASSES, "Topology Classification")
    _ = compute_metrics(torch.tensor(all_test_mag_preds), torch.tensor(all_test_mag_labels), config.NUM_MAGNETISM_CLASSES, "Magnetism Classification")

def main_training_loop(): 
    train_main_classifier()

# Example of how to run the training function
if __name__ == "__main__":
    # --- Dummy Data setup for local execution ---
    # These paths are set up to match the hardcoded paths in your MaterialDataset
    # For a real run, ensure your actual data is at these locations
    dummy_data_root = Path("./dummy_multimodal_db")
    dummy_master_index_path = dummy_data_root / "metadata"
    dummy_kspace_graphs_base_dir = dummy_data_root / "kspace_graphs"
    # IMPORTANT: These paths below are hardcoded in MaterialDataset, ensure they exist or change in MaterialDataset.
    dummy_crystal_graphs_base_dir_scratch = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/crystal_graphs")
    dummy_vectorized_features_base_dir_scratch = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/vectorized_features")
    
    # Create necessary dummy directories
    dummy_master_index_path.mkdir(parents=True, exist_ok=True)
    dummy_kspace_graphs_base_dir.mkdir(parents=True, exist_ok=True)
    dummy_crystal_graphs_base_dir_scratch.mkdir(parents=True, exist_ok=True)
    dummy_vectorized_features_base_dir_scratch.mkdir(parents=True, exist_ok=True)

    # Create dummy JSON metadata files and corresponding feature files for a minimal run
    # Needs at least one 'Trivial', 'NM' and one 'Topological Insulator', 'NM'
    # and one 'Semimetal', 'FM' and one 'Trivial', 'AFM' to have some class diversity
    # in the dummy data for the classifiers.
    dummy_json_data = [
        {'jid': 'mat_001', 'formula': 'GaAs', 'space_group': 'F-43m', 'space_group_number': 216,
         'topological_class': 'Trivial', 'magnetic_type': 'NM', 'band_gap': 1.5,
         'formation_energy': -0.8, 'energy_above_hull': 0.0, 'density': 5.32, 'volume': 45.1, 'nsites': 2, 'total_magnetization': 0.0, 'theoretical': True},
        {'jid': 'mat_002', 'formula': 'Bi2Se3', 'space_group': 'R-3m', 'space_group_number': 166,
         'topological_class': 'Topological Insulator', 'magnetic_type': 'NM', 'band_gap': 0.1,
         'formation_energy': -0.5, 'energy_above_hull': 0.0, 'density': 6.82, 'volume': 120.5, 'nsites': 5, 'total_magnetization': 0.0, 'theoretical': True},
        {'jid': 'mat_003', 'formula': 'MnBi', 'space_group': 'P6_3/mmc', 'space_group_number': 194,
         'topological_class': 'Semimetal', 'magnetic_type': 'FM', 'band_gap': 0.0,
         'formation_energy': -1.0, 'energy_above_hull': 0.0, 'density': 8.0, 'volume': 70.0, 'nsites': 2, 'total_magnetization': 2.2, 'theoretical': True},
        {'jid': 'mat_004', 'formula': 'FeSi', 'space_group': 'P2_13', 'space_group_number': 198,
         'topological_class': 'Trivial', 'magnetic_type': 'AFM', 'band_gap': 1.2,
         'formation_energy': -0.9, 'energy_above_hull': 0.0, 'density': 6.18, 'volume': 60.0, 'nsites': 2, 'total_magnetization': 0.5, 'theoretical': True},
         # Add more diverse data to ensure all classes are present if possible
         {'jid': 'mat_005', 'formula': 'Cd3As2', 'space_group': 'I4_1cd', 'space_group_number': 110,
          'topological_class': 'Semimetal', 'magnetic_type': 'NM', 'band_gap': 0.0,
          'formation_energy': -0.6, 'energy_above_hull': 0.0, 'density': 6.2, 'volume': 150.0, 'nsites': 5, 'total_magnetization': 0.0, 'theoretical': True},
         {'jid': 'mat_006', 'formula': 'Cr2Ge2Te6', 'space_group': 'R-3m', 'space_group_number': 166,
          'topological_class': 'Topological Insulator', 'magnetic_type': 'FM', 'band_gap': 0.2,
          'formation_energy': -0.4, 'energy_above_hull': 0.0, 'density': 5.0, 'volume': 200.0, 'nsites': 10, 'total_magnetization': 1.8, 'theoretical': True},
    ]

    # Dynamically update config based on dummy data requirements
    # This is important for the model to correctly initialize.
    if not hasattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM'): config.BASE_DECOMPOSITION_FEATURE_DIM = 2
    if not hasattr(config, 'ALL_POSSIBLE_IRREPS'): config.ALL_POSSIBLE_IRREPS = ['A1', 'E1', 'T2'] # minimal set
    if not hasattr(config, 'MAX_DECOMPOSITION_INDICES_LEN'): config.MAX_DECOMPOSITION_INDICES_LEN = 5
    # Calculate _expected_decomposition_feature_dim and then set config.DECOMPOSITION_FEATURE_DIM
    _expected_decomp_dim = config.BASE_DECOMPOSITION_FEATURE_DIM + len(config.ALL_POSSIBLE_IRREPS) + config.MAX_DECOMPOSITION_INDICES_LEN
    config.DECOMPOSITION_FEATURE_DIM = _expected_decomp_dim

    # Set some initial dummy values for dynamic config attributes,
    # as MaterialDataset's __getitem__ will try to set them.
    config.CRYSTAL_NODE_FEATURE_DIM = 3 # Dummy, will be inferred by dataset
    config.KSPACE_GRAPH_NODE_FEATURE_DIM = 10 # Dummy
    config.ASPH_FEATURE_DIM = 3115 # Dummy
    config.SCALAR_TOTAL_DIM = 4756 + 7 # Dummy, band_rep_dim + scalar_metadata_dim (7 from MaterialDataset)
    config.BAND_REP_FEATURE_DIM = 4756 # Dummy
    config.BAND_GAP_SCALAR_DIM = 1 # Dummy
    config.DOS_FEATURE_DIM = 100 # Dummy
    config.FERMI_FEATURE_DIM = 1 # Dummy

    for data in tqdm(dummy_json_data, desc="Creating dummy data files"):
        # Create metadata JSON
        with open(dummy_master_index_path / f"{data['jid']}.json", 'w') as f:
            json.dump(data, f)
        
        # Create dummy crystal_graph.pkl
        (dummy_crystal_graphs_base_dir_scratch / data['jid']).mkdir(parents=True, exist_ok=True)
        with open(dummy_crystal_graphs_base_dir_scratch / data['jid'] / "crystal_graph.pkl", 'wb') as f:
            pickle.dump({'x': np.random.rand(10, config.CRYSTAL_NODE_FEATURE_DIM), 
                         'pos': np.random.rand(10, 3), 
                         'edge_index': np.random.randint(0, 10, (2, 20))}, f)
        
        # Create dummy asph_features_rev2.npy and band_rep_features.npy
        (dummy_vectorized_features_base_dir_scratch / data['jid']).mkdir(parents=True, exist_ok=True)
        np.save(dummy_vectorized_features_base_dir_scratch / data['jid'] / "asph_features_rev2.npy", np.random.rand(config.ASPH_FEATURE_DIM))
        np.save(dummy_vectorized_features_base_dir_scratch / data['jid'] / "band_rep_features.npy", np.random.rand(config.BAND_REP_FEATURE_DIM))
        
        # Create dummy kspace graph related files
        sg_folder = dummy_kspace_graphs_base_dir / f"SG_{str(int(data['space_group_number'])).zfill(3)}"
        sg_folder.mkdir(parents=True, exist_ok=True)
        
        dummy_kspace_graph_pyg = PyGData(x=torch.randn(5, config.KSPACE_GRAPH_NODE_FEATURE_DIM), 
                                         edge_index=torch.randint(0, 5, (2, 8)),
                                         pos=torch.randn(5,3)) # Ensure pos for dummy
        torch.save(dummy_kspace_graph_pyg, sg_folder / "kspace_graph.pt")
        
        torch.save({'decomposition_features': torch.randn(config.BASE_DECOMPOSITION_FEATURE_DIM)}, sg_folder / "physics_features.pt")
        
        with open(sg_folder / "metadata.json", 'w') as f:
            json.dump({
                "ebr_data": {"irrep_multiplicities": {irrep: np.random.randint(1,5) for irrep in config.ALL_POSSIBLE_IRREPS[:2]}},
                "decomposition_branches": {"decomposition_indices": [np.random.randint(0,10) for _ in range(config.MAX_DECOMPOSITION_INDICES_LEN)]}
            }, f)

    # Now, run the training
    main_training_loop()

    # Clean up dummy directories and files
    if dummy_data_root.exists():
        shutil.rmtree(dummy_data_root)
        print(f"Cleaned up {dummy_data_root}")
    if dummy_crystal_graphs_base_dir_scratch.exists():
        shutil.rmtree(dummy_crystal_graphs_base_dir_scratch)
        print(f"Cleaned up {dummy_crystal_graphs_base_dir_scratch}")
    if dummy_vectorized_features_base_dir_scratch.exists():
        shutil.rmtree(dummy_vectorized_features_base_dir_scratch)
        print(f"Cleaned up {dummy_vectorized_features_base_dir_scratch}")
    if config.MODEL_SAVE_DIR.exists():
        shutil.rmtree(config.MODEL_SAVE_DIR)
        print(f"Cleaned up {config.MODEL_SAVE_DIR}")

