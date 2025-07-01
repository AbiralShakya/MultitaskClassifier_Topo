import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
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
from collections import Counter # Import Counter for class weight calculation

# Import from local modules
from helper import config
from helper.dataset import MaterialDataset, custom_collate_fn
from src.model import MultiModalMaterialClassifier 
from helper.enhanced_topological_loss import EnhancedTopologicalLoss 

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
])

# --- Drop-in Focal Loss implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure alpha weights are on the same device as inputs
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
        else:
            alpha = None
            
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_metrics(predictions, targets, num_classes, task_name, class_names=None):
    """Computes accuracy and detailed classification report."""
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    acc = accuracy_score(targets_np, preds_np)
    labels = list(range(num_classes))
    
    # Generate class_names if not provided for better report readability
    if class_names is None:
        if task_name == "Topology Classification":
            class_names = [config.TOPOLOGY_INT_TO_CANONICAL_STR.get(i, f"Class {i}") for i in labels]
        elif task_name == "Magnetism Classification":
            class_names = [config.MAGNETISM_INT_TO_CANONICAL_STR.get(i, f"Class {i}") for i in labels]
        elif task_name == "Combined Classification":
            # This is more complex, just use default labels or map them out if needed
            class_names = [f"Class {i}" for i in labels] # You might want to map these to ("Trivial", "NM") etc.

    report = classification_report(targets_np, preds_np, output_dict=True, zero_division=0, labels=labels, target_names=class_names)
    cm = confusion_matrix(targets_np, preds_np, labels=labels)

    print(f"\n--- {task_name} Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Classification Report:\n{json.dumps(report, indent=2)}")
    print(f"Confusion Matrix:\n{cm}")
    
    return acc, report, cm

def train_model(model, train_loader, val_loader, test_loader, 
                topology_class_weights, magnetism_class_weights, combined_class_weights, 
                stage: int):
    
    print(f"\n--- Starting Training Stage {stage} ---")
    
    # --- Balanced Class Weights for Topological Insulator ---
    # Use more moderate weights to avoid overfitting to TI
    topology_class_weights = topology_class_weights.clone()
    # Boost TI weight by 1.5x instead of 2x to avoid extreme overfitting
    topology_class_weights[1] *= 1.5
    print(f"Topology class weights (with moderate TI boost): {topology_class_weights}")

    # --- Loss Function Selection ---
    use_focal_loss = False  # Set to False to use regular CrossEntropyLoss with moderate weights
    if use_focal_loss:
        print("Using FocalLoss for topology classification (moderate alpha, gamma=2)")
        alpha = topology_class_weights.clone()
        # Use more moderate alpha - don't go above 2.5x to avoid overfitting
        alpha[1] = min(alpha[1], 2.5)  # Cap TI weight at 2.5x
        criterion_topology_aux = FocalLoss(alpha=alpha, gamma=2).to(config.DEVICE)
    else:
        print("Using CrossEntropyLoss with moderate class weights for topology classification")
        criterion_topology_aux = nn.CrossEntropyLoss(weight=topology_class_weights.to(config.DEVICE))

    criterion_combined = nn.CrossEntropyLoss(weight=combined_class_weights.to(config.DEVICE))
    criterion_magnetism_aux = nn.CrossEntropyLoss(weight=magnetism_class_weights.to(config.DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch_geometric.data.Batch) or isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.DEVICE)
                elif isinstance(batch[key], dict): # Handle kspace_physics_features
                    for sub_key in batch[key]:
                        if isinstance(batch[key][sub_key], torch.Tensor):
                            batch[key][sub_key] = batch[key][sub_key].to(config.DEVICE)

            optimizer.zero_grad()
            
            model_inputs = {
                'crystal_graph': batch['crystal_graph'],
                'kspace_graph': batch['kspace_graph'],
                'asph_features': batch['asph_features'],
                'scalar_features': batch['scalar_features'],
                'kspace_physics_features': batch['kspace_physics_features']
            }

            outputs = model(model_inputs)
            
            # Initialize losses for the current batch
            loss_combined = torch.tensor(0.0).to(config.DEVICE)
            loss_topology_aux = torch.tensor(0.0).to(config.DEVICE)
            loss_magnetism_aux = torch.tensor(0.0).to(config.DEVICE)

            # Calculate losses based on the current stage
            if stage == 1: # Only auxiliary tasks
                loss_topology_aux = criterion_topology_aux(
                    outputs['topology_logits_aux'], 
                    batch['topology_label']
                )
                loss_magnetism_aux = criterion_magnetism_aux(outputs['magnetism_logits_aux'], batch['magnetism_label'])
                total_loss = loss_topology_aux + loss_magnetism_aux
                
            elif stage == 2: # All tasks with weighted sum
                loss_combined = criterion_combined(outputs['combined_logits'], batch['combined_label'])
                loss_topology_aux = criterion_topology_aux(
                    outputs['topology_logits_aux'], 
                    batch['topology_label']
                )
                loss_magnetism_aux = criterion_magnetism_aux(outputs['magnetism_logits_aux'], batch['magnetism_label'])
                
                total_loss = (config.LOSS_WEIGHT_PRIMARY_COMBINED * loss_combined +
                              config.LOSS_WEIGHT_AUX_TOPOLOGY * loss_topology_aux +
                              config.LOSS_WEIGHT_AUX_MAGNETISM * loss_magnetism_aux)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
            optimizer.step()
            total_train_loss += total_loss.item()
            
            if batch_idx % 100 == 0: # Print less frequently
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {total_loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        all_combined_preds = []
        all_combined_targets = []
        all_topo_preds = []
        all_topo_targets = []
        all_magnetism_preds = []
        all_magnetism_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch_geometric.data.Batch) or isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(config.DEVICE)
                    elif isinstance(batch[key], dict): # Handle kspace_physics_features
                        for sub_key in batch[key]:
                            if isinstance(batch[key][sub_key], torch.Tensor):
                                batch[key][sub_key] = batch[key][sub_key].to(config.DEVICE)
                
                model_inputs = {
                    'crystal_graph': batch['crystal_graph'],
                    'kspace_graph': batch['kspace_graph'],
                    'asph_features': batch['asph_features'],
                    'scalar_features': batch['scalar_features'],
                    'kspace_physics_features': batch['kspace_physics_features']
                }

                outputs = model(model_inputs)

                # Initialize losses for validation
                loss_combined = torch.tensor(0.0).to(config.DEVICE)
                loss_topology_aux = torch.tensor(0.0).to(config.DEVICE)
                loss_magnetism_aux = torch.tensor(0.0).to(config.DEVICE)

                if stage == 1:
                    loss_topology_aux = criterion_topology_aux(
                        outputs['topology_logits_aux'], 
                        batch['topology_label']
                    )
                    loss_magnetism_aux = criterion_magnetism_aux(outputs['magnetism_logits_aux'], batch['magnetism_label'])
                    total_loss = loss_topology_aux + loss_magnetism_aux
                elif stage == 2:
                    loss_combined = criterion_combined(outputs['combined_logits'], batch['combined_label'])
                    loss_topology_aux = criterion_topology_aux(
                        outputs['topology_logits_aux'], 
                        batch['topology_label']
                    )
                    loss_magnetism_aux = criterion_magnetism_aux(outputs['magnetism_logits_aux'], batch['magnetism_label'])
                    
                    total_loss = (config.LOSS_WEIGHT_PRIMARY_COMBINED * loss_combined +
                                  config.LOSS_WEIGHT_AUX_TOPOLOGY * loss_topology_aux +
                                  config.LOSS_WEIGHT_AUX_MAGNETISM * loss_magnetism_aux)
                
                total_val_loss += total_loss.item()

                if stage == 2: # Only collect combined preds/targets in Stage 2
                    all_combined_preds.extend(torch.argmax(outputs['combined_logits'], dim=1).cpu().numpy())
                    all_combined_targets.extend(batch['combined_label'].cpu().numpy())
                
                all_topo_preds.extend(torch.argmax(outputs['topology_logits_aux'], dim=1).cpu().numpy())
                all_topo_targets.extend(batch['topology_label'].cpu().numpy())
                all_magnetism_preds.extend(torch.argmax(outputs['magnetism_logits_aux'], dim=1).cpu().numpy())
                all_magnetism_targets.extend(batch['magnetism_label'].cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} (Stage {stage}):")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")

        if stage == 2: # Report combined metrics only in Stage 2
            _ = compute_metrics(torch.tensor(all_combined_preds), torch.tensor(all_combined_targets), 
                                config.NUM_COMBINED_CLASSES, "Combined Classification")
        
        _ = compute_metrics(torch.tensor(all_topo_preds), torch.tensor(all_topo_targets), 
                            config.NUM_TOPOLOGY_CLASSES, "Topology Classification")
        _ = compute_metrics(torch.tensor(all_magnetism_preds), torch.tensor(all_magnetism_targets), 
                            config.NUM_MAGNETISM_CLASSES, "Magnetism Classification")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_save_path = config.MODEL_SAVE_DIR / f"best_multi_task_classifier_stage_{stage}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"  Model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement in Stage {stage}.")
                break
    
    return model # Return the trained model (or its state_dict path)

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
    
    # Instantiate the dataset for dimension inference and splitting
    # Pass dos_fermi_dir to MaterialDataset
    full_dataset_for_dim_inference = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR, # Pass the new directory
        scaler=None
    )
    
    try:
        # Load a dummy sample to infer dimensions and populate config (if 0 or None)
        _ = full_dataset_for_dim_inference[0] 
        print("Dataset dimensions inferred and config updated based on first sample.")
    except Exception as e:
        print(f"Warning: Could not load first sample to infer dimensions: {e}")
        print("Please ensure config.py has accurate feature dimensions set manually.")

    print("Environment setup complete.")

    print("Fitting Scalers and splitting data...")
    dataset_indices = list(range(len(full_dataset_for_dim_inference)))
    
    # Temporarily get labels for initial stratification
    temp_labels_for_stratification = [full_dataset_for_dim_inference[i]['topology_label'].item() for i in dataset_indices] # Use topology for initial coarse split

    # Split indices (before full data loading to save memory/time)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        dataset_indices, temp_labels_for_stratification, 
        train_size=config.TRAIN_RATIO, 
        random_state=config.SEED,
        stratify=temp_labels_for_stratification
    )
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels, 
        train_size=config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.SEED,
        stratify=temp_labels
    )

    # Create temporary datasets for scaler fitting and label collection
    temp_train_dataset_for_scaler = torch.utils.data.Subset(full_dataset_for_dim_inference, train_indices)
    
    all_asph_features_np = []
    all_scalar_features_np = []
    all_decomp_features_np = [] # For decomposition features
    all_gap_features_np = []    # For band gap features
    all_dos_features_np = []    # For DOS features
    all_fermi_features_np = []  # For Fermi features

    train_topology_labels = []
    train_magnetism_labels = []
    train_combined_labels = [] # Will be dynamically generated in custom_collate_fn

    temp_train_loader_for_scaler = PyGDataLoader( # Use PyGDataLoader with custom_collate_fn
        temp_train_dataset_for_scaler, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn, 
        num_workers=config.NUM_WORKERS
    )

    # Collect data for scaler fitting and class weight calculation
    for batch in temp_train_loader_for_scaler:
        if batch['asph_features'].numel() > 0:
            all_asph_features_np.append(batch['asph_features'].cpu().numpy())
        if batch['scalar_features'].numel() > 0:
            all_scalar_features_np.append(batch['scalar_features'].cpu().numpy())
        
        # Collect kspace_physics_features components
        if 'decomposition_features' in batch['kspace_physics_features'] and batch['kspace_physics_features']['decomposition_features'].numel() > 0:
            all_decomp_features_np.append(batch['kspace_physics_features']['decomposition_features'].cpu().numpy())
        if 'gap_features' in batch['kspace_physics_features'] and batch['kspace_physics_features']['gap_features'].numel() > 0:
            all_gap_features_np.append(batch['kspace_physics_features']['gap_features'].cpu().numpy())
        if 'dos_features' in batch['kspace_physics_features'] and batch['kspace_physics_features']['dos_features'].numel() > 0:
            all_dos_features_np.append(batch['kspace_physics_features']['dos_features'].cpu().numpy())
        if 'fermi_features' in batch['kspace_physics_features'] and batch['kspace_physics_features']['fermi_features'].numel() > 0:
            all_fermi_features_np.append(batch['kspace_physics_features']['fermi_features'].cpu().numpy())

        train_topology_labels.extend(batch['topology_label'].cpu().numpy())
        train_magnetism_labels.extend(batch['magnetism_label'].cpu().numpy())
        train_combined_labels.extend(batch['combined_label'].cpu().numpy())
    
    feature_scalers = {}
    if all_asph_features_np:
        feature_scalers['asph'] = StandardScaler()
        feature_scalers['asph'].fit(np.vstack(all_asph_features_np))
        print("ASPH StandardScaler fitted.")
    if all_scalar_features_np:
        feature_scalers['scalar'] = StandardScaler()
        feature_scalers['scalar'].fit(np.vstack(all_scalar_features_np))
        print("Scalar StandardScaler fitted.")
    if all_decomp_features_np:
        feature_scalers['decomp'] = StandardScaler()
        feature_scalers['decomp'].fit(np.vstack(all_decomp_features_np))
        print("Decomposition StandardScaler fitted.")
    if all_gap_features_np:
        feature_scalers['gap'] = StandardScaler()
        feature_scalers['gap'].fit(np.vstack(all_gap_features_np))
        print("Gap StandardScaler fitted.")
    if all_dos_features_np:
        feature_scalers['dos'] = StandardScaler()
        feature_scalers['dos'].fit(np.vstack(all_dos_features_np))
        print("DOS StandardScaler fitted.")
    if all_fermi_features_np:
        feature_scalers['fermi'] = StandardScaler()
        feature_scalers['fermi'].fit(np.vstack(all_fermi_features_np))
        print("Fermi StandardScaler fitted.")

    # --- Calculate Class Weights for all tasks ---
    def calculate_class_weights(labels, num_classes, mapping):
        class_counts = Counter(labels)
        total_samples = sum(class_counts.values())
        weights_raw = torch.zeros(num_classes, dtype=torch.float32)
        print(f"--- Class Distribution (Training Set) ---")
        for i in range(num_classes):
            class_name = None
            if mapping: # Try to get class name if a mapping is provided
                for name, idx in mapping.items():
                    if idx == i:
                        class_name = name
                        break
            count = class_counts.get(i, 0)
            print(f"Class {i} ('{class_name or i}'): {count} samples")
            if count > 0:
                weights_raw[i] = total_samples / (count * num_classes)
            else:
                weights_raw[i] = 100.0 # High weight for truly missing classes
        # Normalize to sum to num_classes
        weights = weights_raw / weights_raw.sum() * num_classes
        print(f"Calculated Class Weights: {weights.tolist()}")
        print("---------------------------------------------------\n")
        return weights

    # Calculate class weights for all three tasks (for class imbalance handling)
    topology_class_weights = calculate_class_weights(
        train_topology_labels, config.NUM_TOPOLOGY_CLASSES, config.TOPOLOGY_CLASS_MAPPING
    )
    magnetism_class_weights = calculate_class_weights(
        train_magnetism_labels, config.NUM_MAGNETISM_CLASSES, config.MAGNETISM_CLASS_MAPPING
    )
    combined_class_weights = calculate_class_weights(
        train_combined_labels, config.NUM_COMBINED_CLASSES, config.COMBINED_CLASS_MAPPING
    )

    final_full_dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR, 
        scaler=feature_scalers 
    )
    print("Final dataset re-initialized with fitted scalers.")

    # --- Moderate Oversampling for Topological Insulator ---
    print("Setting up WeightedRandomSampler for moderate oversampling of Topological Insulator...")
    # Use train_topology_labels from earlier in the function
    sample_weights = [2 if y == 1 else 1 for y in train_topology_labels]  # 2x oversample TI (reduced from 5x)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"Sample weights for oversampling: TI={sample_weights.count(2)}, others={sample_weights.count(1)}")

    # Create the subsets using the pre-calculated indices
    train_dataset = torch.utils.data.Subset(final_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(final_full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(final_full_dataset, test_indices)

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Use the sampler for the train_loader
    train_loader = PyGDataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    val_loader = PyGDataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    test_loader = PyGDataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)

    # Initialize model
    model = MultiModalMaterialClassifier(
        crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim=config.ASPH_FEATURE_DIM,
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM, # Used by EnhancedKSpacePhysicsFeatures
        
        num_topology_classes=config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes=config.NUM_MAGNETISM_CLASSES,
        num_combined_classes=config.NUM_COMBINED_CLASSES,

        crystal_encoder_hidden_dim=config.crystal_encoder_hidden_dim,
        crystal_encoder_num_layers=config.crystal_encoder_num_layers,
        crystal_encoder_output_dim=config.crystal_encoder_output_dim,
        crystal_encoder_radius=config.crystal_encoder_radius,
        crystal_encoder_num_scales=config.crystal_encoder_num_scales,
        crystal_encoder_use_topological_features=config.crystal_encoder_use_topological_features,

        kspace_gnn_hidden_channels=config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers=config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads=config.KSPACE_GNN_NUM_HEADS,
        
        ffnn_hidden_dims_asph=config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar=config.FFNN_HIDDEN_DIMS_SCALAR,
        
        latent_dim_gnn=config.LATENT_DIM_GNN,
        latent_dim_asph=config.LATENT_DIM_ASPH,
        latent_dim_other_ffnn=config.LATENT_DIM_OTHER_FFNN,
        
        fusion_hidden_dims=config.FUSION_HIDDEN_DIMS
    ).to(config.DEVICE)

    print(f"Model instantiated successfully: \n{model}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- STAGE 1: Train Auxiliary Tasks Only ---
    print("\nStarting Stage 1: Training Auxiliary Topology and Magnetism Tasks...")
    trained_model_stage_1 = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        topology_class_weights=topology_class_weights, 
        magnetism_class_weights=magnetism_class_weights, 
        combined_class_weights=combined_class_weights, # Still pass for consistency, but not used for loss in Stage 1
        stage=1
    )
    
    # Save checkpoint after Stage 1 (already handled within train_model)
    stage1_model_path = config.MODEL_SAVE_DIR / "best_multi_task_classifier_stage_1.pth"
    print(f"Stage 1 training finished. Best model saved to: {stage1_model_path}")

    # --- STAGE 2: Fine-tune for Combined Task ---
    print("\nStarting Stage 2: Fine-tuning for Combined Classification Task...")
    # Load the best model from Stage 1 to continue training
    model.load_state_dict(torch.load(stage1_model_path))
    print("Model weights loaded from Stage 1 for fine-tuning in Stage 2.")

    # You might want to adjust learning rate or freeze layers here for fine-tuning.
    # For now, we'll use the same LR.
    # If you want to freeze layers for a few epochs:
    # for param in model.crystal_encoder.parameters():
    #     param.requires_grad = False
    # ... then unfreeze them later or in a separate loop for fine-tuning.

    trained_model_stage_2 = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        topology_class_weights=topology_class_weights, 
        magnetism_class_weights=magnetism_class_weights, 
        combined_class_weights=combined_class_weights, 
        stage=2
    )

    # Final evaluation on the test set
    print("\n--- Final Evaluation on Test Set (after Stage 2) ---")
    trained_model_stage_2.eval()
    all_combined_preds_test = []
    all_combined_targets_test = []
    all_topo_preds_test = []
    all_topo_targets_test = []
    all_magnetism_preds_test = []
    all_magnetism_targets_test = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch_geometric.data.Batch) or isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.DEVICE)
                elif isinstance(batch[key], dict): # Handle kspace_physics_features
                    for sub_key in batch[key]:
                        if isinstance(batch[key][sub_key], torch.Tensor):
                            batch[key][sub_key] = batch[key][sub_key].to(config.DEVICE)
            
            model_inputs = {
                'crystal_graph': batch['crystal_graph'],
                'kspace_graph': batch['kspace_graph'],
                'asph_features': batch['asph_features'],
                'scalar_features': batch['scalar_features'],
                'kspace_physics_features': batch['kspace_physics_features']
            }
            outputs = model(model_inputs)

            all_combined_preds_test.extend(torch.argmax(outputs['combined_logits'], dim=1).cpu().numpy())
            all_combined_targets_test.extend(batch['combined_label'].cpu().numpy())
            all_topo_preds_test.extend(torch.argmax(outputs['topology_logits_aux'], dim=1).cpu().numpy())
            all_topo_targets_test.extend(batch['topology_label'].cpu().numpy())
            all_magnetism_preds_test.extend(torch.argmax(outputs['magnetism_logits_aux'], dim=1).cpu().numpy())
            all_magnetism_targets_test.extend(batch['magnetism_label'].cpu().numpy())

    print("\nFinal Test Set Results:")
    _ = compute_metrics(torch.tensor(all_combined_preds_test), torch.tensor(all_combined_targets_test), 
                        config.NUM_COMBINED_CLASSES, "Combined Classification")
    _ = compute_metrics(torch.tensor(all_topo_preds_test), torch.tensor(all_topo_targets_test), 
                        config.NUM_TOPOLOGY_CLASSES, "Topology Classification")
    _ = compute_metrics(torch.tensor(all_magnetism_preds_test), torch.tensor(all_magnetism_targets_test), 
                        config.NUM_MAGNETISM_CLASSES, "Magnetism Classification")


def main_training_loop(): 
    train_main_classifier()