# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torch_geometric.loader import DataLoader as PyGDataLoader
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import numpy as np
# import os
# from torch_geometric.data import Data as PyGData
# import json
# import pickle
# from pathlib import Path
# import warnings
# import random
# import torch_geometric
# import shutil
# from collections import Counter
# from tqdm import tqdm

# # Import from local modules
# from src.model import MultiModalMaterialClassifier
# import helper.config as config
# from helper.enhanced_topological_loss import EnhancedTopologicalLoss
# from helper.dataset import MaterialDataset, custom_collate_fn 
# # Make sure these are correctly imported from your helper/data_processing.py
# from helper.data_processing import ImprovedDataPreprocessor, StratifiedDataSplitter 

# # Ensure PyG's global settings are safe for serialization
# torch.serialization.add_safe_globals([
#     torch_geometric.data.data.DataEdgeAttr,
#     torch_geometric.data.data.DataTensorAttr,
#     torch_geometric.data.storage.GlobalStorage
# ])

# def compute_metrics(predictions, targets, num_classes, task_name):
#     """Computes accuracy and detailed classification report."""
#     if len(predictions) == 0 or len(targets) == 0:
#         print(f"\n--- {task_name} Metrics ---")
#         print("No predictions or targets available for this task. Skipping metric computation.")
#         return 0.0, {}, {}

#     preds_np = predictions.cpu().numpy()
#     targets_np = targets.cpu().numpy()

#     acc = accuracy_score(targets_np, preds_np)
#     labels = list(range(num_classes))
#     report = classification_report(targets_np, preds_np, output_dict=True, zero_division=0, labels=labels)
#     cm = confusion_matrix(targets_np, preds_np, labels=labels)

#     print(f"\n--- {task_name} Metrics ---")
#     print(f"Accuracy: {acc:.4f}")
#     print(f"Classification Report:\n{json.dumps(report, indent=2)}")
#     print(f"Confusion Matrix:\n{cm}")
    
#     return acc, report, cm

# def train_main_classifier():
#     print(f"Using device: {config.DEVICE}")

#     torch.manual_seed(config.SEED)
#     np.random.seed(config.SEED)
#     random.seed(config.SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(config.SEED)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     print("\nStarting multi-modal material classification training...")
#     print("Setting up environment...")
    
#     full_dataset_raw = MaterialDataset(
#         master_index_path=config.MASTER_INDEX_PATH,
#         kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
#         data_root_dir=config.DATA_DIR,
#         dos_fermi_dir=config.DOS_FERMI_DIR,
#         scaler=None
#     )
    
#     try:
#         if len(full_dataset_raw) > 0:
#             _ = full_dataset_raw[0] 
#             print("Dataset dimensions inferred and config updated (if dynamic).")
#         else:
#             warnings.warn("Dataset is empty. Cannot infer dimensions. Ensure config.py has accurate feature dimensions.")
#     except Exception as e:
#         warnings.warn(f"Could not load first sample to infer dimensions: {e}. Ensure config.py has accurate feature dimensions set manually.")

#     print("Environment setup complete.")

#     print("\nStarting data preprocessing with ImprovedDataPreprocessor...")
#     preprocessor = ImprovedDataPreprocessor()
    
#     all_raw_data_dicts = []
#     for i in range(len(full_dataset_raw)): # No tqdm here as requested
#         try:
#             item_data = full_dataset_raw[i]
#             item_data_cpu = {}
#             for k, v in item_data.items():
#                 if isinstance(v, torch.Tensor):
#                     item_data_cpu[k] = v.cpu()
#                 elif isinstance(v, dict):
#                     item_data_cpu[k] = {sk: sv.cpu() if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
#                 else:
#                     item_data_cpu[k] = v
#             all_raw_data_dicts.append(item_data_cpu)
#         except Exception as e:
#             warnings.warn(f"Skipping material at index {i} due to loading error during raw data collection: {e}")
#             continue

#     processed_data_list = preprocessor.fit_transform(all_raw_data_dicts)
#     print(f"Data preprocessing complete. Processed {len(processed_data_list)} samples.")
    
#     print("\nCreating stratified data splits with StratifiedDataSplitter...")
    
#     # Explicitly instantiate the splitter object. This line is crucial.
#     my_splitter_instance = StratifiedDataSplitter( 
#         test_size=config.TEST_RATIO,
#         val_size=config.VAL_RATIO,
#         random_state=config.SEED
#     )
    
#     # --- DEBUG PRINT FOR SPLITTER TYPE ---
#     print(f"DEBUG: Type of my_splitter_instance before calling .split(): {type(my_splitter_instance)}")
#     # --- END DEBUG PRINT ---

#     # Call the split method on the *instantiated object*.
#     # The StratifiedDataSplitter in data_processing.py internally extracts labels for stratification.
#     train_data_list, val_data_list, test_data_list = my_splitter_instance.split(processed_data_list)
    
#     print(f"Dataset split: Train={len(train_data_list)}, Val={len(val_data_list)}, Test={len(test_data_list)}")

#     train_loader = PyGDataLoader(train_data_list, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
#     val_loader = PyGDataLoader(val_data_list, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
#     test_loader = PyGDataLoader(test_data_list, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)

#     model = MultiModalMaterialClassifier(
#         crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
#         kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
#         asph_feature_dim=config.ASPH_FEATURE_DIM,
#         scalar_feature_dim=config.SCALAR_TOTAL_DIM,
#         decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
        
#         num_topology_classes=config.NUM_TOPOLOGY_CLASSES,
#         num_magnetism_classes=config.NUM_MAGNETISM_CLASSES,
#         num_combined_classes=config.NUM_COMBINED_CLASSES,
        
#         egnn_hidden_irreps_str=config.EGNN_HIDDEN_IRREPS_STR,
#         egnn_num_layers=config.GNN_NUM_LAYERS, 
#         egnn_radius=config.EGNN_RADIUS, 

#         kspace_gnn_hidden_channels=config.GNN_HIDDEN_CHANNELS,
#         kspace_gnn_num_layers=config.GNN_NUM_LAYERS,
#         kspace_gnn_num_heads=config.KSPACE_GNN_NUM_HEADS, 

#         ffnn_hidden_dims_asph=config.FFNN_HIDDEN_DIMS_ASPH,
#         ffnn_hidden_dims_scalar=config.FFNN_HIDDEN_DIMS_SCALAR,
        
#         latent_dim_gnn=config.LATENT_DIM_GNN,
#         latent_dim_asph=config.LATENT_DIM_ASPH,
#         latent_dim_other_ffnn=config.LATENT_DIM_OTHER_FFNN,
#         fusion_hidden_dims=config.FUSION_HIDDEN_DIMS, 

#         crystal_encoder_hidden_dim=config.crystal_encoder_hidden_dim, 
#         crystal_encoder_num_layers=config.crystal_encoder_num_layers,
#         crystal_encoder_output_dim=config.crystal_encoder_output_dim,
#         crystal_encoder_radius=config.crystal_encoder_radius,
#         crystal_encoder_num_scales=config.crystal_encoder_num_scales,
#         crystal_encoder_use_topological_features=config.crystal_encoder_use_topological_features
#     ).to(config.DEVICE)

#     print(f"Model instantiated successfully: \n{model}")
#     print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

#     # --- Calculate Class Weights (for CrossEntropyLoss) ---
#     train_combined_labels = [data['combined_label'].item() for data in train_data_list]
#     combined_class_counts = Counter(train_combined_labels)
#     total_combined_samples = sum(combined_class_counts.values())
#     combined_num_classes = config.NUM_COMBINED_CLASSES

#     combined_class_weights_raw = torch.zeros(combined_num_classes, dtype=torch.float32)
#     print("\n--- Combined Class Distribution (Training Set) ---")
#     for i in range(combined_num_classes):
#         class_name_tuple = None 
#         for k, v in config.COMBINED_CLASS_MAPPING.items():
#             if v == i:
#                 class_name_tuple = k
#                 break
#         count = combined_class_counts.get(i, 0)
#         print(f"Class {i} ('{class_name_tuple}'): {count} samples")
#         if count > 0:
#             combined_class_weights_raw[i] = total_combined_samples / (count * combined_num_classes)
#         else:
#             combined_class_weights_raw[i] = 1.0
#     combined_class_weights = combined_class_weights_raw / combined_class_weights_raw.sum() * combined_num_classes
#     print(f"Calculated Combined Class Weights: {combined_class_weights.tolist()}")
#     print("---------------------------------------------------\n")

#     train_topology_labels = [data['topology_label'].item() for data in train_data_list]
#     topology_class_counts = Counter(train_topology_labels)
#     total_topology_samples = sum(topology_class_counts.values())
#     topology_num_classes = config.NUM_TOPOLOGY_CLASSES

#     topology_class_weights_raw = torch.zeros(topology_num_classes, dtype=torch.float32)
#     print("\n--- Topology Class Distribution (Training Set) ---")
#     for i in range(topology_num_classes):
#         class_name = None
#         for name, idx in config.TOPOLOGY_CLASS_MAPPING.items():
#             if idx == i:
#                 class_name = name
#                 break
#         count = topology_class_counts.get(i, 0)
#         print(f"Class {i} ('{class_name}'): {count} samples")
#         if count > 0:
#             topology_class_weights_raw[i] = total_topology_samples / (count * topology_num_classes)
#         else:
#             topology_class_weights_raw[i] = 1.0
#     topology_class_weights = topology_class_weights_raw / topology_class_weights_raw.sum() * topology_num_classes
#     print(f"Calculated Topology Class Weights: {topology_class_weights.tolist()}")
#     print("---------------------------------------------------\n")

#     train_magnetism_labels = [data['magnetism_label'].item() for data in train_data_list]
#     magnetism_class_counts = Counter(train_magnetism_labels)
#     total_magnetism_samples = sum(magnetism_class_counts.values())
#     magnetism_num_classes = config.NUM_MAGNETISM_CLASSES

#     magnetism_class_weights_raw = torch.zeros(magnetism_num_classes, dtype=torch.float32)
#     print("--- Magnetism Class Distribution (Training Set) ---")
#     for i in range(magnetism_num_classes):
#         class_name = None
#         for name, idx in config.MAGNETISM_CLASS_MAPPING.items():
#             if idx == i:
#                 class_name = name
#                 break
#         count = magnetism_class_counts.get(i, 0)
#         print(f"Class {i} ('{class_name}'): {count} samples")
#         if count > 0:
#             magnetism_class_weights_raw[i] = total_magnetism_samples / (count * magnetism_num_classes)
#         else:
#             magnetism_class_weights_raw[i] = 1.0
#     magnetism_class_weights = magnetism_class_weights_raw / magnetism_class_weights_raw.sum() * magnetism_num_classes
#     print(f"Calculated Magnetism Class Weights: {magnetism_class_weights.tolist()}")
#     print("---------------------------------------------------\n")


#     # --- Loss Functions ---
#     criterion_combined = nn.CrossEntropyLoss(weight=combined_class_weights.to(config.DEVICE)).to(config.DEVICE)
    
#     criterion_topology_aux = EnhancedTopologicalLoss(
#         alpha=1.0, 
#         beta=config.LOSS_WEIGHT_TOPO_CONSISTENCY,
#         gamma=config.LOSS_WEIGHT_REGULARIZATION
#     ).to(config.DEVICE)
#     criterion_topology_aux.classification_loss = nn.CrossEntropyLoss(
#         weight=topology_class_weights.to(config.DEVICE)
#     )

#     criterion_magnetism_aux = nn.CrossEntropyLoss(
#         weight=magnetism_class_weights.to(config.DEVICE)
#     ).to(config.DEVICE)


#     # --- Optimizer & Scheduler ---
#     optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

#     best_val_loss = float('inf')
#     patience_counter = 0
    
#     print("\nStarting training loop...")

#     for epoch in range(config.NUM_EPOCHS):
#         model.train()
#         total_train_loss = 0
        
#         for batch_idx, batch in enumerate(train_loader):
#             batch['crystal_graph'] = batch['crystal_graph'].to(config.DEVICE)
#             batch['kspace_graph'] = batch['kspace_graph'].to(config.DEVICE)
            
#             for key in ['asph_features', 'scalar_features', 'topology_label', 'magnetism_label', 'combined_label']:
#                 if isinstance(batch[key], torch.Tensor):
#                     batch[key] = batch[key].to(config.DEVICE)
            
#             if 'kspace_physics_features' in batch and isinstance(batch['kspace_physics_features'], dict):
#                 for sub_key in batch['kspace_physics_features']: 
#                     if isinstance(batch['kspace_physics_features'][sub_key], torch.Tensor):
#                         batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 
#             else:
#                  warnings.warn(f"kspace_physics_features not found or not a dict in batch {batch_idx}. Using default values for model_inputs (may cause errors if not handled by model).")
#                  batch['kspace_physics_features'] = {
#                      'decomposition_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DECOMPOSITION_FEATURE_DIM, device=config.DEVICE),
#                      'gap_features': torch.zeros(batch['crystal_graph'].num_graphs, config.BAND_GAP_SCALAR_DIM, device=config.DEVICE),
#                      'dos_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DOS_FEATURE_DIM, device=config.DEVICE),
#                      'fermi_features': torch.zeros(batch['crystal_graph'].num_graphs, config.FERMI_FEATURE_DIM, device=config.DEVICE),
#                  }

#             optimizer.zero_grad()
            
#             model_inputs = {
#                 'crystal_graph': batch['crystal_graph'],
#                 'kspace_graph': batch['kspace_graph'],
#                 'asph_features': batch['asph_features'],
#                 'scalar_features': batch['scalar_features'],
#                 'kspace_physics_features': batch['kspace_physics_features']
#             }

#             outputs = model(model_inputs)
            
#             combined_logits = outputs['combined_logits']
#             topology_logits_aux = outputs['topology_logits_aux']
#             magnetism_logits_aux = outputs['magnetism_logits_aux']
#             topological_features = outputs.get('topological_features', None)

#             loss_combined = criterion_combined(combined_logits, batch['combined_label'])
#             loss_topology_aux = criterion_topology_aux(
#                 topology_logits_aux,
#                 batch['topology_label'],
#                 topological_features=topological_features
#             )
#             loss_magnetism_aux = criterion_magnetism_aux(magnetism_logits_aux, batch['magnetism_label'])

#             total_loss = (
#                 config.LOSS_WEIGHT_PRIMARY_COMBINED * loss_combined +
#                 config.LOSS_WEIGHT_AUX_TOPOLOGY * loss_topology_aux + 
#                 config.LOSS_WEIGHT_AUX_MAGNETISM * loss_magnetism_aux
#             )
            
#             total_train_loss += total_loss.item()

#             total_loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
            
#             optimizer.step()

#         avg_train_loss = total_train_loss / len(train_loader)
        
#         # --- Validation Loop ---
#         model.eval()
#         total_val_loss = 0
#         all_val_combined_preds = []
#         all_val_combined_labels = []
#         all_val_topo_preds = []
#         all_val_topo_labels = []
#         all_val_mag_preds = []
#         all_val_mag_labels = []

#         with torch.no_grad():
#             for batch_idx, batch in enumerate(val_loader):
#                 batch['crystal_graph'] = batch['crystal_graph'].to(config.DEVICE)
#                 batch['kspace_graph'] = batch['kspace_graph'].to(config.DEVICE)
#                 for key in ['asph_features', 'scalar_features', 'topology_label', 'magnetism_label', 'combined_label']:
#                     if isinstance(batch[key], torch.Tensor):
#                         batch[key] = batch[key].to(config.DEVICE)
#                 if 'kspace_physics_features' in batch and isinstance(batch['kspace_physics_features'], dict):
#                     for sub_key in batch['kspace_physics_features']: 
#                         if isinstance(batch['kspace_physics_features'][sub_key], torch.Tensor):
#                             batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 
#                 else:
#                     warnings.warn(f"kspace_physics_features not found or not a dict in batch {batch_idx} during validation. Using dummy.")
#                     batch['kspace_physics_features'] = {
#                         'decomposition_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DECOMPOSITION_FEATURE_DIM, device=config.DEVICE),
#                         'gap_features': torch.zeros(batch['crystal_graph'].num_graphs, config.BAND_GAP_SCALAR_DIM, device=config.DEVICE),
#                         'dos_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DOS_FEATURE_DIM, device=config.DEVICE),
#                         'fermi_features': torch.zeros(batch['crystal_graph'].num_graphs, config.FERMI_FEATURE_DIM, device=config.DEVICE),
#                     }
                
#                 model_inputs = {
#                     'crystal_graph': batch['crystal_graph'],
#                     'kspace_graph': batch['kspace_graph'],
#                     'asph_features': batch['asph_features'],
#                     'scalar_features': batch['scalar_features'],
#                     'kspace_physics_features': batch['kspace_physics_features']
#                 }

#                 outputs = model(model_inputs)
                
#                 combined_logits = outputs['combined_logits']
#                 topology_logits_aux = outputs['topology_logits_aux']
#                 magnetism_logits_aux = outputs['magnetism_logits_aux']
#                 topological_features = outputs.get('topological_features', None)

#                 loss_combined = criterion_combined(combined_logits, batch['combined_label'])
#                 loss_topology_aux = criterion_topology_aux(
#                     topology_logits_aux,
#                     batch['topology_label'],
#                     topological_features=topological_features
#                 )
#                 loss_magnetism_aux = criterion_magnetism_aux(magnetism_logits_aux, batch['magnetism_label'])
                
#                 val_loss = (
#                     config.LOSS_WEIGHT_PRIMARY_COMBINED * loss_combined +
#                     config.LOSS_WEIGHT_AUX_TOPOLOGY * loss_topology_aux +
#                     config.LOSS_WEIGHT_AUX_MAGNETISM * loss_magnetism_aux
#                 )
#                 total_val_loss += val_loss.item()

#                 _, predicted_combined = torch.max(combined_logits, 1)
#                 all_val_combined_preds.extend(predicted_combined.cpu().numpy())
#                 all_val_combined_labels.extend(batch['combined_label'].cpu().numpy())

#                 _, predicted_topo_aux = torch.max(topology_logits_aux, 1)
#                 all_val_topo_preds.extend(predicted_topo_aux.cpu().numpy())
#                 all_val_topo_labels.extend(batch['topology_label'].cpu().numpy())

#                 _, predicted_mag_aux = torch.max(magnetism_logits_aux, 1)
#                 all_val_mag_preds.extend(predicted_mag_aux.cpu().numpy())
#                 all_val_mag_labels.extend(batch['magnetism_label'].cpu().numpy())

#         avg_val_loss = total_val_loss / len(val_loader)

#         print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
#         print(f"  Train Loss: {avg_train_loss:.4f}")
#         print(f"  Validation Loss: {avg_val_loss:.4f}")

#         print(f"DEBUG VAL: all_val_combined_preds length: {len(all_val_combined_preds)}")
#         print(f"DEBUG VAL: all_val_combined_labels length: {len(all_val_combined_labels)}")
#         if len(all_val_combined_preds) > 0:
#             print(f"DEBUG VAL: First 5 val_combined_preds: {all_val_combined_preds[:5]}")
#             print(f"DEBUG VAL: First 5 val_combined_labels: {all_val_combined_labels[:5]}")
#             print(f"DEBUG VAL: Unique val_combined_preds: {np.unique(all_val_combined_preds)}")
#             print(f"DEBUG VAL: Unique val_combined_labels: {np.unique(all_val_combined_labels)}")


#         _ = compute_metrics(torch.tensor(all_val_combined_preds), torch.tensor(all_val_combined_labels), config.NUM_COMBINED_CLASSES, "Combined Classification (Validation)")
#         _ = compute_metrics(torch.tensor(all_val_topo_preds), torch.tensor(all_val_topo_labels), config.NUM_TOPOLOGY_CLASSES, "Topology Classification (Validation)")
#         _ = compute_metrics(torch.tensor(all_val_mag_preds), torch.tensor(all_val_mag_labels), config.NUM_MAGNETISM_CLASSES, "Magnetism Classification (Validation)")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             patience_counter = 0
#             model_save_path = config.MODEL_SAVE_DIR / "best_multi_task_classifier.pth"
#             torch.save(model.state_dict(), model_save_path)
#             print(f"  Model saved to {model_save_path}")
#         else:
#             patience_counter += 1
#             if patience_counter >= config.PATIENCE:
#                 print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
#                 break

#     print("\n--- Evaluating on Test Set ---")
#     if (config.MODEL_SAVE_DIR / "best_multi_task_classifier.pth").exists():
#         model.load_state_dict(torch.load(config.MODEL_SAVE_DIR / "best_multi_task_classifier.pth", map_location=config.DEVICE))
#     else:
#         warnings.warn(f"Best model not found at {config.MODEL_SAVE_DIR / 'best_multi_task_classifier.pth'}. Testing with current model state.")

#     model.eval()

#     all_test_combined_preds = []
#     all_test_combined_labels = []
#     all_test_topo_preds = []
#     all_test_topo_labels = []
#     all_test_mag_preds = []
#     all_test_mag_labels = []

#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Test Evaluation"):
#             batch['crystal_graph'] = batch['crystal_graph'].to(config.DEVICE)
#             batch['kspace_graph'] = batch['kspace_graph'].to(config.DEVICE)
#             for key in ['asph_features', 'scalar_features', 'topology_label', 'magnetism_label', 'combined_label']:
#                 if isinstance(batch[key], torch.Tensor):
#                     batch[key] = batch[key].to(config.DEVICE)
#             if 'kspace_physics_features' in batch and isinstance(batch['kspace_physics_features'], dict):
#                 for sub_key in batch['kspace_physics_features']: 
#                     if isinstance(batch['kspace_physics_features'][sub_key], torch.Tensor):
#                         batch['kspace_physics_features'][sub_key] = batch['kspace_physics_features'][sub_key].to(config.DEVICE) 
#             else:
#                 warnings.warn(f"kspace_physics_features not found or not a dict in batch during test evaluation. Using dummy.")
#                 batch['kspace_physics_features'] = {
#                     'decomposition_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DECOMPOSITION_FEATURE_DIM, device=config.DEVICE),
#                     'gap_features': torch.zeros(batch['crystal_graph'].num_graphs, config.BAND_GAP_SCALAR_DIM, device=config.DEVICE),
#                     'dos_features': torch.zeros(batch['crystal_graph'].num_graphs, config.DOS_FEATURE_DIM, device=config.DEVICE),
#                     'fermi_features': torch.zeros(batch['crystal_graph'].num_graphs, config.FERMI_FEATURE_DIM, device=config.DEVICE),
#                 }

#             model_inputs = {
#                 'crystal_graph': batch['crystal_graph'],
#                 'kspace_graph': batch['kspace_graph'],
#                 'asph_features': batch['asph_features'],
#                 'scalar_features': batch['scalar_features'],
#                 'kspace_physics_features': batch['kspace_physics_features']
#             }
#             outputs = model(model_inputs)
            
#             combined_logits = outputs['combined_logits']
#             topology_logits_aux = outputs['topology_logits_aux']
#             magnetism_logits_aux = outputs['magnetism_logits_aux']

#             _, predicted_combined = torch.max(combined_logits, 1)
#             all_test_combined_preds.extend(predicted_combined.cpu().numpy())
#             all_test_combined_labels.extend(batch['combined_label'].cpu().numpy())

#             _, predicted_topo_aux = torch.max(topology_logits_aux, 1)
#             all_test_topo_preds.extend(predicted_topo_aux.cpu().numpy())
#             all_test_topo_labels.extend(batch['topology_label'].cpu().numpy())

#             _, predicted_mag_aux = torch.max(magnetism_logits_aux, 1)
#             all_test_mag_preds.extend(predicted_mag_aux.cpu().numpy())
#             all_test_mag_labels.extend(batch['magnetism_label'].cpu().numpy())

#     print("\nTest Set Results:")
#     print(f"DEBUG TEST: all_test_combined_preds length: {len(all_test_combined_preds)}")
#     print(f"DEBUG TEST: all_test_combined_labels length: {len(all_test_combined_labels)}")
#     if len(all_test_combined_preds) > 0:
#         print(f"DEBUG TEST: First 5 test_combined_preds: {all_test_combined_preds[:5]}")
#         print(f"DEBUG TEST: First 5 test_combined_labels: {all_test_combined_labels[:5]}")
#         print(f"DEBUG TEST: Unique test_combined_preds: {np.unique(all_test_combined_preds)}")
#         print(f"DEBUG TEST: Unique test_combined_labels: {np.unique(all_test_combined_labels)}")

#     _ = compute_metrics(torch.tensor(all_test_combined_preds), torch.tensor(all_test_combined_labels), config.NUM_COMBINED_CLASSES, "Combined Classification")
#     _ = compute_metrics(torch.tensor(all_test_topo_preds), torch.tensor(all_test_topo_labels), config.NUM_TOPOLOGY_CLASSES, "Topology Classification")
#     _ = compute_metrics(torch.tensor(all_test_mag_preds), torch.tensor(all_test_mag_labels), config.NUM_MAGNETISM_CLASSES, "Magnetism Classification")

# def main_training_loop(): 
#     train_main_classifier()

# if __name__ == "__main__":
#     dummy_data_root = Path("./dummy_multimodal_db")
#     dummy_master_index_path = dummy_data_root / "metadata"
#     dummy_kspace_graphs_base_dir = dummy_data_root / "kspace_graphs"
#     dummy_crystal_graphs_base_dir_scratch = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/crystal_graphs")
#     dummy_vectorized_features_base_dir_scratch = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/vectorized_features")
    
#     dummy_master_index_path.mkdir(parents=True, exist_ok=True)
#     dummy_kspace_graphs_base_dir.mkdir(parents=True, exist_ok=True)
#     dummy_crystal_graphs_base_dir_scratch.mkdir(parents=True, exist_ok=True)
#     dummy_vectorized_features_base_dir_scratch.mkdir(parents=True, exist_ok=True)

#     dummy_json_data = [
#         {'jid': 'mat_001', 'formula': 'GaAs', 'space_group': 'F-43m', 'space_group_number': 216,
#          'topological_class': 'Trivial', 'magnetic_type': 'NM', 'band_gap': 1.5,
#          'formation_energy': -0.8, 'energy_above_hull': 0.0, 'density': 5.32, 'volume': 45.1, 'nsites': 2, 'total_magnetization': 0.0, 'theoretical': True},
#         {'jid': 'mat_002', 'formula': 'Bi2Se3', 'space_group': 'R-3m', 'space_group_number': 166,
#          'topological_class': 'Topological Insulator', 'magnetic_type': 'NM', 'band_gap': 0.1,
#          'formation_energy': -0.5, 'energy_above_hull': 0.0, 'density': 6.82, 'volume': 120.5, 'nsites': 5, 'total_magnetization': 0.0, 'theoretical': True},
#         {'jid': 'mat_003', 'formula': 'MnBi', 'space_group': 'P6_3/mmc', 'space_group_number': 194,
#          'topological_class': 'Semimetal', 'magnetic_type': 'FM', 'band_gap': 0.0,
#          'formation_energy': -1.0, 'energy_above_hull': 0.0, 'density': 8.0, 'volume': 70.0, 'nsites': 2, 'total_magnetization': 2.2, 'theoretical': True},
#         {'jid': 'mat_004', 'formula': 'FeSi', 'space_group': 'P2_13', 'space_group_number': 198,
#          'topological_class': 'Trivial', 'magnetic_type': 'AFM', 'band_gap': 1.2,
#          'formation_energy': -0.9, 'energy_above_hull': 0.0, 'density': 6.18, 'volume': 60.0, 'nsites': 2, 'total_magnetization': 0.5, 'theoretical': True},
#          {'jid': 'mat_005', 'formula': 'Cd3As2', 'space_group': 'I4_1cd', 'space_group_number': 110,
#           'topological_class': 'Semimetal', 'magnetic_type': 'NM', 'band_gap': 0.0,
#           'formation_energy': -0.6, 'energy_above_hull': 0.0, 'density': 6.2, 'volume': 150.0, 'nsites': 5, 'total_magnetization': 0.0, 'theoretical': True},
#          {'jid': 'mat_006', 'formula': 'Cr2Ge2Te6', 'space_group': 'R-3m', 'space_group_number': 166,
#           'topological_class': 'Topological Insulator', 'magnetic_type': 'FM', 'band_gap': 0.2,
#           'formation_energy': -0.4, 'energy_above_hull': 0.0, 'density': 5.0, 'volume': 200.0, 'nsites': 10, 'total_magnetization': 1.8, 'theoretical': True},
#     ]

#     if not hasattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM'): config.BASE_DECOMPOSITION_FEATURE_DIM = 2
#     if not hasattr(config, 'ALL_POSSIBLE_IRREPS'): config.ALL_POSSIBLE_IRREPS = ['A1', 'E1', 'T2'] 
#     if not hasattr(config, 'MAX_DECOMPOSITION_INDICES_LEN'): config.MAX_DECOMPOSITION_INDICES_LEN = 5
#     _expected_decomp_dim = config.BASE_DECOMPOSITION_FEATURE_DIM + len(config.ALL_POSSIBLE_IRREPS) + config.MAX_DECOMPOSITION_INDICES_LEN
#     config.DECOMPOSITION_FEATURE_DIM = _expected_decomp_dim

#     config.CRYSTAL_NODE_FEATURE_DIM = 3 
#     config.KSPACE_GRAPH_NODE_FEATURE_DIM = 10 
#     config.ASPH_FEATURE_DIM = 3115 
#     config.SCALAR_TOTAL_DIM = 4756 + 7 
#     config.BAND_REP_FEATURE_DIM = 4756 
#     config.BAND_GAP_SCALAR_DIM = 1 
#     config.DOS_FEATURE_DIM = 100 
#     config.FERMI_FEATURE_DIM = 1 

#     for data in tqdm(dummy_json_data, desc="Creating dummy data files"):
#         with open(dummy_master_index_path / f"{data['jid']}.json", 'w') as f:
#             json.dump(data, f)
        
#         (dummy_crystal_graphs_base_dir_scratch / data['jid']).mkdir(parents=True, exist_ok=True)
#         with open(dummy_crystal_graphs_base_dir_scratch / data['jid'] / "crystal_graph.pkl", 'wb') as f:
#             pickle.dump({'x': np.random.rand(10, config.CRYSTAL_NODE_FEATURE_DIM), 
#                          'pos': np.random.rand(10, 3), 
#                          'edge_index': np.random.randint(0, 10, (2, 20))}, f)
        
#         (dummy_vectorized_features_base_dir_scratch / data['jid']).mkdir(parents=True, exist_ok=True)
#         np.save(dummy_vectorized_features_base_dir_scratch / data['jid'] / "asph_features_rev2.npy", np.random.rand(config.ASPH_FEATURE_DIM))
#         np.save(dummy_vectorized_features_base_dir_scratch / data['jid'] / "band_rep_features.npy", np.random.rand(config.BAND_REP_FEATURE_DIM))
        
#         sg_folder = dummy_kspace_graphs_base_dir / f"SG_{str(int(data['space_group_number'])).zfill(3)}"
#         sg_folder.mkdir(parents=True, exist_ok=True)
        
#         dummy_kspace_graph_pyg = PyGData(x=torch.randn(5, config.KSPACE_GRAPH_NODE_FEATURE_DIM), 
#                                          edge_index=torch.randint(0, 5, (2, 8)),
#                                          pos=torch.randn(5,3))
#         torch.save(dummy_kspace_graph_pyg, sg_folder / "kspace_graph.pt")
        
#         torch.save({'decomposition_features': torch.randn(config.BASE_DECOMPOSITION_FEATURE_DIM)}, sg_folder / "physics_features.pt")
        
#         with open(sg_folder / "metadata.json", 'w') as f:
#             json.dump({
#                 "ebr_data": {"irrep_multiplicities": {irrep: np.random.randint(1,5) for irrep in config.ALL_POSSIBLE_IRREPS[:2]}},
#                 "decomposition_branches": {"decomposition_indices": [np.random.randint(0,10) for _ in range(config.MAX_DECOMPOSITION_INDICES_LEN)]}
#             }, f)

#     main_training_loop()

#     # Clean up dummy directories and files
#     import shutil
#     if dummy_data_root.exists():
#         shutil.rmtree(dummy_data_root)
#         print(f"Cleaned up {dummy_data_root}")
#     if dummy_crystal_graphs_base_dir_scratch.exists():
#         shutil.rmtree(dummy_crystal_graphs_base_dir_scratch)
#         print(f"Cleaned up {dummy_crystal_graphs_base_dir_scratch}")
#     if dummy_vectorized_features_base_dir_scratch.exists():
#         shutil.rmtree(dummy_vectorized_features_base_dir_scratch)
#         print(f"Cleaned up {dummy_vectorized_features_base_dir_scratch}")
#     if config.MODEL_SAVE_DIR.exists():
#         shutil.rmtree(config.MODEL_SAVE_DIR)
#         print(f"Cleaned up {config.MODEL_SAVE_DIR}")

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
    
    # Initialize loss functions
    criterion_combined = nn.CrossEntropyLoss(weight=combined_class_weights.to(config.DEVICE))
    # For auxiliary topology, use the EnhancedTopologicalLoss
    criterion_topology_aux = EnhancedTopologicalLoss(
        alpha=config.LOSS_WEIGHT_AUX_TOPOLOGY,
        beta=config.LOSS_WEIGHT_TOPO_CONSISTENCY,
        gamma=config.LOSS_WEIGHT_REGULARIZATION
    ).to(config.DEVICE)
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
                    batch['topology_label'], 
                    outputs['extracted_topo_features'] # Pass topological_features for consistency loss
                )
                loss_magnetism_aux = criterion_magnetism_aux(outputs['magnetism_logits_aux'], batch['magnetism_label'])
                total_loss = loss_topology_aux + loss_magnetism_aux
                
            elif stage == 2: # All tasks with weighted sum
                loss_combined = criterion_combined(outputs['combined_logits'], batch['combined_label'])
                loss_topology_aux = criterion_topology_aux(
                    outputs['topology_logits_aux'], 
                    batch['topology_label'], 
                    outputs['extracted_topo_features']
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
                        batch['topology_label'], 
                        outputs['extracted_topo_features']
                    )
                    loss_magnetism_aux = criterion_magnetism_aux(outputs['magnetism_logits_aux'], batch['magnetism_label'])
                    total_loss = loss_topology_aux + loss_magnetism_aux
                elif stage == 2:
                    loss_combined = criterion_combined(outputs['combined_logits'], batch['combined_label'])
                    loss_topology_aux = criterion_topology_aux(
                        outputs['topology_logits_aux'], 
                        batch['topology_label'], 
                        outputs['extracted_topo_features']
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

    topology_class_weights = calculate_class_weights(
        train_topology_labels, config.NUM_TOPOLOGY_CLASSES, config.TOPOLOGY_CLASS_MAPPING
    )
    magnetism_class_weights = calculate_class_weights(
        train_magnetism_labels, config.NUM_MAGNETISM_CLASSES, config.MAGNETISM_CLASS_MAPPING
    )
    combined_class_weights = calculate_class_weights(
        train_combined_labels, config.NUM_COMBINED_CLASSES, config.COMBINED_CLASS_MAPPING
    ) # Use the new combined mapping

    final_full_dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR, 
        scaler=feature_scalers 
    )
    print("Final dataset re-initialized with fitted scalers.")

    # Create the subsets using the pre-calculated indices
    train_dataset = torch.utils.data.Subset(final_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(final_full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(final_full_dataset, test_indices)

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = PyGDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
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