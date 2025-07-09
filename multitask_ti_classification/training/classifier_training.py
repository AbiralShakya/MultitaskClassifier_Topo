''' 
src/model_with_topological_ml.py
EnhancedMultiModalMaterialClassifier with Spectral Graph Encoder and optional Topological ML.
Supports 3-way classification (trivial/semimetal/topological-insulator) plus auxiliary tasks.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from torch_geometric.data import Data
import time
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
import pickle
import json
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch as PyGBatch

# Core encoders
from helper.topological_crystal_encoder import TopologicalCrystalEncoder
from src.model_w_debug import KSpaceTransformerGNNEncoder, ScalarFeatureEncoder
from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures
from encoders.ph_token_encoder import PHTokenEncoder
# Spectral graph features
from helper.graph_spectral_encoder import GraphSpectralEncoder
# Optional ML-based topology encoder
from src.topological_ml_encoder import (
    TopologicalMLEncoder, TopologicalMLEncoder2D,
    create_hamiltonian_from_features, compute_topological_loss
)
import helper.config as config

class EnhancedMultiModalMaterialClassifier(nn.Module):
    def __init__(
        self,
        # Feature dims
        crystal_node_feature_dim: int,
        kspace_node_feature_dim: int,
        asph_feature_dim: int,
        scalar_feature_dim: int,
        decomposition_feature_dim: int,
        # Class counts
        num_combined_classes: int = config.NUM_COMBINED_CLASSES,
        num_topology_classes: int = config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes: int = config.NUM_MAGNETISM_CLASSES,
        # Crystal encoder params
        crystal_encoder_hidden_dim: int = 128,
        crystal_encoder_num_layers: int = 4,
        crystal_encoder_output_dim: int = 128,
        crystal_encoder_radius: float = 5.0,
        crystal_encoder_num_scales: int = 3,
        crystal_encoder_use_topological_features: bool = True,
        # k-space GNN params
        kspace_gnn_hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers: int = config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads: int = config.KSPACE_GNN_NUM_HEADS,
        latent_dim_gnn: int = config.LATENT_DIM_GNN,
        # ASPH & scalar dims
        latent_dim_asph: int = config.LATENT_DIM_ASPH,
        latent_dim_other_ffnn: int = config.LATENT_DIM_OTHER_FFNN,
        # Fusion MLP
        fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,
        dropout_rate: float = config.DROPOUT_RATE,
        # Spectral dim
        spectral_hidden: int = config.SPECTRAL_HID,
        # Topological ML params
        use_topo_ml: bool = True,
        topological_ml_dim: int = 128,
        topological_ml_k_points: int = 32,
        topological_ml_model_type: str = "1d_a3",
        topological_ml_auxiliary_weight: float = config.AUXILIARY_WEIGHT,
    ):
        super().__init__()
        # Store dims for fusion
        self._crystal_dim  = crystal_encoder_output_dim
        self._kspace_dim   = latent_dim_gnn
        self._asph_dim     = latent_dim_asph
        self._scalar_dim   = latent_dim_other_ffnn
        self._phys_dim     = latent_dim_other_ffnn
        self._spec_dim     = spectral_hidden
        self._topo_ml_dim  = topological_ml_dim if use_topo_ml else 0

        # Instantiate encoders
        self.crystal_encoder = TopologicalCrystalEncoder(
            node_feature_dim=crystal_node_feature_dim,
            hidden_dim=crystal_encoder_hidden_dim,
            num_layers=crystal_encoder_num_layers,
            output_dim=crystal_encoder_output_dim,
            radius=crystal_encoder_radius,
            num_scales=crystal_encoder_num_scales,
            use_topological_features=crystal_encoder_use_topological_features
        )
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=kspace_node_feature_dim,
            hidden_dim=kspace_gnn_hidden_channels,
            out_channels=latent_dim_gnn,
            n_layers=kspace_gnn_num_layers,
            num_heads=kspace_gnn_num_heads
        )
        self.asph_encoder = PHTokenEncoder(
            input_dim=asph_feature_dim,
            output_dim=latent_dim_asph
        )
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=scalar_feature_dim,
            hidden_dims=config.FFNN_HIDDEN_DIMS_SCALAR,
            out_channels=latent_dim_other_ffnn
        )
        self.enhanced_kspace_physics_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=decomposition_feature_dim,
            gap_features_dim=config.BAND_GAP_SCALAR_DIM,
            dos_features_dim=config.DOS_FEATURE_DIM,
            fermi_features_dim=config.FERMI_FEATURE_DIM,
            output_dim=latent_dim_other_ffnn
        )
        self.spectral_encoder = GraphSpectralEncoder(
            k_eigs=config.K_LAPLACIAN_EIGS,
            hidden=spectral_hidden
        )

        # Topological ML encoder
        self.use_topo_ml = use_topo_ml
        self.topo_ml_aux_weight = topological_ml_auxiliary_weight
        
        if use_topo_ml:
            if topological_ml_model_type == "1d_a3":
                # Use the actual concatenated feature dimension (3659 from debug output)
                # This is the sum of: 128 + 128 + 3115 + 128 + 128 + 32 = 3659
                actual_input_dim = 3659
                self.topo_ml_encoder = TopologicalMLEncoder(
                    input_dim=actual_input_dim,
                    k_points=topological_ml_k_points,
                    hidden_dims=[64,128,256],
                    num_classes=num_topology_classes,
                    output_features=topological_ml_dim,
                    extract_local_features=True
                )
            else:
                k_grid = int(topological_ml_k_points**0.5)
                self.topo_ml_encoder = TopologicalMLEncoder2D(
                    input_dim=3,
                    k_grid=k_grid,
                    hidden_dims=[32,64,128],
                    num_classes=num_topology_classes,
                    output_features=topological_ml_dim,
                    extract_berry_curvature=True
                )

        # Fusion MLP - will be dynamically created based on actual input dimensions
        self.fusion_hidden_dims = fusion_hidden_dims
        self.dropout_rate = dropout_rate
        self.fusion_network = None  # Will be created in forward pass

        # Output heads - will be created dynamically
        self.num_combined_classes = num_combined_classes
        self.num_topology_classes = num_topology_classes
        self.num_magnetism_classes = num_magnetism_classes
        self.combined_head = None
        self.topology_head_aux = None
        self.magnetism_head_aux = None

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Encode each modality
        crystal_emb, topo_aux_logits, _ = self.crystal_encoder(
            inputs['crystal_graph'], return_topological_logits=True
        )
        kspace_emb    = self.kspace_encoder(inputs['kspace_graph'])
        asph_emb      = self.asph_encoder(inputs['asph_features'])
        scalar_emb    = self.scalar_encoder(inputs['scalar_features'])
        phys_emb      = self.enhanced_kspace_physics_encoder(
            decomposition_features=inputs['kspace_physics_features']['decomposition_features'],
            gap_features=inputs['kspace_physics_features'].get('gap_features'),
            dos_features=inputs['kspace_physics_features'].get('dos_features'),
            fermi_features=inputs['kspace_physics_features'].get('fermi_features')
        )
        spec_emb      = self.spectral_encoder(
            inputs['crystal_graph'].edge_index,
            inputs['crystal_graph'].num_nodes,
            getattr(inputs['crystal_graph'], 'batch', None)
        ) 

        # Optional Topological ML features
        ml_emb, ml_logits = None, None
        if self.use_topo_ml:
            raw = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, phys_emb, spec_emb], dim=-1)
            hams = create_hamiltonian_from_features(raw)
            out = self.topo_ml_encoder(hams)
            ml_emb    = out.get('topological_features')
            # Fix the boolean tensor issue by properly checking for None
            ml_logits = out.get('topological_logits')
            if ml_logits is None:
                ml_logits = out.get('chern_logits')

        # Concatenate all
        features = [crystal_emb, kspace_emb, asph_emb, scalar_emb, phys_emb, spec_emb]
        if ml_emb is not None:
            features.append(ml_emb)
        x = torch.cat(features, dim=-1)
        
        # Debug: Print actual dimensions
        print(f"DEBUG - Actual concatenated features shape: {x.shape}")
        print(f"DEBUG - Expected base_dim: {self._crystal_dim + self._kspace_dim + self._asph_dim + self._scalar_dim + self._phys_dim + self._spec_dim + self._topo_ml_dim}")
        print(f"DEBUG - Individual feature dimensions:")
        print(f"  crystal_emb: {crystal_emb.shape}")
        print(f"  kspace_emb: {kspace_emb.shape}")
        print(f"  asph_emb: {asph_emb.shape}")
        print(f"  scalar_emb: {scalar_emb.shape}")
        print(f"  phys_emb: {phys_emb.shape}")
        print(f"  spec_emb: {spec_emb.shape}")
        if ml_emb is not None:
            print(f"  ml_emb: {ml_emb.shape}")

        # Dynamically create fusion network if not exists
        if self.fusion_network is None:
            actual_input_dim = x.shape[1]
            print(f"Creating fusion network with input dimension: {actual_input_dim}")
            layers = []
            in_dim = actual_input_dim
            for h in self.fusion_hidden_dims:
                layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout_rate)]
                in_dim = h
            self.fusion_network = nn.Sequential(*layers).to(x.device)
            
            # Create output heads
            self.combined_head = nn.Linear(in_dim, self.num_combined_classes).to(x.device)
            self.topology_head_aux = nn.Linear(in_dim, self.num_topology_classes).to(x.device)
            self.magnetism_head_aux = nn.Linear(in_dim, self.num_magnetism_classes).to(x.device)

        # Fuse and predict with safety checks
        try:
            # Check for NaN or infinite values
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("WARNING: NaN or infinite values detected in input features!")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            fused = self.fusion_network(x)
            
            # Check output for NaN or infinite values
            if torch.isnan(fused).any() or torch.isinf(fused).any():
                print("WARNING: NaN or infinite values detected in fusion output!")
                fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
                
        except Exception as e:
            print(f"ERROR in fusion network: {e}")
            print(f"Input shape: {x.shape}")
            print(f"Input stats - min: {x.min()}, max: {x.max()}, mean: {x.mean()}")
            raise e
        combined_logits    = self.combined_head(fused)
        topology_primary   = ml_logits if ml_logits is not None else combined_logits
        topology_aux       = topo_aux_logits
        magnetism_aux      = self.magnetism_head_aux(fused)

        return {
            'combined_logits': combined_logits,
            'topology_logits_primary': topology_primary,
            'topology_logits_auxiliary': topology_aux,
            'magnetism_logits_aux': magnetism_aux
        }

    def compute_enhanced_loss(self,
             predictions: Dict[str, torch.Tensor],
             targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        losses['combined_loss'] = F.cross_entropy(predictions['combined_logits'], targets['combined'])
        losses['topology_loss'] = F.cross_entropy(predictions['topology_logits_primary'], targets['topology'])
        if predictions.get('topology_logits_auxiliary') is not None:
            losses['topology_aux_loss'] = F.cross_entropy(
                predictions['topology_logits_auxiliary'], targets['topology']
            )
        losses['magnetism_loss'] = F.cross_entropy(
            predictions['magnetism_logits_aux'], targets['magnetism']
        )
        if self.use_topo_ml:
            ml_preds = {'topological_logits': predictions['topology_logits_primary']}
            if ml_emb := predictions.get('topological_ml_features'):
                ml_preds['topological_features'] = ml_emb
            ml_losses = compute_topological_loss(
                ml_preds, targets['topology'], auxiliary_weight=self.topo_ml_aux_weight
            )
            losses.update({
                'ml_main': ml_losses['main_loss'],
                'ml_feature': ml_losses['feature_loss'],
                'ml_total': ml_losses['total_loss']
            })
        losses['total_loss'] = sum(losses.values())
        return losses


def main_training_loop():
    """
    Main training loop for the multi-modal material classifier.
    This function sets up the training pipeline and runs the training.
    """
    print("Starting main training loop...")
    
    # Import necessary modules
    from helper.dataset import MaterialDataset, custom_collate_fn
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch_geometric
    import torch
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import numpy as np
    from sklearn.model_selection import train_test_split
    from collections import Counter
    from sklearn.preprocessing import StandardScaler
    import helper.config as config
    from pathlib import Path
    import warnings
    
    # Set device and configure GPU memory
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Configure GPU memory growth to prevent segmentation faults
    if device.type == 'cuda':
        import torch.cuda
        torch.cuda.empty_cache()
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available GPU memory
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    # Load dataset with preloading enabled
    print("Loading dataset with preloading...")
    dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR,
        preload=getattr(config, 'PRELOAD_DATASET', True)  # Use config option with fallback
    )
    
    # Split dataset
    # Get labels for stratification, with error handling
    try:
        combined_labels = [dataset[i]['combined_label'].item() for i in range(len(dataset))]
        train_indices, temp_indices = train_test_split(
            range(len(dataset)), 
            test_size=0.3, 
            random_state=42,
            stratify=combined_labels
        )
        val_indices, test_indices = train_test_split(
            temp_indices, 
            test_size=0.5, 
            random_state=42,
            stratify=[combined_labels[i] for i in temp_indices]
        )
    except Exception as e:
        print(f"Warning: Could not stratify dataset split due to error: {e}")
        print("Falling back to random split without stratification.")
        train_indices, temp_indices = train_test_split(
            range(len(dataset)), 
            test_size=0.3, 
            random_state=42
        )
        val_indices, test_indices = train_test_split(
            temp_indices, 
            test_size=0.5, 
            random_state=42
        )
    
    # Create data loaders
    from torch.utils.data import Subset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_loader = PyGDataLoader(
        train_subset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=custom_collate_fn, 
        num_workers=config.NUM_WORKERS
    )
    val_loader = PyGDataLoader(
        val_subset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=config.NUM_WORKERS
    )
    test_loader = PyGDataLoader(
        test_subset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=config.NUM_WORKERS
    )
    
    # Test DataLoader creation
    print(f"[DEBUG] DataLoaders created successfully")
    print(f"[DEBUG] Train loader length: {len(train_loader)}")
    print(f"[DEBUG] Testing first batch access...")
    try:
        first_batch = next(iter(train_loader))
        print(f"[DEBUG] First batch accessed successfully with keys: {list(first_batch.keys())}")
    except Exception as e:
        print(f"[DEBUG] Error accessing first batch: {e}")
        import traceback
        traceback.print_exc()
    
    # Initialize model
    print("Initializing model...")
    model = EnhancedMultiModalMaterialClassifier(
        crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim=config.ASPH_FEATURE_DIM,
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
        num_combined_classes=config.NUM_COMBINED_CLASSES,
        num_topology_classes=config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes=config.NUM_MAGNETISM_CLASSES
    ).to(device)
    print(f"[DEBUG] Model initialized successfully on device: {device}")
    
    # Test model forward pass
    print(f"[DEBUG] Testing model forward pass...")
    try:
        # Move test batch to device
        for key in first_batch:
            if isinstance(first_batch[key], torch.Tensor):
                first_batch[key] = first_batch[key].to(device)
            elif hasattr(first_batch[key], 'batch'):  # Check if it's a PyG Batch object
                first_batch[key] = first_batch[key].to(device)
            elif isinstance(first_batch[key], dict):
                for sub_key in first_batch[key]:
                    if isinstance(first_batch[key][sub_key], torch.Tensor):
                        first_batch[key][sub_key] = first_batch[key][sub_key].to(device)
        
        with torch.no_grad():
            test_output = model(first_batch)
        print(f"[DEBUG] Model forward pass successful, output keys: {list(test_output.keys())}")
    except Exception as e:
        print(f"[DEBUG] Error in model forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    # Setup optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    print("Starting training...")
    
    # Add diagnostic prints
    print(f"[DEBUG] About to start training loop with {len(train_loader)} batches")
    print(f"[DEBUG] First batch will be created now...")
    
    # Test DataLoader iteration
    print(f"[DEBUG] Testing DataLoader iteration...")
    try:
        first_batch_from_loader = next(iter(train_loader))
        print(f"[DEBUG] Successfully got first batch from DataLoader with keys: {list(first_batch_from_loader.keys())}")
        
        # Test getting second batch
        print(f"[DEBUG] Testing second batch...")
        train_iter = iter(train_loader)
        second_batch = next(train_iter)
        print(f"[DEBUG] Successfully got second batch from DataLoader")
        
        # Test manual iteration without enumerate
        print(f"[DEBUG] Testing manual iteration...")
        train_iter2 = iter(train_loader)
        for i in range(3):  # Test first 3 batches
            batch = next(train_iter2)
            print(f"[DEBUG] Successfully got batch {i+1} manually")
        print(f"[DEBUG] Manual iteration test completed successfully")
        
    except Exception as e:
        print(f"[DEBUG] Error getting batch from DataLoader: {e}")
        import traceback
        traceback.print_exc()
        
        # Try with a smaller batch size
        print(f"[DEBUG] Trying with batch size 1...")
        try:
            test_loader = PyGDataLoader(
                train_subset, 
                batch_size=1, 
                shuffle=False, 
                collate_fn=custom_collate_fn, 
                num_workers=0
            )
            test_batch = next(iter(test_loader))
            print(f"[DEBUG] Successfully got test batch with batch_size=1: {list(test_batch.keys())}")
        except Exception as e2:
            print(f"[DEBUG] Error with batch_size=1: {e2}")
            import traceback
            traceback.print_exc()
        
        return None
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_losses = []
        
        print(f"[DEBUG] Starting epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"[DEBUG] About to iterate through {len(train_loader)} batches")
        
        # Use manual iteration instead of enumerate to avoid the hang
        train_iter = iter(train_loader)
        batch_idx = 0
        
        while batch_idx < len(train_loader):
            try:
                batch = next(train_iter)
                batch_start_time = time.time()
                print(f"[TRAIN] Starting batch {batch_idx} of {len(train_loader)} (epoch {epoch+1})...")
                try:
                    print(f"[TRAIN] Batch {batch_idx}: Moving data to device...")
                    device_start = time.time()
                    # Move data to device
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device)
                        elif hasattr(batch[key], 'batch'):  # Check if it's a PyG Batch object
                            batch[key] = batch[key].to(device)
                        elif isinstance(batch[key], dict):
                            for sub_key in batch[key]:
                                if isinstance(batch[key][sub_key], torch.Tensor):
                                    batch[key][sub_key] = batch[key][sub_key].to(device)
                    device_time = time.time() - device_start
                    print(f"[TRAIN] Batch {batch_idx}: Data moved to device successfully in {device_time:.2f}s")
                    
                    # Forward pass
                    print(f"[TRAIN] Batch {batch_idx}: Starting forward pass...")
                    forward_start = time.time()
                    optimizer.zero_grad()
                    outputs = model(batch)
                    forward_time = time.time() - forward_start
                    print(f"[TRAIN] Batch {batch_idx}: Forward pass completed in {forward_time:.2f}s")
                    
                    print(f"[TRAIN] Batch {batch_idx}: Computing losses...")
                    loss_start = time.time()
                    losses = model.compute_enhanced_loss(outputs, {
                        'combined': batch['combined_label'],
                        'topology': batch['topology_label'],
                        'magnetism': batch['magnetism_label']
                    })
                    loss_time = time.time() - loss_start
                    print(f"[TRAIN] Batch {batch_idx}: Losses computed in {loss_time:.2f}s")
                    
                    # Backward pass
                    print(f"[TRAIN] Batch {batch_idx}: Starting backward pass...")
                    backward_start = time.time()
                    losses['total_loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    optimizer.step()
                    backward_time = time.time() - backward_start
                    print(f"[TRAIN] Batch {batch_idx}: Backward pass completed in {backward_time:.2f}s")
                    
                    total_batch_time = time.time() - batch_start_time
                    print(f"[TRAIN] Batch {batch_idx}: Total batch time: {total_batch_time:.2f}s")
                    
                    train_losses.append(losses['total_loss'].item())
                    if batch_idx % 10 == 0:
                        # Monitor GPU memory usage
                        if device.type == 'cuda':
                            gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**3
                            gpu_memory_cached = torch.cuda.memory_reserved(device) / 1024**3
                            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, "
                                  f"Loss: {losses['total_loss'].item():.4f}, "
                                  f"GPU Memory: {gpu_memory_used:.2f}GB used, {gpu_memory_cached:.2f}GB cached")
                        else:
                            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, "
                                  f"Loss: {losses['total_loss'].item():.4f}")
                except Exception as e:
                    print(f"ERROR in training batch {batch_idx}: {e}")
                    print(f"Batch keys: {list(batch.keys())}")
                    import traceback
                    traceback.print_exc()
                
                print(f"[TRAIN] Finished batch {batch_idx} of {len(train_loader)} (epoch {epoch+1})")
                batch_idx += 1
                
            except StopIteration:
                print(f"[DEBUG] Reached end of DataLoader at batch {batch_idx}")
                break
            except Exception as e:
                print(f"ERROR in DataLoader iteration at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                    elif hasattr(batch[key], 'batch'):  # Check if it's a PyG Batch object
                        batch[key] = batch[key].to(device)
                    elif isinstance(batch[key], dict):
                        for sub_key in batch[key]:
                            if isinstance(batch[key][sub_key], torch.Tensor):
                                batch[key][sub_key] = batch[key][sub_key].to(device)
                
                outputs = model(batch)
                targets = {
                    'combined': batch['combined_label'],
                    'topology': batch['topology_label'],
                    'magnetism': batch['magnetism_label']
                }
                losses = model.compute_enhanced_loss(outputs, targets)
                val_losses.append(losses['total_loss'].item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Clean up GPU memory between epochs
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"GPU memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config.MODEL_SAVE_DIR / "best_model.pth")
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print("Training completed!")
    return model
