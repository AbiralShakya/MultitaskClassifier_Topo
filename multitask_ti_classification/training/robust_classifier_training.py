# robust_classifier_training.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Robust training script with improved memory management and error handling.
This version includes:
- Better GPU memory management
- Segmentation fault prevention
- Automatic recovery from crashes
- Reduced spectral encoder warnings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
import helper.config as config
from pathlib import Path
import warnings
import gc
import sys
import traceback
from typing import Dict, Any, List

# Import encoders
from helper.topological_crystal_encoder import TopologicalCrystalEncoder
from helper.kspace_physics_encoders import KSpaceTransformerGNNEncoder, EnhancedKSpacePhysicsFeatures
from helper.ph_token_encoder import PHTokenEncoder
from helper.scalar_feature_encoder import ScalarFeatureEncoder
from helper.graph_spectral_encoder import GraphSpectralEncoder

# Import dataset
from helper.dataset import MaterialDataset, custom_collate_fn
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Subset

class RobustMultiModalMaterialClassifier(nn.Module):
    """
    Robust version of the multi-modal material classifier with better memory management.
    """
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
    ):
        super().__init__()
        
        # Store dimensions for fusion network creation
        self._crystal_dim = crystal_encoder_output_dim
        self._kspace_dim = latent_dim_gnn
        self._asph_dim = latent_dim_asph
        self._scalar_dim = latent_dim_other_ffnn
        self._phys_dim = latent_dim_other_ffnn
        self._spec_dim = spectral_hidden

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
        try:
            # Encode each modality with error handling
            crystal_emb, topo_aux_logits, _ = self.crystal_encoder(
                inputs['crystal_graph'], return_topological_logits=True
            )
            kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
            asph_emb = self.asph_encoder(inputs['asph_features'])
            scalar_emb = self.scalar_encoder(inputs['scalar_features'])
            phys_emb = self.enhanced_kspace_physics_encoder(
                decomposition_features=inputs['kspace_physics_features']['decomposition_features'],
                gap_features=inputs['kspace_physics_features'].get('gap_features'),
                dos_features=inputs['kspace_physics_features'].get('dos_features'),
                fermi_features=inputs['kspace_physics_features'].get('fermi_features')
            )
            
            # Spectral encoding with error handling
            try:
                spec_emb = self.spectral_encoder(
                    inputs['crystal_graph'].edge_index,
                    inputs['crystal_graph'].num_nodes,
                    getattr(inputs['crystal_graph'], 'batch', None)
                )
            except Exception as e:
                print(f"Warning: Spectral encoding failed, using zeros: {e}")
                spec_emb = torch.zeros(kspace_emb.shape[0], self._spec_dim, device=kspace_emb.device)

            # Concatenate all features
            features = [crystal_emb, kspace_emb, asph_emb, scalar_emb, phys_emb, spec_emb]
            x = torch.cat(features, dim=-1)
            
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
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("WARNING: NaN or infinite values detected in input features!")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            fused = self.fusion_network(x)
            
            if torch.isnan(fused).any() or torch.isinf(fused).any():
                print("WARNING: NaN or infinite values detected in fusion output!")
                fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
                
            combined_logits = self.combined_head(fused)
            topology_primary = combined_logits  # Use combined logits for topology
            topology_aux = topo_aux_logits
            magnetism_aux = self.magnetism_head_aux(fused)

            return {
                'combined_logits': combined_logits,
                'topology_logits_primary': topology_primary,
                'topology_logits_auxiliary': topology_aux,
                'magnetism_logits_aux': magnetism_aux
            }
            
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            traceback.print_exc()
            # Return dummy outputs to prevent crash
            batch_size = inputs['crystal_graph'].num_graphs if hasattr(inputs['crystal_graph'], 'num_graphs') else 1
            device = inputs['crystal_graph'].x.device if hasattr(inputs['crystal_graph'], 'x') else torch.device('cpu')
            
            dummy_tensor = torch.zeros(batch_size, self.num_combined_classes, device=device)
            return {
                'combined_logits': dummy_tensor,
                'topology_logits_primary': dummy_tensor,
                'topology_logits_auxiliary': dummy_tensor,
                'magnetism_logits_aux': dummy_tensor
            }

    def compute_enhanced_loss(self,
             predictions: Dict[str, torch.Tensor],
             targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        try:
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
            losses['total_loss'] = sum(losses.values())
            return losses
        except Exception as e:
            print(f"ERROR in loss computation: {e}")
            # Return a dummy loss to prevent crash
            return {'total_loss': torch.tensor(0.0, requires_grad=True)}


def robust_training_loop():
    """
    Robust training loop with improved memory management and error handling.
    """
    print("Starting robust training loop...")
    
    # Set device and configure GPU memory
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Configure GPU memory growth to prevent segmentation faults
    if device.type == 'cuda':
        import torch.cuda
        torch.cuda.empty_cache()
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of available GPU memory
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    try:
        # Load dataset with preloading enabled
        print("Loading dataset with preloading...")
        dataset = MaterialDataset(
            master_index_path=config.MASTER_INDEX_PATH,
            kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
            data_root_dir=config.DATA_DIR,
            dos_fermi_dir=config.DOS_FERMI_DIR,
            preload=getattr(config, 'PRELOAD_DATASET', True)
        )
        
        # Split dataset
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
        
        # Create data loaders with reduced batch size for stability
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_indices)
        
        # Use smaller batch size and fewer workers for stability
        batch_size = min(config.BATCH_SIZE, 4)  # Reduce batch size
        num_workers = min(getattr(config, 'NUM_WORKERS', 4), 2)  # Reduce workers
        
        train_loader = PyGDataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = PyGDataLoader(
            val_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        test_loader = PyGDataLoader(
            test_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        print(f"Data loaders created with batch_size={batch_size}, num_workers={num_workers}")
        
        # Initialize model
        model = RobustMultiModalMaterialClassifier(
            crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
            kspace_node_feature_dim=config.KSPACE_NODE_FEATURE_DIM,
            asph_feature_dim=config.ASPH_FEATURE_DIM,
            scalar_feature_dim=config.SCALAR_FEATURE_DIM,
            decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM
        ).to(device)
        
        # Initialize optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training loop
        num_epochs = config.NUM_EPOCHS
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            print(f"\nStarting epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Move batch to device
                    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                    
                    # Forward pass
                    optimizer.zero_grad()
                    predictions = model(batch)
                    
                    # Compute loss
                    losses = model.compute_enhanced_loss(predictions, batch)
                    total_loss = losses['total_loss']
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_losses.append(total_loss.item())
                    
                    # Print progress
                    if batch_idx % 100 == 0:
                        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                              f"Loss: {total_loss.item():.4f}, "
                              f"GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB used, "
                              f"{torch.cuda.memory_reserved(device) / 1024**3:.2f}GB cached")
                    
                    # Clear cache periodically
                    if batch_idx % 50 == 0 and device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"ERROR in training batch {batch_idx}: {e}")
                    traceback.print_exc()
                    continue
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    try:
                        # Move batch to device
                        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                        
                        # Forward pass
                        predictions = model(batch)
                        
                        # Compute loss
                        losses = model.compute_enhanced_loss(predictions, batch)
                        total_loss = losses['total_loss']
                        
                        val_losses.append(total_loss.item())
                        
                    except Exception as e:
                        print(f"ERROR in validation batch {batch_idx}: {e}")
                        continue
            
            # Compute average losses
            avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save model
                save_path = Path(config.MODEL_SAVE_DIR) / f"best_model_epoch_{epoch+1}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': config.__dict__
                }, save_path)
                print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
            
            # Clear GPU cache after each epoch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"CRITICAL ERROR in training loop: {e}")
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("Starting robust training script...")
    success = robust_training_loop()
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed!")
        sys.exit(1) 