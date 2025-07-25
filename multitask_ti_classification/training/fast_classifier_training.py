'''
Fast classifier training that removes the computationally expensive encoders.
This version removes GraphSpectralEncoder and TopologicalMLEncoder for speed.
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

# Core encoders (FAST VERSION - no spectral or topological ML)
from helper.topological_crystal_encoder import TopologicalCrystalEncoder
from src.model_w_debug import KSpaceTransformerGNNEncoder, ScalarFeatureEncoder
from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures
from encoders.ph_token_encoder import PHTokenEncoder
# REMOVED: GraphSpectralEncoder and TopologicalMLEncoder for speed
import helper.config as config

class FastMultiModalMaterialClassifier(nn.Module):
    """
    Fast multi-modal material classifier without expensive encoders.
    Removes GraphSpectralEncoder and TopologicalMLEncoder for speed.
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
        crystal_encoder_hidden_dim: int = 256,  # Increased
        crystal_encoder_num_layers: int = 6,    # Increased
        crystal_encoder_output_dim: int = 256,  # Increased
        crystal_encoder_radius: float = 5.0,
        crystal_encoder_num_scales: int = 3,
        crystal_encoder_use_topological_features: bool = True,
        # k-space GNN params
        kspace_gnn_hidden_channels: int = 256,  # Increased
        kspace_gnn_num_layers: int = 8,         # Increased
        kspace_gnn_num_heads: int = 16,         # Increased
        latent_dim_gnn: int = 256,              # Increased
        # ASPH & scalar dims
        latent_dim_asph: int = 256,             # Increased
        latent_dim_other_ffnn: int = 256,       # Increased
        # Fusion MLP
        fusion_hidden_dims: List[int] = [2048, 1024, 512, 256, 128],  # Deeper/wider
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        # Store dims for fusion (FAST VERSION - no spectral or topological ML)
        self._crystal_dim  = crystal_encoder_output_dim
        self._kspace_dim   = latent_dim_gnn
        self._asph_dim     = latent_dim_asph
        self._scalar_dim   = latent_dim_other_ffnn
        self._phys_dim     = latent_dim_other_ffnn

        # Instantiate encoders (FAST VERSION)
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

        # REMOVED: GraphSpectralEncoder and TopologicalMLEncoder for speed

        # Fusion MLP - simplified input dimension (no spectral or topological ML)
        total_input_dim = self._crystal_dim + self._kspace_dim + self._asph_dim + self._scalar_dim + self._phys_dim
        print(f"Fast model total input dimension: {total_input_dim}")
        self.fusion_network = self._build_fusion_network(total_input_dim, fusion_hidden_dims, dropout_rate)

        # Output heads
        self.num_combined_classes = num_combined_classes
        self.num_topology_classes = num_topology_classes
        self.num_magnetism_classes = num_magnetism_classes
        self.combined_head = nn.Linear(fusion_hidden_dims[-1], num_combined_classes)
        self.topology_head_aux = nn.Linear(fusion_hidden_dims[-1], num_topology_classes)
        self.magnetism_head_aux = nn.Linear(fusion_hidden_dims[-1], num_magnetism_classes)

    def _build_fusion_network(self, input_dim: int, hidden_dims: List[int], dropout_rate: float) -> nn.Module:
        """Build the fusion MLP network."""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        return nn.Sequential(*layers)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Encode each modality (FAST VERSION - no spectral or topological ML)
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

        # Concatenate all features (FAST VERSION - no spectral or topological ML)
        x = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, phys_emb], dim=-1)
        
        # Fusion
        fused_features = self.fusion_network(x)
        
        # Output heads
        combined_logits = self.combined_head(fused_features)
        topology_logits_aux = self.topology_head_aux(fused_features)
        magnetism_logits_aux = self.magnetism_head_aux(fused_features)
        
        return {
            'combined_logits': combined_logits,
            'topology_logits_primary': topo_aux_logits,  # From crystal encoder
            'topology_logits_auxiliary': topology_logits_aux,
            'magnetism_logits_aux': magnetism_logits_aux,
        }

    def compute_enhanced_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute the multi-task loss."""
        # Primary combined classification loss
        combined_loss = F.cross_entropy(predictions['combined_logits'], targets['combined'])
        
        # Auxiliary topology classification loss
        topology_loss_aux = F.cross_entropy(predictions['topology_logits_auxiliary'], targets['topology'])
        
        # Auxiliary magnetism classification loss
        magnetism_loss_aux = F.cross_entropy(predictions['magnetism_logits_aux'], targets['magnetism'])
        
        # Primary topology loss (from crystal encoder)
        topology_loss_primary = F.cross_entropy(predictions['topology_logits_primary'], targets['topology'])
        
        # Total loss
        total_loss = (combined_loss + 
                     0.1 * topology_loss_aux + 
                     0.1 * magnetism_loss_aux + 
                     0.1 * topology_loss_primary)
        
        return {
            'total_loss': total_loss,
            'combined_loss': combined_loss,
            'topology_loss_aux': topology_loss_aux,
            'magnetism_loss_aux': magnetism_loss_aux,
            'topology_loss_primary': topology_loss_primary,
        }

def main_training_loop():
    """
    Main training loop using the fast model architecture.
    """
    print("Starting FAST training loop...")
    
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
    
    # Initialize fast model
    print("Initializing FAST model...")
    model = FastMultiModalMaterialClassifier(
        crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim=config.ASPH_FEATURE_DIM,
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
        num_combined_classes=config.NUM_COMBINED_CLASSES,
        num_topology_classes=config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes=config.NUM_MAGNETISM_CLASSES
    ).to(device)
    print(f"FAST model initialized successfully on device: {device}")
    
    # Setup optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    print("Starting FAST training...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_losses = []
        
        print(f"Starting epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(train_loader):
            try:
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
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch)
                losses = model.compute_enhanced_loss(outputs, {
                    'combined': batch['combined_label'],
                    'topology': batch['topology_label'],
                    'magnetism': batch['magnetism_label']
                })
                
                # Backward pass
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
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
                continue
        
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
            torch.save(model.state_dict(), config.MODEL_SAVE_DIR / "best_fast_model.pth")
            print(f"New best FAST model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print("FAST training completed!")
    return model

if __name__ == "__main__":
    main_training_loop() 