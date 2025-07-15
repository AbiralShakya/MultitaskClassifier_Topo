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
# GPU-accelerated spectral graph features
from helper.gpu_spectral_encoder import GPUSpectralEncoder, FastSpectralEncoder
# REMOVED: Topological ML encoder (expensive synthetic Hamiltonian generation)
# from src.topological_ml_encoder import (
#     TopologicalMLEncoder, TopologicalMLEncoder2D,
#     create_hamiltonian_from_features, compute_topological_loss
# )
import helper.config as config
import pickle
import gc

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
        # Use GPU-accelerated spectral encoder (much faster than CPU SciPy)
        self.spectral_encoder = GPUSpectralEncoder(
            k_eigs=config.K_LAPLACIAN_EIGS,
            hidden=spectral_hidden
        )

        # REMOVED: Topological ML encoder (expensive synthetic Hamiltonian generation)
        self.use_topo_ml = False  # Disable topological ML
        self.topo_ml_aux_weight = 0.0  # No auxiliary weight needed

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
        # Smart spectral encoding with caching
        try:
            spec_emb = self.spectral_encoder(
                inputs['crystal_graph'].edge_index,
                inputs['crystal_graph'].num_nodes,
                getattr(inputs['crystal_graph'], 'batch', None)
            )
        except Exception as e:
            print(f"Warning: Spectral encoding failed, using zeros: {e}")
            spec_emb = torch.zeros(kspace_emb.shape[0], self._spec_dim, device=kspace_emb.device) 

        # REMOVED: Topological ML features (expensive synthetic Hamiltonian generation)
        ml_emb, ml_logits = None, None

        # Concatenate all
        features = [crystal_emb, kspace_emb, asph_emb, scalar_emb, phys_emb, spec_emb]
        if ml_emb is not None:
            features.append(ml_emb)
        x = torch.cat(features, dim=-1)
        
        # Feature dimensions for debugging (commented out for speed)
        # print(f"DEBUG - Actual concatenated features shape: {x.shape}")
        # print(f"DEBUG - Expected base_dim: {self._crystal_dim + self._kspace_dim + self._asph_dim + self._scalar_dim + self._phys_dim + self._spec_dim + self._topo_ml_dim}")

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
        # REMOVED: Topological ML loss computation (expensive synthetic Hamiltonian generation)
        losses['total_loss'] = sum(losses.values())
        return losses


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, train_losses, val_losses, 
                   train_indices, val_indices, test_indices, checkpoint_dir="./checkpoints"):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear spectral encoder cache before saving to avoid PyCapsule serialization issues
    if hasattr(model, 'spectral_encoder') and hasattr(model.spectral_encoder, 'clear_cache'):
        model.spectral_encoder.clear_cache()
    
    # Create a clean config dict without non-serializable objects
    config_dict = {}
    for key, value in config.__dict__.items():
        try:
            # Test if the value can be pickled
            pickle.dumps(value)
            config_dict[key] = value
        except (TypeError, pickle.PicklingError):
            # Skip non-serializable objects
            config_dict[key] = f"<non-serializable: {type(value).__name__}>"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'config': config_dict
    }
    
    # Save checkpoint atomically to prevent corruption
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    temp_path = checkpoint_dir / f"temp_checkpoint_epoch_{epoch}.pt"
    
    try:
        # Save to temporary file first
        torch.save(checkpoint, temp_path)
        # Then move to final location (atomic operation)
        temp_path.replace(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        # Clean up temporary file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise e
    
    # Also save indices separately for easy access
    indices_path = checkpoint_dir / f"indices_epoch_{epoch}.pkl"
    temp_indices_path = checkpoint_dir / f"temp_indices_epoch_{epoch}.pkl"
    
    try:
        with open(temp_indices_path, 'wb') as f:
            pickle.dump({
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices
            }, f)
        temp_indices_path.replace(indices_path)
    except Exception as e:
        print(f"Error saving indices: {e}")
        if temp_indices_path.exists():
            temp_indices_path.unlink()
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint with error handling."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_indices = checkpoint['train_indices']
        val_indices = checkpoint['val_indices']
        test_indices = checkpoint['test_indices']
        
        print(f"Checkpoint loaded from epoch {epoch} with best val loss: {best_val_loss:.4f}")
        return epoch, best_val_loss, train_losses, val_losses, train_indices, val_indices, test_indices
        
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        print("Checkpoint file appears to be corrupted. Attempting to find previous checkpoint...")
        
        # Try to find a previous checkpoint
        checkpoint_dir = Path(checkpoint_path).parent
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove the corrupted checkpoint
        try:
            Path(checkpoint_path).unlink()
            print(f"Removed corrupted checkpoint: {checkpoint_path}")
        except:
            pass
        
        # Try the previous checkpoint
        if len(checkpoint_files) > 1:
            previous_checkpoint = checkpoint_files[-2]  # Second to last
            print(f"Trying previous checkpoint: {previous_checkpoint}")
            return load_checkpoint(previous_checkpoint, model, optimizer, scheduler)
        else:
            print("No previous checkpoint found. Starting fresh training.")
            raise ValueError("No valid checkpoint found")

def find_latest_checkpoint(checkpoint_dir="./checkpoints"):
    """Find the latest checkpoint file."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoint_files[-1]

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
    
    # Checkpoint functionality
    checkpoint_dir = "./checkpoints"
    checkpoint_frequency = 5  # Save every 5 epochs
    start_epoch = 0
    train_losses = []
    val_losses = []
    
    # Check for existing checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        print(f"Found existing checkpoint: {latest_checkpoint}")
        print("Resuming from checkpoint...")
        try:
            start_epoch, best_val_loss, train_losses, val_losses, train_indices, val_indices, test_indices = load_checkpoint(
                latest_checkpoint, model, optimizer, scheduler
            )
            start_epoch += 1  # Start from next epoch
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh training...")
            start_epoch = 0
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
    else:
        print("No checkpoint found. Starting fresh training...")
        start_epoch = 0
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
    
    # Training loop
    print("Starting training...")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
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
                
                # Clear cache periodically to prevent memory buildup
                if batch_idx % 25 == 0 and device.type == 'cuda':  # More frequent cleanup
                    torch.cuda.empty_cache()
                    # Clear spectral encoder cache to prevent memory buildup (if available)
                    if hasattr(model, 'spectral_encoder') and model.spectral_encoder is not None and hasattr(model.spectral_encoder, 'clear_cache'):
                        model.spectral_encoder.clear_cache()
                    # Force garbage collection
                    import gc
                    gc.collect()
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
        
        # Append to loss history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % checkpoint_frequency == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, 
                          train_losses, val_losses, train_indices, val_indices, test_indices, checkpoint_dir)
        
        # Clean up GPU memory between epochs
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            # Clear spectral encoder cache to prevent memory buildup (if available)
            if hasattr(model, 'spectral_encoder') and model.spectral_encoder is not None and hasattr(model.spectral_encoder, 'clear_cache'):
                model.spectral_encoder.clear_cache()
            # Force garbage collection
            import gc
            gc.collect()
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
