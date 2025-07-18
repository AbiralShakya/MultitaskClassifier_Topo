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
from sklearn.metrics import accuracy_score

# Core encoders
from helper.topological_crystal_encoder import TopologicalCrystalEncoder
from src.model_w_debug import KSpaceTransformerGNNEncoder, ScalarFeatureEncoder
from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures
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

# Import for attention mechanism
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for better feature fusion."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + output)
        
        return output

def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def add_feature_noise(features, noise_std=0.01):
    """Add small noise to features for regularization."""
    noise = torch.randn_like(features) * noise_std
    return features + noise

class FocalLoss(nn.Module):
    """Focal Loss for better handling of class imbalance."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnhancedMultiModalMaterialClassifier(nn.Module):
    def __init__(
        self,
        # Feature dims
        crystal_node_feature_dim: int,
        kspace_node_feature_dim: int,
        scalar_feature_dim: int,
        decomposition_feature_dim: int,
        # Class counts
        num_topology_classes: int = config.NUM_TOPOLOGY_CLASSES,
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
        # Scalar dims
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
        
        # NEW: Attention mechanism for better feature fusion
        self.attention_heads = 8
        self.attention_dropout = 0.1
        
        # NEW: Data augmentation parameters
        self.use_mixup = True
        self.mixup_alpha = 0.2
        self.feature_noise_std = 0.01
        self.training = True

        # Output heads - initialize with placeholder input dim (will update in forward if needed)
        self.num_topology_classes = num_topology_classes
        self.topology_head = nn.Linear(1, self.num_topology_classes)
        
        # NEW: Ensemble heads for better accuracy
        self.ensemble_heads = []
        self.num_ensemble_heads = 3

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        try:
            # Encode each modality
            crystal_emb, _, _ = self.crystal_encoder(
                inputs['crystal_graph'], return_topological_logits=False
            )
            kspace_emb    = self.kspace_encoder(inputs['kspace_graph'])
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
            features = [crystal_emb, kspace_emb, scalar_emb, phys_emb, spec_emb]
            if ml_emb is not None:
                features.append(ml_emb)
            x = torch.cat(features, dim=-1)
            
            # --- ADD FEATURE NORMALIZATION ---
            # Normalize features to prevent gradient explosion
            x = F.layer_norm(x, x.shape[1:])
            
            # --- ADD DATA AUGMENTATION ---
            if self.training:
                # Add feature noise for regularization
                x = add_feature_noise(x, self.feature_noise_std)
            # ---------------------------------------------------
            
            # --- ADD DEBUGGING FOR FIRST FEW BATCHES ---
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            if self._debug_count < 3:
                print(f"[DEBUG] Feature stats - min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}, std: {x.std():.4f}")
                self._debug_count += 1
            # ---------------------------------------------------

            # Dynamically create fusion network if not exists
            if self.fusion_network is None:
                actual_input_dim = x.shape[1]
                print(f"Creating fusion network with input dimension: {actual_input_dim}")
                
                # Create attention mechanism
                self.attention = MultiHeadAttention(
                    d_model=actual_input_dim,
                    num_heads=self.attention_heads,
                    dropout=self.attention_dropout
                ).to(x.device)
                
                # Create fusion MLP
                layers = []
                in_dim = actual_input_dim
                for h in self.fusion_hidden_dims:
                    layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(self.dropout_rate)]
                    in_dim = h
                self.fusion_network = nn.Sequential(*layers).to(x.device)
                
                # Update output heads to match new input dim
                self.topology_head = nn.Linear(in_dim, self.num_topology_classes).to(x.device)
                
                # Create ensemble heads
                self.ensemble_heads = []
                for i in range(self.num_ensemble_heads):
                    head = nn.Sequential(
                        nn.Linear(in_dim, in_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(in_dim // 2, self.num_topology_classes)
                    ).to(x.device)
                    self.ensemble_heads.append(head)

            # Apply attention mechanism for better feature interaction
            x_reshaped = x.unsqueeze(1)  # Add sequence dimension for attention
            x_attended = self.attention(x_reshaped)
            x = x_attended.squeeze(1)  # Remove sequence dimension
            
            fused = self.fusion_network(x)
            
            # --- ADD GRADIENT CLIPPING TO FUSED FEATURES ---
            fused = torch.clamp(fused, -10, 10)  # Prevent extreme values
            
            # --- ADD DEBUGGING FOR FUSED FEATURES ---
            if self._debug_count < 3:
                print(f"[DEBUG] Fused stats - min: {fused.min():.4f}, max: {fused.max():.4f}, mean: {fused.mean():.4f}, std: {fused.std():.4f}")
            # ---------------------------------------------------

            # Main head
            main_logits = self.topology_head(fused)
            
            # Ensemble heads
            ensemble_logits = []
            for head in self.ensemble_heads:
                ensemble_logits.append(head(fused))
            
            # Average ensemble predictions
            ensemble_logits = torch.stack(ensemble_logits)
            ensemble_logits = torch.mean(ensemble_logits, dim=0)
            
            # Combine main and ensemble predictions
            final_logits = 0.7 * main_logits + 0.3 * ensemble_logits

            return {
                'logits': final_logits,
                'main_logits': main_logits,
                'ensemble_logits': ensemble_logits
            }
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_mixup:
            # Apply mixup
            mixed_features, targets_a, targets_b, lam = mixup_data(
                predictions['logits'], targets, self.mixup_alpha
            )
            main_loss = mixup_criterion(F.cross_entropy, mixed_features, targets_a, targets_b, lam)
            
            # Add ensemble loss
            ensemble_loss = 0
            for i in range(self.num_ensemble_heads):
                ensemble_loss += F.cross_entropy(predictions['ensemble_logits'], targets, label_smoothing=0.1)
            ensemble_loss /= self.num_ensemble_heads
            
            return main_loss + 0.1 * ensemble_loss
        else:
            main_loss = F.cross_entropy(predictions['logits'], targets, label_smoothing=0.1)
            
            # Add ensemble loss
            ensemble_loss = 0
            for i in range(self.num_ensemble_heads):
                ensemble_loss += F.cross_entropy(predictions['ensemble_logits'], targets, label_smoothing=0.1)
            ensemble_loss /= self.num_ensemble_heads
            
            return main_loss + 0.1 * ensemble_loss


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
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
        num_topology_classes=config.NUM_TOPOLOGY_CLASSES
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
    
    # Setup optimizer and scheduler using config values
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4, eps=1e-8, betas=(0.9, 0.999))
    
    # Advanced learning rate scheduling with warmup
    def warmup_cosine_schedule(epoch):
        if epoch < 10:  # Warmup for first 10 epochs
            return epoch / 10
        else:  # Cosine annealing
            return 0.5 * (1 + math.cos(math.pi * (epoch - 10) / (config.NUM_EPOCHS - 10)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
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
        train_correct = 0
        train_total = 0
        
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
                losses = model.compute_loss(outputs, batch['topology_label'])
                
                # Compute accuracy for this batch
                preds = outputs['logits'].argmax(dim=1).detach().cpu().numpy()
                targets = batch['topology_label'].detach().cpu().numpy()
                batch_acc = accuracy_score(targets, preds)
                train_correct += (preds == targets).sum()
                train_total += len(targets)
                
                # Backward pass
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                train_losses.append(losses.item())
                
                if batch_idx % 10 == 0:
                    # Monitor GPU memory usage
                    if device.type == 'cuda':
                        gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**3
                        gpu_memory_cached = torch.cuda.memory_reserved(device) / 1024**3
                        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, "
                              f"Loss: {losses.item():.4f}, "
                              f"Acc: {batch_acc:.4f}, "
                              f"GPU Memory: {gpu_memory_used:.2f}GB used, {gpu_memory_cached:.2f}GB cached")
                    else:
                        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, "
                              f"Loss: {losses.item():.4f}, Acc: {batch_acc:.4f}")
                
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
        val_correct = 0
        val_total = 0
        
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
                targets = batch['topology_label']
                losses = model.compute_loss(outputs, targets)
                val_losses.append(losses.item())
                # Compute accuracy for this batch
                preds = outputs['logits'].argmax(dim=1).detach().cpu().numpy()
                targs = batch['topology_label'].detach().cpu().numpy()
                val_correct += (preds == targs).sum()
                val_total += len(targs)
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        # Append to loss history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
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
        scheduler.step()
        
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
    
    # --- EVALUATE ON VALIDATION AND TEST SETS ---
    from sklearn.metrics import f1_score, confusion_matrix, classification_report
    def evaluate(loader, name):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in loader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                    elif hasattr(batch[key], 'batch'):
                        batch[key] = batch[key].to(device)
                    elif isinstance(batch[key], dict):
                        for sub_key in batch[key]:
                            if isinstance(batch[key][sub_key], torch.Tensor):
                                batch[key][sub_key] = batch[key][sub_key].to(device)
                outputs = model(batch)
                preds = outputs['logits'].argmax(dim=1).detach().cpu().numpy()
                targs = batch['topology_label'].detach().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targs)
        print(f"\n=== {name.upper()} SET RESULTS ===")
        print(f"Accuracy: {np.mean(np.array(all_preds) == np.array(all_targets)):.4f}")
        print(f"F1 Score (macro): {f1_score(all_targets, all_preds, average='macro'):.4f}")
        print(f"F1 Score (per class): {f1_score(all_targets, all_preds, average=None)}")
        print(f"Confusion Matrix:\n{confusion_matrix(all_targets, all_preds)}")
        print(classification_report(all_targets, all_preds, digits=4))
    
    evaluate(val_loader, "validation")
    evaluate(test_loader, "test")
    
    # --- GPU MEMORY USAGE REPORT ---
    if device.type == 'cuda':
        max_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"\n[GPU] Max memory used during training: {max_mem:.2f} GB")
        if max_mem < 1.0:
            print("[WARNING] GPU memory usage is very low (<1GB). This may indicate your model or batch size is too small, or tensors are not being moved to the GPU. Consider increasing batch size or model complexity if appropriate.")
    
    return model
