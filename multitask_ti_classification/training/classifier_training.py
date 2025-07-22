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
# Enhanced integrated model with all improvements
from src.enhanced_integrated_model import EnhancedIntegratedMaterialClassifier, create_enhanced_model
from helper.enhanced_topological_loss import EnhancedTopologicalLoss, FocalLoss
from encoders.enhanced_graph_construction import EnhancedGraphConstructor

# Legacy imports for backward compatibility
from helper.topological_crystal_encoder import TopologicalCrystalEncoder
from src.model_w_debug import KSpaceTransformerGNNEncoder, ScalarFeatureEncoder
from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures
from encoders.asph_encoder import ASPHEncoder
from helper.gpu_spectral_encoder import GPUSpectralEncoder, FastSpectralEncoder
from src.topological_ml_encoder import (
    TopologicalMLEncoder, TopologicalMLEncoder2D,
    create_hamiltonian_from_features, compute_topological_loss
)
import helper.config as config
import pickle
import gc

# Use the enhanced integrated model with all improvements
from src.enhanced_integrated_model import EnhancedIntegratedMaterialClassifier

# Import optimized classifier
from models.optimized_classifier import OptimizedMaterialClassifier, EarlyStopping, CosineWarmupScheduler

# Optimized Multi-Modal Material Classifier for 92%+ accuracy
class EnhancedMultiModalMaterialClassifier(OptimizedMaterialClassifier):
    """
    Optimized classifier integrated into existing training workflow.
    Designed for 92%+ accuracy without overfitting.
    """
    
    def __init__(
        self,
        # Feature dims
        crystal_node_feature_dim: int,
        kspace_node_feature_dim: int,
        scalar_feature_dim: int,
        decomposition_feature_dim: int,
        # Class counts - ONLY topology, no magnetism
        num_topology_classes: int = config.NUM_TOPOLOGY_CLASSES,
        # Legacy parameters (for compatibility)
        crystal_encoder_hidden_dim: int = 256,
        crystal_encoder_num_layers: int = 4,
        crystal_encoder_output_dim: int = 256,
        kspace_gnn_hidden_channels: int = 256,
        kspace_gnn_num_layers: int = 4,
        kspace_gnn_num_heads: int = 8,
        latent_dim_gnn: int = 256,
        latent_dim_other_ffnn: int = 256,
        fusion_hidden_dims: List[int] = [1024, 512, 256],
        dropout_rate: float = 0.3,
        spectral_hidden: int = 128,
        use_topo_ml: bool = False,  # Disabled for stability
        **kwargs
    ):
        # Initialize optimized classifier with proper parameters
        super().__init__(
            crystal_node_feature_dim=3,  # Actual crystal dimension
            kspace_node_feature_dim=kspace_node_feature_dim,
            scalar_feature_dim=scalar_feature_dim,
            decomposition_feature_dim=decomposition_feature_dim,
            num_topology_classes=num_topology_classes,
            hidden_dim=crystal_encoder_hidden_dim,
            dropout_rate=dropout_rate
        )
        
        # Store legacy parameters for compatibility
        self.num_topology_classes = num_topology_classes
        self.hidden_dim = crystal_encoder_hidden_dim
        self.use_topo_ml = use_topo_ml
        
        # Initialize essential encoders for compatibility
        self._init_legacy_encoders(
            crystal_node_feature_dim, kspace_node_feature_dim, 
            scalar_feature_dim, decomposition_feature_dim,
            crystal_encoder_hidden_dim, kspace_gnn_hidden_channels,
            kspace_gnn_num_layers, kspace_gnn_num_heads,
            latent_dim_gnn, latent_dim_other_ffnn
        )
        
        # Add attention mechanism for feature fusion
        total_feature_dim = crystal_encoder_hidden_dim + latent_dim_gnn + latent_dim_other_ffnn + latent_dim_other_ffnn + (crystal_encoder_hidden_dim // 2)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=total_feature_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, crystal_encoder_hidden_dim),
            nn.BatchNorm1d(crystal_encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(crystal_encoder_hidden_dim, crystal_encoder_hidden_dim // 2),
            nn.BatchNorm1d(crystal_encoder_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(crystal_encoder_hidden_dim // 2, num_topology_classes)
        )
    
    def _init_legacy_encoders(self, crystal_node_dim, kspace_node_dim, scalar_dim, 
                      decomp_dim, crystal_hidden, kspace_hidden, kspace_layers,
                      kspace_heads, kspace_out, scalar_out):
        """Initialize essential encoders for compatibility"""
        
        # Crystal encoder - use actual 3D input dimension
        from helper.topological_crystal_encoder import TopologicalCrystalEncoder
        self.crystal_encoder = TopologicalCrystalEncoder(
            node_feature_dim=3,  # Actual crystal graph node features
            hidden_dim=crystal_hidden,
            num_layers=4,
            output_dim=crystal_hidden,
            radius=5.0,
            num_scales=3,
            use_topological_features=True
        )
        
        # K-space encoder
        from src.model_w_debug import KSpaceTransformerGNNEncoder
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=kspace_node_dim,
            hidden_dim=kspace_hidden,
            out_channels=kspace_out,
            n_layers=kspace_layers,
            num_heads=kspace_heads
        )
        
        # Scalar encoder
        from src.model_w_debug import ScalarFeatureEncoder
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=scalar_dim,
            hidden_dims=[scalar_out * 2, scalar_out],
            out_channels=scalar_out
        )
        
        # Physics encoder
        from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures
        self.physics_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=decomp_dim,
            gap_features_dim=getattr(config, 'BAND_GAP_SCALAR_DIM', 1),
            dos_features_dim=getattr(config, 'DOS_FEATURE_DIM', 500),
            fermi_features_dim=getattr(config, 'FERMI_FEATURE_DIM', 1),
            output_dim=scalar_out
        )
        
        # ASPH encoder - use ASPHEncoder with correct dimension (3115)
        from encoders.asph_encoder import ASPHEncoder
        self.asph_encoder = ASPHEncoder(
            input_dim=3115,  # Actual ASPH feature dimension
            hidden_dims=crystal_hidden,
            out_dim=crystal_hidden // 2
        )
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass with input format compatibility"""
        
        # Encode each modality using existing encoders
        crystal_emb, _, _ = self.crystal_encoder(
            inputs['crystal_graph'], return_topological_logits=False
        )
        
        kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
        
        scalar_emb = self.scalar_encoder(inputs['scalar_features'])
        
        phys_emb = self.physics_encoder(
            decomposition_features=inputs['kspace_physics_features']['decomposition_features'],
            gap_features=inputs['kspace_physics_features'].get('gap_features'),
            dos_features=inputs['kspace_physics_features'].get('dos_features'),
            fermi_features=inputs['kspace_physics_features'].get('fermi_features')
        )
        
        # ASPH features
        asph_emb = self.asph_encoder(inputs['asph_features'])
        
        # Concatenate all features
        features = [crystal_emb, kspace_emb, scalar_emb, phys_emb, asph_emb]
        x = torch.cat(features, dim=-1)
        
        # Apply self-attention for feature interaction
        features_unsqueezed = x.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.attention(
            features_unsqueezed, features_unsqueezed, features_unsqueezed
        )
        features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Add residual connection
        features = features + x
        
        # Classification
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features,
            'crystal_emb': crystal_emb,
            'kspace_emb': kspace_emb,
            'scalar_emb': scalar_emb,
            'phys_emb': phys_emb,
            'asph_emb': asph_emb
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Enhanced loss with focal loss and regularization"""
        return super().compute_loss(predictions, targets, alpha=0.1)


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
    # Get topology labels for stratification (BINARY ONLY)
    try:
        topology_labels = [dataset[i]['topology_label'].item() for i in range(len(dataset))]
        print(f"Topology label distribution: {np.bincount(topology_labels)}")
        
        train_indices, temp_indices = train_test_split(
            range(len(dataset)), 
            test_size=0.3, 
            random_state=42,
            stratify=topology_labels
        )
        val_indices, test_indices = train_test_split(
            temp_indices, 
            test_size=0.5, 
            random_state=42,
            stratify=[topology_labels[i] for i in temp_indices]
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
    
    # Setup optimized optimizer and scheduler
    from torch.optim import AdamW
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Use cosine warmup scheduler for better convergence
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=config.NUM_EPOCHS,
        eta_min=1e-6
    )
    
    # Enhanced early stopping
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        min_delta=0.001,
        restore_best_weights=True
    )
    
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
        # Update learning rate with cosine warmup
        current_lr = scheduler.step(epoch)
        
        # Training phase
        model.train()
        epoch_train_losses = []
        train_correct = 0
        train_total = 0
        
        print(f"Starting epoch {epoch+1}/{config.NUM_EPOCHS} (LR: {current_lr:.2e})")
        
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
                
                # Backward pass with gradient clipping
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_losses.append(losses.item())
                
                if batch_idx % 20 == 0:
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
                if batch_idx % 25 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            except Exception as e:
                print(f"ERROR in training batch {batch_idx}: {e}")
                continue
        
        # Validation phase
        model.eval()
        epoch_val_losses = []
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []
        
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
                epoch_val_losses.append(losses.item())
                
                # Compute accuracy for this batch
                preds = outputs['logits'].argmax(dim=1).detach().cpu().numpy()
                targs = batch['topology_label'].detach().cpu().numpy()
                val_correct += (preds == targs).sum()
                val_total += len(targs)
                
                all_val_preds.extend(preds)
                all_val_targets.extend(targs)
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Calculate F1 score for validation
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_val_targets, all_val_preds, average='macro')
        
        # Append to loss history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % checkpoint_frequency == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, 
                          train_losses, val_losses, train_indices, val_indices, test_indices, checkpoint_dir)
        
        # Clean up GPU memory between epochs
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print(f"GPU memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        
        # Enhanced early stopping with best model restoration
        if early_stopping(avg_val_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save best model based on validation accuracy
        if val_acc > best_val_loss:  # Using val_acc instead of loss for best model
            best_val_loss = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_DIR / "best_model.pth")
            print(f"✅ New best model saved with validation accuracy: {best_val_loss:.4f}")
        
        print("-" * 60)
    
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
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"F1 Score (per class): {f1_score(all_targets, all_preds, average=None)}")
        print(f"Confusion Matrix:\n{confusion_matrix(all_targets, all_preds)}")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(classification_report(all_targets, all_preds, digits=4))
        
        return accuracy, f1_macro
    
    # Load best model for final evaluation
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_DIR / "best_model.pth"))
        print("Loaded best model for final evaluation")
    except:
        print("Could not load best model, using current model")
    
    evaluate(val_loader, "validation")
    test_acc, test_f1 = evaluate(test_loader, "test")
    
    # Success metrics
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    if test_acc >= 0.92:
        print("🎉 SUCCESS: Achieved 92%+ test accuracy!")
    else:
        print(f"📈 Progress made. Need {(0.92 - test_acc)*100:.1f}% more accuracy.")
    
    return test_acc, test_f1
    
    # --- GPU MEMORY USAGE REPORT ---
    if device.type == 'cuda':
        max_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"\n[GPU] Max memory used during training: {max_mem:.2f} GB")
        if max_mem < 1.0:
            print("[WARNING] GPU memory usage is very low (<1GB). This may indicate your model or batch size is too small, or tensors are not being moved to the GPU. Consider increasing batch size or model complexity if appropriate.")
    
    return test_acc, test_f1
