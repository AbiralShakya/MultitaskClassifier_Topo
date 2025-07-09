# checkpoint_training.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Checkpoint-based training script that can recover from segmentation faults.
Saves progress every few epochs and can restart from the last checkpoint.
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
import pickle
from typing import Dict, Any, List

# Import the existing classifier
from classifier_training import EnhancedMultiModalMaterialClassifier

# Import dataset
from helper.dataset import MaterialDataset, custom_collate_fn
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Subset

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, train_losses, val_losses, 
                   train_indices, val_indices, test_indices, checkpoint_dir):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
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
        'config': config.__dict__
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save indices separately for easy access
    indices_path = checkpoint_dir / f"indices_epoch_{epoch}.pkl"
    with open(indices_path, 'wb') as f:
        pickle.dump({
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }, f)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint."""
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

def find_latest_checkpoint(checkpoint_dir):
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

def checkpoint_training_loop(checkpoint_dir="./checkpoints", checkpoint_frequency=5):
    """
    Checkpoint-based training loop that can recover from crashes.
    """
    print("Starting checkpoint-based training loop...")
    
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
    
    # Check for existing checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_indices, val_indices, test_indices = None, None, None
    
    if latest_checkpoint is not None:
        print(f"Found existing checkpoint: {latest_checkpoint}")
        response = input("Do you want to resume from this checkpoint? (y/n): ")
        if response.lower() == 'y':
            # Load dataset first
            dataset = MaterialDataset(
                master_index_path=config.MASTER_INDEX_PATH,
                kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
                data_root_dir=config.DATA_DIR,
                dos_fermi_dir=config.DOS_FERMI_DIR,
                preload=getattr(config, 'PRELOAD_DATASET', True)
            )
            
            # Initialize model
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
            
            # Initialize optimizer and scheduler
            optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
            
            # Load checkpoint
            start_epoch, best_val_loss, train_losses, val_losses, train_indices, val_indices, test_indices = load_checkpoint(
                latest_checkpoint, model, optimizer, scheduler
            )
            start_epoch += 1  # Start from next epoch
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("Starting fresh training...")
    
    # If no checkpoint or user chose not to resume, start fresh
    if train_indices is None:
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
            
            # Initialize model
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
            
            # Initialize optimizer and scheduler
            optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
            
        except Exception as e:
            print(f"ERROR loading dataset or initializing model: {e}")
            return False
    
    # Create data loaders
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
    
    print(f"Data loaders created. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # Training loop
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nStarting epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Training phase
        model.train()
        epoch_train_losses = []
        
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
                epoch_train_losses.append(losses['total_loss'].item())
                
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
                if batch_idx % 25 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    # Clear spectral encoder cache to prevent memory buildup
                    if hasattr(model.spectral_encoder, 'clear_cache'):
                        model.spectral_encoder.clear_cache()
                    # Force garbage collection
                    gc.collect()
                    
            except Exception as e:
                print(f"ERROR in training batch {batch_idx}: {e}")
                continue
        
        # Validation phase
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
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
                    
                    outputs = model(batch)
                    targets = {
                        'combined': batch['combined_label'],
                        'topology': batch['topology_label'],
                        'magnetism': batch['magnetism_label']
                    }
                    losses = model.compute_enhanced_loss(outputs, targets)
                    epoch_val_losses.append(losses['total_loss'].item())
                    
                except Exception as e:
                    print(f"ERROR in validation batch: {e}")
                    continue
        
        avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else float('inf')
        avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else float('inf')
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Clean up GPU memory between epochs
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            # Clear spectral encoder cache to prevent memory buildup
            if hasattr(model.spectral_encoder, 'clear_cache'):
                model.spectral_encoder.clear_cache()
            # Force garbage collection
            gc.collect()
            print(f"GPU memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save checkpoint every few epochs
        if (epoch + 1) % checkpoint_frequency == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, 
                          train_losses, val_losses, train_indices, val_indices, test_indices, checkpoint_dir)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config.MODEL_SAVE_DIR / "best_model.pth")
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print("Training completed!")
    return True

if __name__ == "__main__":
    print("Starting checkpoint-based training script...")
    success = checkpoint_training_loop()
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed!")
        sys.exit(1) 