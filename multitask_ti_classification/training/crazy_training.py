"""
Crazy Training Script - Advanced training with Mixup, CutMix, feature masking,
focal loss, and state-of-the-art optimization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.crazy_fusion_model import create_crazy_fusion_model
from helper.config import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
from tqdm import tqdm


class MixupAugmentation:
    """Mixup data augmentation for multimodal data."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(self, batch_data: Dict[str, torch.Tensor], labels: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
        if self.alpha <= 0:
            return batch_data, labels, labels, 1.0
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Shuffle batch
        batch_size = labels.size(0)
        indices = torch.randperm(batch_size)
        
        # Create mixed data
        mixed_data = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                mixed_data[key] = lam * value + (1 - lam) * value[indices]
            else:
                mixed_data[key] = value
        
        # Create mixed labels
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_data, mixed_labels, labels[indices], lam


class CutMixAugmentation:
    """CutMix data augmentation for multimodal data."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(self, batch_data: Dict[str, torch.Tensor], labels: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
        if self.alpha <= 0:
            return batch_data, labels, labels, 1.0
        
        batch_size = labels.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # For graph data, we'll use a simplified version
        # In practice, you might want to implement more sophisticated graph mixing
        mixed_data = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 2:  # Feature matrices
                    # Simple feature mixing
                    mixed_data[key] = lam * value + (1 - lam) * value[indices]
                else:
                    mixed_data[key] = value
            else:
                mixed_data[key] = value
        
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_data, mixed_labels, labels[indices], lam


class FeatureMasking:
    """Random feature masking for robustness."""
    
    def __init__(self, mask_prob: float = 0.1):
        self.mask_prob = mask_prob
        
    def __call__(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        masked_data = {}
        
        for key, value in batch_data.items():
            # Only apply masking to feature tensors (not edge indices)
            if (isinstance(value, torch.Tensor) and 
                value.dtype == torch.float32 and 
                key in ['crystal_x', 'kspace_x', 'scalar_features', 'decomposition_features']):
                # Create mask
                mask = torch.rand_like(value) > self.mask_prob
                masked_data[key] = value * mask
            else:
                masked_data[key] = value
                
        return masked_data


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CosineAnnealingWarmRestarts(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warm restarts."""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        T_cur = self.last_epoch
        T_i = self.T_0
        
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * T_cur / T_i)) / 2
                for base_lr in self.base_lrs]


class CrazyTrainer:
    """Advanced trainer with all the crazy features."""
    
    def __init__(self, model: nn.Module, config: dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Augmentation
        self.mixup = MixupAugmentation(alpha=config.get('MIXUP_ALPHA', 0.2))
        self.cutmix = CutMixAugmentation(alpha=config.get('CUTMIX_ALPHA', 1.0))
        self.feature_masking = FeatureMasking(mask_prob=config.get('MASK_PROB', 0.1))
        
        # Loss functions
        self.criterion = FocalLoss(
            alpha=config.get('FOCAL_ALPHA', 1.0),
            gamma=config.get('FOCAL_GAMMA', 2.0)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('LEARNING_RATE', 1e-3),
            weight_decay=config.get('WEIGHT_DECAY', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('SCHEDULER_T0', 10),
            T_mult=config.get('SCHEDULER_T_MULT', 2),
            eta_min=config.get('SCHEDULER_ETA_MIN', 1e-6)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Initialize wandb if enabled
        if config.get('USE_WANDB', False) and WANDB_AVAILABLE:
            wandb.init(project="crazy-fusion-model", config=config)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with advanced augmentation."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # Extract labels (use 'topology_label' as main label; change if needed)
            labels = batch['topology_label']

            # Prepare batch_data (exclude label keys)
            batch_data = {k: v for k, v in batch.items() if k not in ['topology_label', 'combined_label', 'magnetism_label']}

            # Apply augmentation with probability (simplified for now)
            lam = 1.0
            labels_b = labels

            # Apply feature masking
            batch_data = self.feature_masking(batch_data)

            # # Print all batch keys and their shapes for debugging
            # print(f"[DEBUG] Batch keys: {list(batch_data.keys())}")
            # for k, v in batch_data.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"[DEBUG] batch_data['{k}'].shape: {v.shape}")
            #     else:
            #         print(f"[DEBUG] batch_data['{k}'] type: {type(v)}")

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)

            # Compute loss
            if lam != 1.0:
                loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(outputs, labels_b)
            else:
                loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                labels = batch['topology_label']  # or 'combined_label' if that's your main label
                batch_data = {k: v for k, v in batch.items() if k not in ['topology_label', 'combined_label', 'magnetism_label']}
                
                # Forward pass
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Compute additional metrics with warning suppression
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, save_path: str = 'best_model.pth'):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
                    # Log to wandb
        if self.config.get('USE_WANDB', False) and WANDB_AVAILABLE:
            wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1': val_metrics['f1'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'config': self.config
                }, save_path)
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def plot_training_curves(self, save_path: str = 'training_curves.png'):
        """Plot training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate curve
        lrs = [self.scheduler.get_last_lr()[0]] * len(self.train_losses)
        ax3.plot(lrs)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # Loss vs LR
        ax4.scatter(lrs, self.train_losses, alpha=0.6, label='Train Loss')
        ax4.scatter(lrs, self.val_losses, alpha=0.6, label='Val Loss')
        ax4.set_title('Loss vs Learning Rate')
        ax4.set_xlabel('Learning Rate')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_best_model(self, model_path: str):
        """Load the best model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        print(f"Loaded best model with validation accuracy: {self.best_val_acc:.2f}%")


def create_crazy_trainer(model: nn.Module, config: dict, device: str = 'cuda') -> CrazyTrainer:
    """Factory function to create a crazy trainer."""
    return CrazyTrainer(model, config, device)


if __name__ == "__main__":
    # Test configuration
    test_config = {
        'HIDDEN_DIM': 256,
        'FUSION_DIM': 512,
        'NUM_CLASSES': 2,
        'USE_CRYSTAL': True,
        'USE_KSPACE': True,
        'USE_SCALAR': True,
        'USE_DECOMPOSITION': True,
        'USE_SPECTRAL': True,
        'CRYSTAL_INPUT_DIM': 92,
        'KSPACE_INPUT_DIM': 2,
        'SCALAR_INPUT_DIM': 4763,
        'DECOMPOSITION_INPUT_DIM': 100,
        'K_EIGS': 64,
        'CRYSTAL_LAYERS': 4,
        'KSPACE_LAYERS': 3,
        'SCALAR_BLOCKS': 3,
        'FUSION_BLOCKS': 3,
        'FUSION_HEADS': 8,
        'KSPACE_GNN_TYPE': 'transformer',
        'MIXUP_ALPHA': 0.2,
        'CUTMIX_ALPHA': 1.0,
        'MASK_PROB': 0.1,
        'FOCAL_ALPHA': 1.0,
        'FOCAL_GAMMA': 2.0,
        'LEARNING_RATE': 1e-3,
        'WEIGHT_DECAY': 1e-4,
        'SCHEDULER_T0': 10,
        'SCHEDULER_T_MULT': 2,
        'SCHEDULER_ETA_MIN': 1e-6,
        'USE_WANDB': False
    }
    
    # Create model and trainer
    model = create_crazy_fusion_model(test_config)
    trainer = create_crazy_trainer(model, test_config)
    
    print(f"Crazy trainer created with {sum(p.numel() for p in model.parameters()):,} parameters")


def main_training_loop():
    """Main training loop using the new Crazy Fusion Model architecture with MaterialDataset pipeline."""
    print("Starting main training loop (MaterialDataset pipeline)...")

    # Import necessary modules
    from helper.dataset import MaterialDataset, custom_collate_fn
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split
    import helper.config as config
    from torch.utils.data import Subset

    # Set random seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Set device
    device = config.DEVICE
    print(f"Using device: {device}")

    # Load dataset with preloading enabled
    print("Loading dataset with preloading...")
    dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR,
        preload=getattr(config, 'PRELOAD_DATASET', True)
    )

    if len(dataset) == 0:
        raise RuntimeError("No data found! Please check your data paths and preprocessing.")

    # Split dataset (stratified by combined_label if possible)
    try:
        combined_labels = [dataset[i]['combined_label'].item() for i in range(len(dataset))]
        train_indices, temp_indices = train_test_split(
            range(len(dataset)),
            test_size=0.3,
            random_state=config.SEED,
            stratify=combined_labels
        )
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=0.5,
            random_state=config.SEED,
            stratify=[combined_labels[i] for i in temp_indices]
        )
    except Exception as e:
        print(f"Warning: Could not stratify dataset split due to error: {e}")
        print("Falling back to random split without stratification.")
        train_indices, temp_indices = train_test_split(
            range(len(dataset)),
            test_size=0.3,
            random_state=config.SEED
        )
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=0.5,
            random_state=config.SEED
        )

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

    # Create model
    model = create_crazy_fusion_model(vars(config))
    print(f"Created Crazy Fusion Model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer with appropriate device
    trainer = CrazyTrainer(model, vars(config), device=device)

    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=config.NUM_EPOCHS, save_path='best_crazy_model.pth')

    # Plot training curves
    trainer.plot_training_curves('crazy_training_curves.png')

    print("Training completed successfully!") 