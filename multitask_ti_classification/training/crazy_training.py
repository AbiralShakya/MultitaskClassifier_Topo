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
        
        for batch_idx, (batch_data, labels) in enumerate(pbar):
            # Move to device
            batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            labels = labels.to(self.device)
            
            # Apply augmentation with probability (simplified for now)
            lam = 1.0
            labels_b = labels
            
            # Apply feature masking
            batch_data = self.feature_masking(batch_data)
            
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
            for batch_data, labels in val_loader:
                # Move to device
                batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
                labels = labels.to(self.device)
                
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
        
        # Compute additional metrics
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
        'SCALAR_INPUT_DIM': 200,
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
    """Main training loop using the new Crazy Fusion Model architecture."""
    
    # Import real data loaders
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
    try:
        from real_data_loaders import create_data_loaders, get_class_weights
        REAL_DATA_AVAILABLE = True
    except ImportError:
        print("⚠️  Real data loaders not available, will use dummy data")
        REAL_DATA_AVAILABLE = False
    
    # Configuration for the new architecture
    config_dict = {
        # Model architecture
        'HIDDEN_DIM': 256,
        'FUSION_DIM': 512,
        'NUM_CLASSES': 3,  # 3 classes: trivial/semimetal/topological-insulator
        'USE_CRYSTAL': True,
        'USE_KSPACE': True,
        'USE_SCALAR': True,
        'USE_DECOMPOSITION': True,
        'USE_SPECTRAL': True,
        'CRYSTAL_INPUT_DIM': 92,
        'KSPACE_INPUT_DIM': 2,
        'SCALAR_INPUT_DIM': 200,
        'DECOMPOSITION_INPUT_DIM': 100,
        'K_EIGS': 64,
        'CRYSTAL_LAYERS': 4,
        'KSPACE_LAYERS': 3,
        'SCALAR_BLOCKS': 3,
        'FUSION_BLOCKS': 3,
        'FUSION_HEADS': 8,
        'KSPACE_GNN_TYPE': 'transformer',
        
        # Training hyperparameters
        'LEARNING_RATE': 1e-3,
        'WEIGHT_DECAY': 1e-4,
        'FOCAL_ALPHA': 1.0,
        'FOCAL_GAMMA': 2.0,
        'SCHEDULER_T0': 15,
        'SCHEDULER_T_MULT': 2,
        'SCHEDULER_ETA_MIN': 1e-6,
        
        # Data augmentation parameters
        'MIXUP_ALPHA': 0.2,
        'CUTMIX_PROB': 0.5,
        'FEATURE_MASK_PROB': 0.1,
        'EDGE_DROPOUT': 0.1,
        'NODE_FEATURE_NOISE': 0.05,
        
        # Data loading parameters
        'BATCH_SIZE': 32,
        'NUM_WORKERS': 4,
        'MAX_CRYSTAL_NODES': 1000,
        'MAX_KSPACE_NODES': 500,
        
        # Other settings
        'USE_WANDB': False,
        'SEED': 42
    }
    
    # Set random seed
    torch.manual_seed(config_dict['SEED'])
    np.random.seed(config_dict['SEED'])
    
    # Create model
    model = create_crazy_fusion_model(config_dict)
    print(f"Created Crazy Fusion Model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer with appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = CrazyTrainer(model, config_dict, device=device)
    
    # Check if real data is available
    from helper import config
    data_dir = config.DATA_DIR / "processed"
    
    if REAL_DATA_AVAILABLE and data_dir.exists():
        print("🎉 Using real data loaders with full augmentations!")
        
        # Create real data loaders with augmentations
        modalities = ['crystal', 'kspace', 'scalar', 'decomposition', 'spectral']
        
        loaders = create_data_loaders(
            data_dir=data_dir,
            batch_size=config_dict['BATCH_SIZE'],
            modalities=modalities,
            num_workers=config_dict['NUM_WORKERS'],
            augment=True,  # Enable all augmentations
            mixup_alpha=config_dict['MIXUP_ALPHA'],
            cutmix_prob=config_dict['CUTMIX_PROB'],
            feature_mask_prob=config_dict['FEATURE_MASK_PROB'],
            edge_dropout=config_dict['EDGE_DROPOUT'],
            node_feature_noise=config_dict['NODE_FEATURE_NOISE']
        )
        
        train_loader = loaders['train']
        val_loader = loaders['val']
        test_loader = loaders['test']
        
        # Get class weights for imbalanced dataset
        class_weights = get_class_weights(data_dir)
        print(f"Class weights: {class_weights}")
        
        # Update trainer with class weights
        trainer.criterion = FocalLoss(
            alpha=config_dict['FOCAL_ALPHA'],
            gamma=config_dict['FOCAL_GAMMA']
        )
        
    else:
        print("⚠️  Real data not found, falling back to dummy data loaders")
        print(f"Expected data directory: {data_dir}")
        
        # Create dummy data loaders as fallback
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=1000):
                self.size = size
                self.data = []
                for i in range(size):
                    sample = {
                        'crystal_x': torch.randn(50, 92),
                        'crystal_edge_index': torch.randint(0, 50, (2, 100)).long(),
                        'kspace_x': torch.randn(30, 2),
                        'kspace_edge_index': torch.randint(0, 30, (2, 60)).long(),
                        'scalar_features': torch.randn(200),
                        'decomposition_features': torch.randn(100)
                    }
                    self.data.append((sample, torch.randint(0, config_dict['NUM_CLASSES'], (1,)).item()))
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Create datasets
        train_dataset = DummyDataset(800)
        val_dataset = DummyDataset(100)
        test_dataset = DummyDataset(100)
        
        def custom_collate(batch):
            """Custom collate function to handle edge indices properly."""
            data, label = batch[0]
            return data, torch.tensor([label])
        
        # Create data loaders with custom collate
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=50, save_path='best_crazy_model.pth')
    
    # Plot training curves
    trainer.plot_training_curves('crazy_training_curves.png')
    
    print("Training completed successfully!") 