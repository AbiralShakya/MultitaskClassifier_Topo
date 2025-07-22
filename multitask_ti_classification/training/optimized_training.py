"""
Optimized training pipeline for 92%+ accuracy
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import helper.config as config
from helper.dataset import MaterialDataset
from models.optimized_classifier import OptimizedMaterialClassifier, EarlyStopping, CosineWarmupScheduler


class OptimizedTrainer:
    """Optimized trainer with advanced techniques"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with weight decay
        self.optimizer = AdamW(
            model.parameters(),
            lr=2e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=5,
            max_epochs=50,
            eta_min=1e-6
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=10,
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.model.compute_loss(outputs, batch['topology_label'])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            preds = outputs['logits'].argmax(dim=1)
            correct += (preds == batch['topology_label']).sum().item()
            total += len(batch['topology_label'])
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {(preds == batch['topology_label']).float().mean().item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_to_device(batch)
                
                outputs = self.model(batch)
                loss = self.model.compute_loss(outputs, batch['topology_label'])
                
                total_loss += loss.item()
                preds = outputs['logits'].argmax(dim=1)
                correct += (preds == batch['topology_label']).sum().item()
                total += len(batch['topology_label'])
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch['topology_label'].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, accuracy, f1, all_preds, all_targets
    
    def train(self, train_loader, val_loader, num_epochs=50):
        """Full training loop"""
        print("Starting optimized training...")
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            # Update learning rate
            current_lr = self.scheduler.step(epoch)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_f1, val_preds, val_targets = self.validate_epoch(val_loader)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  LR: {current_lr:.2e}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_optimized_model.pt')
                print(f"  ✅ New best validation accuracy: {best_val_acc:.4f}")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            print("-" * 60)
        
        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        return best_val_acc
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        # Load best model
        self.model.load_state_dict(torch.load('best_optimized_model.pt'))
        
        test_loss, test_acc, test_f1, test_preds, test_targets = self.validate_epoch(test_loader)
        
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(test_targets, test_preds))
        print("\nClassification Report:")
        print(classification_report(test_targets, test_preds, digits=4))
        
        return test_acc, test_f1
    
    def _move_to_device(self, batch):
        """Move batch to device"""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
            elif hasattr(batch[key], 'to'):
                batch[key] = batch[key].to(self.device)
            elif isinstance(batch[key], dict):
                for sub_key in batch[key]:
                    if isinstance(batch[key][sub_key], torch.Tensor):
                        batch[key][sub_key] = batch[key][sub_key].to(self.device)
        return batch


def run_optimized_training():
    """Run the optimized training pipeline"""
    
    print("🚀 Starting Optimized Training Pipeline")
    print("="*70)
    
    # Load dataset
    print("Loading dataset...")
    dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR,
        preload=getattr(config, 'PRELOAD_DATASET', True)
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Get labels for stratified split
    all_labels = []
    for i in range(len(dataset)):
        try:
            label = dataset[i]['topology_label'].item()
            all_labels.append(label)
        except Exception as e:
            print(f"Error getting label for index {i}: {e}")
            all_labels.append(0)
    
    all_labels = np.array(all_labels)
    print(f"Label distribution: {np.bincount(all_labels)}")
    
    # Stratified train/val/test split
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test (80/20)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )
    
    # Second split: train vs val (80/20 of remaining)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        stratify=all_labels[train_val_idx],
        random_state=42
    )
    
    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create data loaders
    train_loader = PyGDataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = PyGDataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = PyGDataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = OptimizedMaterialClassifier(
        crystal_node_feature_dim=3,
        kspace_node_feature_dim=config.KSPACE_NODE_FEATURE_DIM,
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
        num_topology_classes=2,
        hidden_dim=256,
        dropout_rate=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and train
    trainer = OptimizedTrainer(model, device)
    best_val_acc = trainer.train(train_loader, val_loader, num_epochs=50)
    
    # Final evaluation
    test_acc, test_f1 = trainer.evaluate(test_loader)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    if test_acc >= 0.92:
        print("🎉 SUCCESS: Achieved 92%+ test accuracy!")
    else:
        print(f"❌ Target not reached. Need {0.92 - test_acc:.4f} more accuracy.")
    
    return test_acc, test_f1


if __name__ == "__main__":
    run_optimized_training()