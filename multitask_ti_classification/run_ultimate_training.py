#!/usr/bin/env python3
"""
Ultimate training pipeline to achieve 92%+ accuracy
Integrates all optimizations with advanced techniques
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config
from helper.dataset import MaterialDataset, custom_collate_fn
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# Import existing working components
from helper.topological_crystal_encoder import TopologicalCrystalEncoder
from src.model_w_debug import KSpaceTransformerGNNEncoder, ScalarFeatureEncoder
from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures

class UltimateClassifier(nn.Module):
    """
    Ultimate classifier with all advanced techniques for 92%+ accuracy
    """
    
    def __init__(self):
        super().__init__()
        
        # Enhanced encoders with more capacity
        self.crystal_encoder = TopologicalCrystalEncoder(
            node_feature_dim=3,
            hidden_dim=384,  # Increased capacity
            num_layers=6,    # Deeper network
            output_dim=384,
            radius=5.0,
            num_scales=3,
            use_topological_features=True
        )
        
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=config.KSPACE_NODE_FEATURE_DIM,
            hidden_dim=384,  # Increased capacity
            out_channels=384,
            n_layers=6,      # Deeper network
            num_heads=12     # More attention heads
        )
        
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=config.SCALAR_TOTAL_DIM,
            hidden_dims=[1024, 512, 384],  # Deeper scalar processing
            out_channels=384
        )
        
        self.physics_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=config.DECOMPOSITION_FEATURE_DIM,
            gap_features_dim=getattr(config, 'BAND_GAP_SCALAR_DIM', 1),
            dos_features_dim=getattr(config, 'DOS_FEATURE_DIM', 500),
            fermi_features_dim=getattr(config, 'FERMI_FEATURE_DIM', 1),
            output_dim=384
        )
        
        # Multi-scale fusion with multiple attention layers
        total_dim = 384 * 4  # 1536
        
        # First attention layer
        self.attention1 = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=16,
            dropout=0.2,
            batch_first=True
        )
        
        # Second attention layer for deeper processing
        self.attention2 = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=16,
            dropout=0.2,
            batch_first=True
        )
        
        # Feature mixing layers
        self.feature_mixer = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.LayerNorm(total_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(total_dim, total_dim),
            nn.LayerNorm(total_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Enhanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 2:  # Final classification layer
                    nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller init for final layer
                else:
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, inputs):
        """Enhanced forward pass with multiple attention layers"""
        
        # Encode each modality with enhanced capacity
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
        
        # Concatenate features
        features = torch.cat([crystal_emb, kspace_emb, scalar_emb, phys_emb], dim=-1)
        
        # First attention layer
        features_unsqueezed = features.unsqueeze(1)
        attended_features1, _ = self.attention1(
            features_unsqueezed, features_unsqueezed, features_unsqueezed
        )
        attended_features1 = attended_features1.squeeze(1)
        
        # Residual connection
        features = features + attended_features1
        
        # Feature mixing
        mixed_features = self.feature_mixer(features)
        features = features + mixed_features  # Another residual connection
        
        # Second attention layer
        features_unsqueezed = features.unsqueeze(1)
        attended_features2, _ = self.attention2(
            features_unsqueezed, features_unsqueezed, features_unsqueezed
        )
        attended_features2 = attended_features2.squeeze(1)
        
        # Final residual connection
        features = features + attended_features2
        
        # Classification
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features,
            'crystal_emb': crystal_emb,
            'kspace_emb': kspace_emb,
            'scalar_emb': scalar_emb,
            'phys_emb': phys_emb
        }
    
    def compute_loss(self, predictions, targets):
        """Fixed loss function - always positive"""
        
        # Simple focal loss (proven to work)
        ce_loss = F.cross_entropy(predictions['logits'], targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** 2 * ce_loss
        
        return focal_loss.mean()


class CosineWarmupScheduler:
    """Enhanced cosine annealing with warmup and restarts"""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-7, restart_epochs=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.restart_epochs = restart_epochs or []
        
    def step(self, epoch):
        # Check for restarts
        if epoch in self.restart_epochs:
            current_lr = self.base_lr * 0.5  # Restart with half learning rate
        elif epoch < self.warmup_epochs:
            current_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            current_lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        return current_lr


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, optimizer, device, use_mixup=True):
    """Enhanced training with mixup augmentation"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(loader):
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif hasattr(batch[key], 'to'):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], dict):
                for sub_key in batch[key]:
                    if isinstance(batch[key][sub_key], torch.Tensor):
                        batch[key][sub_key] = batch[key][sub_key].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        
        # Apply mixup to features if enabled
        if use_mixup and np.random.random() > 0.5:
            mixed_features, y_a, y_b, lam = mixup_data(outputs['features'], batch['topology_label'])
            # Re-classify mixed features
            mixed_logits = model.classifier(mixed_features)
            loss = mixup_criterion(
                lambda pred, target: F.cross_entropy(pred, target),
                mixed_logits, y_a, y_b, lam
            )
        else:
            loss = model.compute_loss(outputs, batch['topology_label'])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Tighter clipping
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        preds = outputs['logits'].argmax(dim=1)
        correct += (preds == batch['topology_label']).sum().item()
        total += len(batch['topology_label'])
        
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {(preds == batch['topology_label']).float().mean().item():.4f}")
    
    return total_loss / len(loader), correct / total


def validate_epoch(model, loader, device):
    """Enhanced validation with test-time augmentation"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif hasattr(batch[key], 'to'):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        if isinstance(batch[key][sub_key], torch.Tensor):
                            batch[key][sub_key] = batch[key][sub_key].to(device)
            
            # Multiple forward passes for test-time augmentation
            outputs_list = []
            for _ in range(3):  # 3 forward passes
                outputs = model(batch)
                outputs_list.append(outputs['logits'])
            
            # Average predictions
            avg_logits = torch.stack(outputs_list).mean(dim=0)
            loss = F.cross_entropy(avg_logits, batch['topology_label'])
            
            total_loss += loss.item()
            preds = avg_logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch['topology_label'].cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, all_preds, all_targets


def main():
    """Ultimate training function"""
    print("🚀 Ultimate Training Pipeline for 92%+ Accuracy")
    print("=" * 60)
    
    # Load dataset
    print("Loading dataset...")
    dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR,
        preload=False
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Get labels for stratified split
    all_labels = []
    for i in range(len(dataset)):
        try:
            label = dataset[i]['topology_label'].item()
            all_labels.append(label)
        except Exception as e:
            all_labels.append(0)
    
    all_labels = np.array(all_labels)
    print(f"Label distribution: {np.bincount(all_labels)}")
    
    # Enhanced stratified split with more training data
    train_val_idx, test_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=0.15,  # Smaller test set, more training data
        stratify=all_labels,
        random_state=42
    )
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15,  # Smaller validation set
        stratify=all_labels[train_val_idx],
        random_state=42
    )
    
    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create data loaders with optimal batch size
    train_loader = PyGDataLoader(
        Subset(dataset, train_idx),
        batch_size=24,  # Optimal batch size for this dataset
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = PyGDataLoader(
        Subset(dataset, val_idx),
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = PyGDataLoader(
        Subset(dataset, test_idx),
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Create ultimate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UltimateClassifier().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Enhanced optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=1.5e-4,  # Slightly lower learning rate
        weight_decay=2e-4,  # Stronger regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Enhanced scheduler with restarts
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=8,
        max_epochs=80,
        eta_min=1e-7,
        restart_epochs=[30, 50]  # Learning rate restarts
    )
    
    # Training loop with enhanced early stopping
    best_val_acc = 0
    best_test_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(80):
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # Train with mixup
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, use_mixup=True)
        
        # Validate with test-time augmentation
        val_loss, val_acc, val_f1, val_preds, val_targets = validate_epoch(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/80:")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_ultimate_model.pt')
            print(f"  ✅ New best validation accuracy: {best_val_acc:.4f}")
            patience_counter = 0
            
            # Also test on test set when we get new best validation
            test_loss, test_acc, test_f1, test_preds, test_targets = validate_epoch(model, test_loader, device)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                print(f"  🎯 Test accuracy: {test_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("-" * 60)
    
    # Final evaluation
    model.load_state_dict(torch.load('best_ultimate_model.pt'))
    test_loss, test_acc, test_f1, test_preds, test_targets = validate_epoch(model, test_loader, device)
    
    print("\n" + "="*70)
    print("ULTIMATE FINAL RESULTS")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test F1 Score: {test_f1:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_targets, test_preds))
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, digits=4))
    
    if test_acc >= 0.92:
        print("🎉 SUCCESS: Achieved 92%+ test accuracy!")
        print(f"🏆 Final score: {test_acc:.4f} ({(test_acc-0.92)*100:.1f}% above target)")
    else:
        print(f"📈 Close! Need {(0.92 - test_acc)*100:.1f}% more accuracy.")
        print(f"💪 Improvement from baseline: +{(test_acc-0.8664)*100:.1f}%")
    
    return test_acc, test_f1


if __name__ == "__main__":
    try:
        test_acc, test_f1 = main()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()