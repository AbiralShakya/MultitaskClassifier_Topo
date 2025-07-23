#!/usr/bin/env python3
"""
Safe ultimate training - builds on proven 86.64% baseline with conservative improvements
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

class SafeUltimateClassifier(nn.Module):
    """
    Safe classifier that builds conservatively on the proven 86.64% baseline
    """
    
    def __init__(self):
        super().__init__()
        
        # Slightly enhanced encoders (conservative improvement)
        self.crystal_encoder = TopologicalCrystalEncoder(
            node_feature_dim=3,
            hidden_dim=320,  # Modest increase from 256
            num_layers=5,    # One more layer
            output_dim=320,
            radius=5.0,
            num_scales=3,
            use_topological_features=True
        )
        
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=config.KSPACE_NODE_FEATURE_DIM,
            hidden_dim=320,  # Modest increase
            out_channels=320,
            n_layers=5,      # One more layer
            num_heads=10     # Slightly more heads
        )
        
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=config.SCALAR_TOTAL_DIM,
            hidden_dims=[768, 512, 320],  # Slightly deeper
            out_channels=320
        )
        
        self.physics_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=config.DECOMPOSITION_FEATURE_DIM,
            gap_features_dim=getattr(config, 'BAND_GAP_SCALAR_DIM', 1),
            dos_features_dim=getattr(config, 'DOS_FEATURE_DIM', 500),
            fermi_features_dim=getattr(config, 'FERMI_FEATURE_DIM', 1),
            output_dim=320
        )
        
        # Enhanced fusion (conservative)
        total_dim = 320 * 4  # 1280
        
        self.attention = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=10,  # More heads but not extreme
            dropout=0.25,
            batch_first=True
        )
        
        # Enhanced classifier (conservative)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_dim, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(96, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, inputs):
        """Safe forward pass"""
        
        # Encode each modality
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
        
        # Single attention layer (proven to work)
        features_unsqueezed = features.unsqueeze(1)
        attended_features, _ = self.attention(
            features_unsqueezed, features_unsqueezed, features_unsqueezed
        )
        attended_features = attended_features.squeeze(1)
        
        # Residual connection
        features = features + attended_features
        
        # Classification
        logits = self.classifier(features)
        
        return {'logits': logits}
    
    def compute_loss(self, predictions, targets):
        """SAFE loss function - proven focal loss only"""
        
        # Simple focal loss (exactly what worked before)
        ce_loss = F.cross_entropy(predictions['logits'], targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** 2 * ce_loss
        
        return focal_loss.mean()


class CosineWarmupScheduler:
    """Safe scheduler"""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def train_epoch(model, loader, optimizer, device):
    """Safe training epoch"""
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
        loss = model.compute_loss(outputs, batch['topology_label'])
        
        # Sanity check - loss should always be positive
        if loss.item() < 0:
            print(f"WARNING: Negative loss detected: {loss.item()}")
            continue
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    """Safe validation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
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
            
            outputs = model(batch)
            loss = model.compute_loss(outputs, batch['topology_label'])
            
            total_loss += loss.item()
            preds = outputs['logits'].argmax(dim=1)
            correct += (preds == batch['topology_label']).sum().item()
            total += len(batch['topology_label'])
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch['topology_label'].cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, all_preds, all_targets


def main():
    """Safe ultimate training"""
    print("🛡️ Safe Ultimate Training (Conservative Improvements)")
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
    
    # Conservative split (more training data)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=0.18,  # Slightly more training data
        stratify=all_labels,
        random_state=42
    )
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.18,
        stratify=all_labels[train_val_idx],
        random_state=42
    )
    
    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create data loaders
    train_loader = PyGDataLoader(
        Subset(dataset, train_idx),
        batch_size=28,  # Slightly larger batch
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    val_loader = PyGDataLoader(
        Subset(dataset, val_idx),
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    test_loader = PyGDataLoader(
        Subset(dataset, test_idx),
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SafeUltimateClassifier().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Conservative optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=1.8e-4,  # Slightly higher than working baseline
        weight_decay=1.5e-4,  # Moderate regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Safe scheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=6,
        max_epochs=60,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0
    patience = 12
    patience_counter = 0
    
    for epoch in range(60):
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_targets = validate_epoch(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/60:")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_safe_ultimate_model.pt')
            print(f"  ✅ New best validation accuracy: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("-" * 60)
    
    # Final evaluation
    model.load_state_dict(torch.load('best_safe_ultimate_model.pt'))
    test_loss, test_acc, test_f1, test_preds, test_targets = validate_epoch(model, test_loader, device)
    
    print("\n" + "="*70)
    print("SAFE ULTIMATE RESULTS")
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
        print(f"🏆 Final score: {test_acc:.4f}")
    else:
        print(f"📈 Progress from 86.64%: +{(test_acc-0.8664)*100:.1f}%")
        print(f"📈 Still need: {(0.92 - test_acc)*100:.1f}% more accuracy.")
    
    return test_acc, test_f1


if __name__ == "__main__":
    try:
        test_acc, test_f1 = main()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()