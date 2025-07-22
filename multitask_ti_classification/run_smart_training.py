#!/usr/bin/env python3
"""
Smart training that uses the existing working model with optimized training techniques
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
from encoders.asph_encoder import ASPHEncoder

class SmartOptimizedClassifier(nn.Module):
    """
    Smart classifier that uses existing working encoders with optimized training
    """
    
    def __init__(self):
        super().__init__()
        
        # Use existing working encoders
        self.crystal_encoder = TopologicalCrystalEncoder(
            node_feature_dim=3,
            hidden_dim=256,
            num_layers=4,
            output_dim=256,
            radius=5.0,
            num_scales=3,
            use_topological_features=True
        )
        
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=config.KSPACE_NODE_FEATURE_DIM,
            hidden_dim=256,
            out_channels=256,
            n_layers=4,
            num_heads=8
        )
        
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=config.SCALAR_TOTAL_DIM,
            hidden_dims=[512, 256],
            out_channels=256
        )
        
        self.physics_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=config.DECOMPOSITION_FEATURE_DIM,
            gap_features_dim=getattr(config, 'BAND_GAP_SCALAR_DIM', 1),
            dos_features_dim=getattr(config, 'DOS_FEATURE_DIM', 500),
            fermi_features_dim=getattr(config, 'FERMI_FEATURE_DIM', 1),
            output_dim=256
        )
        
        # Skip ASPH encoder since dataset doesn't load asph_features
        # self.asph_encoder = ASPHEncoder(
        #     input_dim=3115,
        #     hidden_dims=256,
        #     out_dim=128
        # )
        
        # Optimized fusion with self-attention (without ASPH)
        total_dim = 256 + 256 + 256 + 256  # 1024 (crystal + kspace + scalar + physics)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )
        
        # Optimized classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, inputs):
        """Forward pass"""
        
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
        
        # Skip ASPH since dataset doesn't provide asph_features
        # asph_emb = self.asph_encoder(inputs['asph_features'])
        
        # Concatenate features (without ASPH)
        features = torch.cat([crystal_emb, kspace_emb, scalar_emb, phys_emb], dim=-1)
        
        # Self-attention
        features_unsqueezed = features.unsqueeze(1)
        attended_features, _ = self.attention(
            features_unsqueezed, features_unsqueezed, features_unsqueezed
        )
        features = attended_features.squeeze(1)
        
        # Add residual connection (without ASPH)
        original_features = torch.cat([crystal_emb, kspace_emb, scalar_emb, phys_emb], dim=-1)
        features = features + original_features
        
        # Classification
        logits = self.classifier(features)
        
        return {'logits': logits}
    
    def compute_loss(self, predictions, targets):
        """Enhanced loss with focal loss"""
        
        # Focal loss
        ce_loss = F.cross_entropy(predictions['logits'], targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** 2 * ce_loss
        
        return focal_loss.mean()


class CosineWarmupScheduler:
    """Cosine annealing with warmup"""
    
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
    """Train for one epoch"""
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
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        preds = outputs['logits'].argmax(dim=1)
        correct += (preds == batch['topology_label']).sum().item()
        total += len(batch['topology_label'])
        
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {(preds == batch['topology_label']).float().mean().item():.4f}")
    
    return total_loss / len(loader), correct / total


def validate_epoch(model, loader, device):
    """Validate for one epoch"""
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
    """Main training function"""
    print("🚀 Smart Optimized Training")
    print("=" * 50)
    
    # Load dataset
    print("Loading dataset...")
    dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR,
        preload=False  # Don't preload to avoid memory issues
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
    
    # Stratified split
    train_val_idx, test_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        stratify=all_labels[train_val_idx],
        random_state=42
    )
    
    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create data loaders
    train_loader = PyGDataLoader(
        Subset(dataset, train_idx),
        batch_size=32,  # Smaller batch size for stability
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=2
    )
    
    val_loader = PyGDataLoader(
        Subset(dataset, val_idx),
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=2
    )
    
    test_loader = PyGDataLoader(
        Subset(dataset, test_idx),
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=2
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SmartOptimizedClassifier().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=50,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_targets = validate_epoch(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/50:")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_smart_model.pt')
            print(f"  ✅ New best validation accuracy: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("-" * 50)
    
    # Final evaluation
    model.load_state_dict(torch.load('best_smart_model.pt'))
    test_loss, test_acc, test_f1, test_preds, test_targets = validate_epoch(model, test_loader, device)
    
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
    
    if test_acc >= 0.92:
        print("🎉 SUCCESS: Achieved 92%+ test accuracy!")
    else:
        print(f"📈 Progress made. Need {(0.92 - test_acc)*100:.1f}% more accuracy.")
    
    return test_acc, test_f1


if __name__ == "__main__":
    try:
        test_acc, test_f1 = main()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()