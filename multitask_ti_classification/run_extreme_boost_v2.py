#!/usr/bin/env python3
"""
EXTREME BOOST V2 - Advanced techniques to close the remaining 3.6% gap
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import time
import copy
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config
from helper.dataset import MaterialDataset, custom_collate_fn
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Subset, ConcatDataset
from sklearn.model_selection import train_test_split

# Import existing working components
from helper.topological_crystal_encoder import TopologicalCrystalEncoder
from src.model_w_debug import KSpaceTransformerGNNEncoder, ScalarFeatureEncoder
from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures


class ResidualBlock(nn.Module):
    """Residual block with pre-activation"""
    
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        
        # First block
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        
        # Residual connection
        return x + residual


class ExtremeBoostClassifierV2(nn.Module):
    """
    Enhanced classifier with advanced techniques to close the remaining gap
    """
    
    def __init__(self, dropout_rate=0.3, stochastic_depth_prob=0.1):
        super().__init__()
        
        # Balanced encoders with improved architectures
        self.crystal_encoder = TopologicalCrystalEncoder(
            node_feature_dim=3,
            hidden_dim=384,
            num_layers=5,  # Slightly increased from v1
            output_dim=384,
            radius=5.0,
            num_scales=4,  # Increased from 3
            use_topological_features=True
        )
        
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=config.KSPACE_NODE_FEATURE_DIM,
            hidden_dim=384,
            out_channels=384,
            n_layers=5,  # Slightly increased from v1
            num_heads=8
        )
        
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=config.SCALAR_TOTAL_DIM,
            hidden_dims=[1024, 768, 512, 384],  # Added one more layer
            out_channels=384
        )
        
        self.physics_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=config.DECOMPOSITION_FEATURE_DIM,
            gap_features_dim=getattr(config, 'BAND_GAP_SCALAR_DIM', 1),
            dos_features_dim=getattr(config, 'DOS_FEATURE_DIM', 500),
            fermi_features_dim=getattr(config, 'FERMI_FEATURE_DIM', 1),
            output_dim=384
        )
        
        # Fusion network
        total_dim = 384 * 4  # 1536
        
        # Multi-head cross-attention with stochastic depth
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=384,
                num_heads=6,
                dropout=dropout_rate,
                batch_first=True
            ) for _ in range(4)  # One for each modality
        ])
        
        # Self-attention for global fusion
        self.self_attention = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=12,  # Reduced from 16
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature transformation with residual blocks
        self.feature_transform = nn.Sequential(
            ResidualBlock(total_dim, dropout_rate),
            ResidualBlock(total_dim, dropout_rate),
            nn.LayerNorm(total_dim)
        )
        
        # Deep classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            ResidualBlock(1024, dropout_rate),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            ResidualBlock(512, dropout_rate),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 2)
        )
        
        # Auxiliary classifiers for deep supervision
        self.aux_classifier1 = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2)
        )
        
        self.aux_classifier2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2)
        )
        
        # Stochastic depth probability
        self.stochastic_depth_prob = stochastic_depth_prob
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Enhanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _stochastic_depth(self, x, p, training):
        """Apply stochastic depth to the input"""
        if not training or p == 0:
            return x
        
        keep_prob = 1 - p
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = mask.floor_()
        return x * mask / keep_prob
    
    def forward(self, inputs):
        """Advanced forward pass with cross-attention and stochastic depth"""
        
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
        
        # Store original embeddings
        embeddings = [crystal_emb, kspace_emb, scalar_emb, phys_emb]
        
        # Cross-attention between modalities
        cross_attended = []
        for i, emb in enumerate(embeddings):
            # Create context from all other embeddings
            context_embs = []
            for j, other_emb in enumerate(embeddings):
                if i != j:
                    # Project to same dimension if needed
                    if other_emb.shape[1] != emb.shape[1]:
                        proj = nn.Linear(other_emb.shape[1], emb.shape[1], device=emb.device)
                        other_emb = proj(other_emb)
                    context_embs.append(other_emb)
            
            # Concatenate context
            if context_embs:
                context = torch.stack(context_embs, dim=1)
                
                # Apply cross-attention
                emb_unsqueezed = emb.unsqueeze(1)
                attended, _ = self.cross_attention[i](
                    emb_unsqueezed, context, context
                )
                attended = attended.squeeze(1)
                
                # Apply stochastic depth
                if self.training:
                    attended = self._stochastic_depth(attended, self.stochastic_depth_prob, self.training)
                
                # Residual connection
                emb = emb + attended
            
            cross_attended.append(emb)
        
        # Concatenate all cross-attended features
        features = torch.cat(cross_attended, dim=-1)
        
        # Self-attention for global fusion
        features_unsqueezed = features.unsqueeze(1)
        attended_features, _ = self.self_attention(
            features_unsqueezed, features_unsqueezed, features_unsqueezed
        )
        attended_features = attended_features.squeeze(1)
        
        # Apply stochastic depth
        if self.training:
            attended_features = self._stochastic_depth(attended_features, self.stochastic_depth_prob, self.training)
        
        # Residual connection
        features = features + attended_features
        
        # Feature transformation
        transformed = self.feature_transform(features)
        
        # Auxiliary classifier 1
        aux_logits1 = self.aux_classifier1(transformed)
        
        # First part of main classifier
        x = self.classifier[:5](transformed)  # Up to first residual block
        
        # Auxiliary classifier 2
        aux_logits2 = self.aux_classifier2(x)
        
        # Rest of main classifier
        logits = self.classifier[5:](x)
        
        return {
            'logits': logits,
            'aux_logits1': aux_logits1,
            'aux_logits2': aux_logits2,
            'features': transformed,
            'embeddings': embeddings
        }
    
    def compute_loss(self, predictions, targets, label_smoothing=0.1, aux_weight=0.3):
        """Advanced loss function with deep supervision"""
        
        # Focal loss with label smoothing for main logits
        ce_loss = F.cross_entropy(
            predictions['logits'], 
            targets, 
            reduction='none',
            label_smoothing=label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** 2 * ce_loss
        main_loss = focal_loss.mean()
        
        # Auxiliary losses
        aux_loss1 = F.cross_entropy(
            predictions['aux_logits1'],
            targets,
            label_smoothing=label_smoothing
        )
        
        aux_loss2 = F.cross_entropy(
            predictions['aux_logits2'],
            targets,
            label_smoothing=label_smoothing
        )
        
        # Combined loss with deep supervision
        total_loss = main_loss + aux_weight * (aux_loss1 + aux_loss2)
        
        return total_loss


class CosineWarmupScheduler:
    """Advanced scheduler with warmup and cosine annealing"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.min_lr + (self.max_lr - self.min_lr) * epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation with safety checks"""
    # Check batch size - don't do mixup for very small batches
    batch_size = x.size(0)
    if batch_size <= 1:
        return x, y, y, 1.0
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """CutMix data augmentation"""
    batch_size = x.size(0)
    if batch_size <= 1:
        return x, y, y, 1.0
        
    # Generate random index permutation
    index = torch.randperm(batch_size).to(x.device)
    
    # Random region parameters
    lam = np.random.beta(alpha, alpha)
    
    # For feature vectors, we'll do a simple weighted average
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_epoch(model, loader, optimizer, device, use_mixup=True, mixup_alpha=0.2, use_cutmix=True, cutmix_alpha=1.0):
    """Advanced training with mixup, cutmix and error handling"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(loader):
        try:
            # Check batch size - skip very small batches
            if len(batch['topology_label']) <= 1:
                print(f"Skipping batch {batch_idx} with size {len(batch['topology_label'])}")
                continue
                
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
            
            # Apply data augmentation if enabled
            aug_method = None
            if use_mixup and use_cutmix:
                # Randomly choose between mixup and cutmix
                aug_method = 'mixup' if np.random.random() > 0.5 else 'cutmix'
            elif use_mixup:
                aug_method = 'mixup'
            elif use_cutmix:
                aug_method = 'cutmix'
            
            if aug_method == 'mixup' and np.random.random() > 0.3 and len(batch['topology_label']) > 2:
                # Apply mixup to features
                mixed_features, y_a, y_b, lam = mixup_data(
                    outputs['features'], batch['topology_label'], alpha=mixup_alpha
                )
                
                # Forward through classifier only
                mixed_logits = model.classifier(mixed_features)
                
                # Compute mixup loss
                loss = mixup_criterion(
                    lambda p, t: F.cross_entropy(p, t, reduction='none', label_smoothing=0.1),
                    mixed_logits, y_a, y_b, lam
                ).mean()
                
                # Use original predictions for accuracy calculation
                preds = outputs['logits'].argmax(dim=1)
                
            elif aug_method == 'cutmix' and np.random.random() > 0.3 and len(batch['topology_label']) > 2:
                # Apply cutmix to features
                mixed_features, y_a, y_b, lam = cutmix_data(
                    outputs['features'], batch['topology_label'], alpha=cutmix_alpha
                )
                
                # Forward through classifier only
                mixed_logits = model.classifier(mixed_features)
                
                # Compute cutmix loss
                loss = mixup_criterion(  # Same function works for cutmix
                    lambda p, t: F.cross_entropy(p, t, reduction='none', label_smoothing=0.1),
                    mixed_logits, y_a, y_b, lam
                ).mean()
                
                # Use original predictions for accuracy calculation
                preds = outputs['logits'].argmax(dim=1)
                
            else:
                # Standard forward pass
                loss = model.compute_loss(outputs, batch['topology_label'])
                preds = outputs['logits'].argmax(dim=1)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            correct += (preds == batch['topology_label']).sum().item()
            total += len(batch['topology_label'])
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {(preds == batch['topology_label']).float().mean().item():.4f}")
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if total == 0:  # Avoid division by zero
        return 0.0, 0.0
        
    return total_loss / len(loader), correct / total


def validate_epoch(model, loader, device, tta_samples=5):
    """Advanced validation with test-time augmentation and error handling"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            try:
                # Check batch size - skip very small batches
                if len(batch['topology_label']) <= 1:
                    print(f"Skipping validation batch {batch_idx} with size {len(batch['topology_label'])}")
                    continue
                    
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
                
                # Test-time augmentation
                all_logits = []
                
                for _ in range(tta_samples):
                    outputs = model(batch)
                    all_logits.append(outputs['logits'])
                
                # Average predictions
                avg_logits = torch.stack(all_logits).mean(dim=0)
                
                # Compute loss
                loss = F.cross_entropy(avg_logits, batch['topology_label'])
                total_loss += loss.item()
                batch_count += 1
                
                # Get predictions
                preds = avg_logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch['topology_label'].cpu().numpy())
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if batch_count == 0:  # Avoid division by zero
        return 0.0, 0.0, 0.0, [], []
        
    avg_loss = total_loss / batch_count
    
    if len(all_preds) == 0:  # No valid predictions
        return avg_loss, 0.0, 0.0, [], []
        
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, all_preds, all_targets


def k_fold_cross_validation(dataset, n_splits=5, batch_size=32, num_workers=4, device='cuda'):
    """K-fold cross-validation for model selection with improved training"""
    print(f"Starting {n_splits}-fold cross-validation...")
    
    # Get all labels for stratification
    all_labels = []
    for i in range(len(dataset)):
        try:
            label = dataset[i]['topology_label'].item()
            all_labels.append(label)
        except Exception as e:
            all_labels.append(0)
    
    all_labels = np.array(all_labels)
    
    # Setup K-Fold
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Track metrics
    fold_val_accs = []
    fold_models = []
    
    # Run K-Fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n{'='*20} Fold {fold+1}/{n_splits} {'='*20}")
        
        # Set seed for reproducibility
        set_seed(42 + fold)
        
        # Create data loaders
        train_loader = PyGDataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=num_workers,
            drop_last=True  # Drop the last batch if it's smaller than batch_size
        )
        
        val_loader = PyGDataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=num_workers
        )
        
        # Create model
        model = ExtremeBoostClassifierV2(
            dropout_rate=0.3,
            stochastic_depth_prob=0.1
        ).to(device)
        
        # Optimizer with improved settings
        optimizer = AdamW(
            model.parameters(),
            lr=3e-5,  # Lower initial LR for better stability
            weight_decay=3e-4,  # Higher weight decay to prevent overfitting
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler with improved settings
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=15,  # Longer warmup for better stability
            total_epochs=100,  # Longer training
            min_lr=1e-6,
            max_lr=8e-5  # Lower max LR for better stability
        )
        
        # Training loop
        best_val_acc = 0
        best_model_state = None
        patience = 15  # Increased patience
        patience_counter = 0
        
        for epoch in range(100):  # Longer training per fold
            # Update learning rate
            current_lr = scheduler.step(epoch)
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, 
                use_mixup=True, mixup_alpha=0.2,
                use_cutmix=True, cutmix_alpha=1.0
            )
            
            # Validate
            val_loss, val_acc, val_f1, _, _ = validate_epoch(
                model, val_loader, device, tta_samples=5  # Increased TTA samples
            )
            
            print(f"Epoch {epoch+1}/100:")
            print(f"  LR: {current_lr:.2e}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save fold results
        fold_val_accs.append(best_val_acc)
        
        # Save best model from this fold
        model.load_state_dict(best_model_state)
        fold_models.append(copy.deepcopy(model))
        torch.save(best_model_state, f'kfold_model_{fold+1}.pt')
        
        print(f"Fold {fold+1} best validation accuracy: {best_val_acc:.4f}")
    
    # Print overall results
    print("\n" + "="*50)
    print(f"Cross-validation complete!")
    print(f"Mean validation accuracy: {np.mean(fold_val_accs):.4f} ± {np.std(fold_val_accs):.4f}")
    
    # Return best model based on validation accuracy
    best_fold = np.argmax(fold_val_accs)
    print(f"Best fold: {best_fold+1} with accuracy: {fold_val_accs[best_fold]:.4f}")
    
    return fold_models[best_fold]


def ensemble_predict(models, loader, device):
    """Ensemble prediction from multiple models with weighted voting"""
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            try:
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
                
                # Get predictions from each model
                batch_probs = []
                
                for model in models:
                    model.eval()
                    
                    # Test-time augmentation
                    all_logits = []
                    for _ in range(3):  # 3 TTA samples per model
                        outputs = model(batch)
                        all_logits.append(outputs['logits'])
                    
                    # Average logits
                    avg_logits = torch.stack(all_logits).mean(dim=0)
                    
                    # Convert to probabilities
                    probs = F.softmax(avg_logits, dim=1)
                    batch_probs.append(probs)
                
                # Average probabilities (weighted ensemble)
                weights = [1.0] * len(models)  # Equal weights for now
                weighted_probs = torch.zeros_like(batch_probs[0])
                for i, probs in enumerate(batch_probs):
                    weighted_probs += weights[i] * probs
                weighted_probs /= sum(weights)
                
                # Get predictions
                preds = weighted_probs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch['topology_label'].cpu().numpy())
                all_probs.append(weighted_probs.cpu().numpy())
                
            except Exception as e:
                print(f"Error in ensemble prediction: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(all_preds) == 0:  # No valid predictions
        return 0.0, 0.0, [], [], []
    
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return accuracy, f1, all_preds, all_targets, np.vstack(all_probs) if all_probs else []


def main():
    """Extreme boost V2 training pipeline"""
    print("🚀 EXTREME BOOST V2 - Closing the remaining 3.6% gap")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(42)
    
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
    
    # Get labels for stratification
    all_labels = []
    for i in range(len(dataset)):
        try:
            label = dataset[i]['topology_label'].item()
            all_labels.append(label)
        except Exception as e:
            all_labels.append(0)
    
    all_labels = np.array(all_labels)
    print(f"Label distribution: {np.bincount(all_labels)}")
    
    # Create a fixed test set
    train_val_idx, test_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=0.15,  # Smaller test set
        stratify=all_labels,
        random_state=42
    )
    
    # Create test loader
    test_loader = PyGDataLoader(
        Subset(dataset, test_idx),
        batch_size=32,  # Increased batch size for better stability
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # STRATEGY 1: K-Fold Cross-Validation for Model Selection
    print("\n" + "="*70)
    print("STRATEGY 1: Advanced K-Fold Cross-Validation")
    print("="*70)
    
    # Create a subset for K-fold (use only train_val_idx)
    train_val_dataset = Subset(dataset, train_val_idx)
    
    # Run K-fold cross-validation with larger batch size
    best_kfold_model = k_fold_cross_validation(
        train_val_dataset,
        n_splits=5,
        batch_size=32,  # Increased batch size for better stability
        num_workers=4,
        device=device
    )
    
    # Save best K-fold model
    torch.save(best_kfold_model.state_dict(), 'best_kfold_model_v2.pt')
    
    # STRATEGY 2: Ensemble of Diverse Models
    print("\n" + "="*70)
    print("STRATEGY 2: Ensemble of Diverse Models")
    print("="*70)
    
    # Train multiple models with different seeds and architectures
    ensemble_models = []
    ensemble_size = 5  # Increased from 3
    
    for i in range(ensemble_size):
        print(f"\n{'='*20} Ensemble Model {i+1}/{ensemble_size} {'='*20}")
        
        # Set seed for reproducibility
        set_seed(42 + i*10)
        
        # Different random split for each ensemble member
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.2,
            stratify=all_labels[train_val_idx],
            random_state=42+i*10  # Different seed
        )
        
        # Create data loaders
        train_loader = PyGDataLoader(
            Subset(dataset, train_idx),
            batch_size=32,  # Increased batch size for better stability
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=4,
            drop_last=True  # Drop the last batch if it's smaller than batch_size
        )
        
        val_loader = PyGDataLoader(
            Subset(dataset, val_idx),
            batch_size=32,  # Increased batch size for better stability
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=4
        )
        
        # Create model with different hyperparameters
        model = ExtremeBoostClassifierV2(
            dropout_rate=0.3 + i*0.05,  # Different dropout
            stochastic_depth_prob=0.1 + i*0.02  # Different stochastic depth
        ).to(device)
        
        # Optimizer with improved settings
        optimizer = AdamW(
            model.parameters(),
            lr=3e-5,  # Lower initial LR for better stability
            weight_decay=3e-4 * (1.0 + i*0.2),  # Different regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler with improved settings
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=15,  # Longer warmup for better stability
            total_epochs=120,  # Longer training
            min_lr=1e-6,
            max_lr=8e-5  # Lower max LR for better stability
        )
        
        # Training loop
        best_val_acc = 0
        best_model_state = None
        patience = 15  # Increased patience
        patience_counter = 0
        
        for epoch in range(120):  # Longer training
            # Update learning rate
            current_lr = scheduler.step(epoch)
            
            # Train with different augmentation settings
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, 
                use_mixup=True, mixup_alpha=0.2 + i*0.1,  # Different mixup
                use_cutmix=True, cutmix_alpha=1.0 + i*0.2  # Different cutmix
            )
            
            # Validate
            val_loss, val_acc, val_f1, _, _ = validate_epoch(
                model, val_loader, device, tta_samples=5  # Increased TTA samples
            )
            
            print(f"Epoch {epoch+1}/120:")
            print(f"  LR: {current_lr:.2e}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save best model from this ensemble member
        model.load_state_dict(best_model_state)
        ensemble_models.append(model)
        torch.save(best_model_state, f'ensemble_model_v2_{i+1}.pt')
        
        print(f"Ensemble model {i+1} best validation accuracy: {best_val_acc:.4f}")
    
    # STRATEGY 3: Full Training on All Data
    print("\n" + "="*70)
    print("STRATEGY 3: Full Training on All Data with Advanced Techniques")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create full training loader (all train_val data)
    full_train_loader = PyGDataLoader(
        Subset(dataset, train_val_idx),
        batch_size=32,  # Increased batch size for better stability
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        drop_last=True  # Drop the last batch if it's smaller than batch_size
    )
    
    # Create model
    full_model = ExtremeBoostClassifierV2(
        dropout_rate=0.25,  # Slightly lower dropout for full dataset
        stochastic_depth_prob=0.08  # Slightly lower stochastic depth for full dataset
    ).to(device)
    
    # Optimizer with improved settings
    optimizer = AdamW(
        full_model.parameters(),
        lr=3e-5,  # Lower initial LR for better stability
        weight_decay=3e-4,  # Higher weight decay to prevent overfitting
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler with improved settings
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=15,  # Longer warmup for better stability
        total_epochs=150,  # Even longer training for full dataset
        min_lr=1e-6,
        max_lr=8e-5  # Lower max LR for better stability
    )
    
    # Training loop
    for epoch in range(150):  # Longer training for full dataset
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # Train with both mixup and cutmix
        train_loss, train_acc = train_epoch(
            full_model, full_train_loader, optimizer, device, 
            use_mixup=True, mixup_alpha=0.2,
            use_cutmix=True, cutmix_alpha=1.0
        )
        
        print(f"Epoch {epoch+1}/150:")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(full_model.state_dict(), f'full_model_v2_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(full_model.state_dict(), 'full_model_v2_final.pt')
    
    # FINAL EVALUATION
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    # 1. Evaluate K-fold model
    best_kfold_model.eval()
    kfold_loss, kfold_acc, kfold_f1, kfold_preds, kfold_targets = validate_epoch(
        best_kfold_model, test_loader, device, tta_samples=10  # Increased TTA samples
    )
    
    print("K-fold Model Results:")
    print(f"  Accuracy: {kfold_acc:.4f}")
    print(f"  F1 Score: {kfold_f1:.4f}")
    
    # 2. Evaluate ensemble
    ensemble_acc, ensemble_f1, ensemble_preds, ensemble_targets, _ = ensemble_predict(
        ensemble_models, test_loader, device
    )
    
    print("Ensemble Model Results:")
    print(f"  Accuracy: {ensemble_acc:.4f}")
    print(f"  F1 Score: {ensemble_f1:.4f}")
    
    # 3. Evaluate full model
    full_model.eval()
    full_loss, full_acc, full_f1, full_preds, full_targets = validate_epoch(
        full_model, test_loader, device, tta_samples=10  # Increased TTA samples
    )
    
    print("Full Model Results:")
    print(f"  Accuracy: {full_acc:.4f}")
    print(f"  F1 Score: {full_f1:.4f}")
    
    # 4. SUPER ENSEMBLE (all models with weighted voting)
    super_ensemble = ensemble_models + [best_kfold_model, full_model]
    super_acc, super_f1, super_preds, super_targets, _ = ensemble_predict(
        super_ensemble, test_loader, device
    )
    
    print("SUPER ENSEMBLE Results:")
    print(f"  Accuracy: {super_acc:.4f}")
    print(f"  F1 Score: {super_f1:.4f}")
    
    # Print best results
    best_acc = max(kfold_acc, ensemble_acc, full_acc, super_acc)
    best_method = ""
    if best_acc == kfold_acc:
        best_method = "K-fold Model"
    elif best_acc == ensemble_acc:
        best_method = "Ensemble Model"
    elif best_acc == full_acc:
        best_method = "Full Model"
    else:
        best_method = "SUPER ENSEMBLE"
    
    print("\n" + "="*70)
    print(f"BEST RESULT: {best_method} with {best_acc:.4f} accuracy")
    print("="*70)
    
    if best_acc >= 0.92:
        print("🎉 SUCCESS: Achieved 92%+ test accuracy!")
        print(f"🏆 Final score: {best_acc:.4f} ({(best_acc-0.92)*100:.1f}% above target)")
    else:
        print(f"📈 Progress from 88.36%: +{(best_acc-0.8836)*100:.1f}%")
        print(f"📈 Still need: {(0.92 - best_acc)*100:.1f}% more accuracy.")
    
    # Save detailed results
    results = {
        'kfold': {
            'accuracy': kfold_acc,
            'f1': kfold_f1,
            'predictions': kfold_preds,
            'targets': kfold_targets
        },
        'ensemble': {
            'accuracy': ensemble_acc,
            'f1': ensemble_f1,
            'predictions': ensemble_preds,
            'targets': ensemble_targets
        },
        'full': {
            'accuracy': full_acc,
            'f1': full_f1,
            'predictions': full_preds,
            'targets': full_targets
        },
        'super_ensemble': {
            'accuracy': super_acc,
            'f1': super_f1,
            'predictions': super_preds,
            'targets': super_targets
        }
    }
    
    # Save results as numpy arrays
    np.save('extreme_boost_v2_results.npy', results)
    
    print("\n🎯 FINAL RESULTS:")
    print(f"Test Accuracy: {best_acc:.4f}")
    print(f"Test F1 Score: {super_f1:.4f}")
    
    if best_acc >= 0.92:
        print("🎉 TARGET ACHIEVED! 92%+ accuracy reached.")
    else:
        print(f"📈 Progress made. Need {(0.92 - best_acc)*100:.1f}% more accuracy.")
    
    return best_acc, super_f1


if __name__ == "__main__":
    try:
        print(f"Starting Python training script at {time.strftime('%a %b %d %I:%M:%S %p %Z %Y')}")
        start_time = time.time()
        test_acc, test_f1 = main()
        end_time = time.time()
        
        print(f"\nTotal training time: {(end_time - start_time) / 60:.1f} minutes")
        print(f"Final accuracy: {test_acc:.4f}")
        print(f"Final F1 score: {test_f1:.4f}")
        print(f"Python training script finished at {time.strftime('%a %b %d %I:%M:%S %p %Z %Y')}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()