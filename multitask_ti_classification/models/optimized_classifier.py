"""
Optimized Multi-Modal Material Classifier
Designed for 92%+ accuracy without overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import numpy as np

class OptimizedMaterialClassifier(nn.Module):
    """
    Clean, optimized classifier with balanced capacity and regularization
    """
    
    def __init__(
        self,
        crystal_node_feature_dim=3,
        kspace_node_feature_dim=64,
        scalar_feature_dim=10,
        decomposition_feature_dim=64,
        num_topology_classes=2,
        hidden_dim=256,
        dropout_rate=0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Crystal structure encoder (GNN)
        self.crystal_conv1 = GCNConv(crystal_node_feature_dim, 64)
        self.crystal_conv2 = GCNConv(64, 128)
        self.crystal_conv3 = GCNConv(128, 128)
        
        # K-space encoder (GNN)
        self.kspace_conv1 = GCNConv(kspace_node_feature_dim, 128)
        self.kspace_conv2 = GCNConv(128, 256)
        self.kspace_conv3 = GCNConv(256, 256)
        
        # Scalar features encoder
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128)
        )
        
        # Decomposition features encoder
        self.decomp_encoder = nn.Sequential(
            nn.Linear(decomposition_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256)
        )
        
        # Feature fusion with attention
        total_feature_dim = 256 + 512 + 128 + 256  # crystal + kspace + scalar + decomp
        
        self.attention = nn.MultiheadAttention(
            embed_dim=total_feature_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classification head with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, num_topology_classes)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode_crystal(self, crystal_data):
        """Encode crystal structure with GNN"""
        x, edge_index, batch = crystal_data.x, crystal_data.edge_index, crystal_data.batch
        
        # GNN layers with residual connections
        x1 = F.relu(self.crystal_conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)
        
        x2 = F.relu(self.crystal_conv2(x1, edge_index))
        x2 = F.dropout(x2, p=self.dropout_rate, training=self.training)
        
        x3 = self.crystal_conv3(x2, edge_index)
        x3 = x3 + x2  # Residual connection
        x3 = F.relu(x3)
        
        # Global pooling
        mean_pool = global_mean_pool(x3, batch)
        max_pool = global_max_pool(x3, batch)
        
        return torch.cat([mean_pool, max_pool], dim=1)  # 256 features
    
    def encode_kspace(self, kspace_data):
        """Encode k-space with GNN"""
        x, edge_index, batch = kspace_data.x, kspace_data.edge_index, kspace_data.batch
        
        # GNN layers with residual connections
        x1 = F.relu(self.kspace_conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)
        
        x2 = F.relu(self.kspace_conv2(x1, edge_index))
        x2 = F.dropout(x2, p=self.dropout_rate, training=self.training)
        
        x3 = self.kspace_conv3(x2, edge_index)
        x3 = x3 + x2  # Residual connection
        x3 = F.relu(x3)
        
        # Global pooling
        mean_pool = global_mean_pool(x3, batch)
        max_pool = global_max_pool(x3, batch)
        
        return torch.cat([mean_pool, max_pool], dim=1)  # 512 features
    
    def forward(self, batch):
        """Forward pass with all modalities"""
        
        # Encode each modality
        crystal_emb = self.encode_crystal(batch['crystal_graph'])  # 256
        kspace_emb = self.encode_kspace(batch['kspace_graph'])     # 512
        scalar_emb = self.scalar_encoder(batch['scalar_features']) # 128
        decomp_emb = self.decomp_encoder(batch['decomposition_features']) # 256
        
        # Concatenate all features
        features = torch.cat([crystal_emb, kspace_emb, scalar_emb, decomp_emb], dim=1)
        
        # Apply self-attention for feature interaction
        features_unsqueezed = features.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.attention(
            features_unsqueezed, features_unsqueezed, features_unsqueezed
        )
        features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Add residual connection
        features = features + torch.cat([crystal_emb, kspace_emb, scalar_emb, decomp_emb], dim=1)
        
        # Classification
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features,
            'crystal_emb': crystal_emb,
            'kspace_emb': kspace_emb,
            'scalar_emb': scalar_emb,
            'decomp_emb': decomp_emb
        }
    
    def compute_loss(self, predictions, targets, alpha=0.1):
        """Enhanced loss with focal loss and regularization"""
        
        # Focal loss for handling class imbalance
        ce_loss = F.cross_entropy(predictions['logits'], targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** 2 * ce_loss
        focal_loss = focal_loss.mean()
        
        # Feature diversity regularization
        features = predictions['features']
        feature_std = torch.std(features, dim=0).mean()
        diversity_loss = -torch.log(feature_std + 1e-8)  # Encourage feature diversity
        
        return focal_loss + alpha * diversity_loss


class EarlyStopping:
    """Early stopping with patience and best model saving"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


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
            # Warmup phase
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr