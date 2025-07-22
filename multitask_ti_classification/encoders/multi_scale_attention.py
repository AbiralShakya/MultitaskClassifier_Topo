import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.norm import BatchNorm

class MultiScaleGraphAttention(nn.Module):
    """
    Multi-scale graph attention network inspired by the Nature paper.
    Captures both local and global topological features.
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projections
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim // 4)
        
        # Multi-scale attention layers
        self.attention_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        
        for i in range(num_layers):
            # Graph attention with different head configurations for multi-scale
            heads = max(1, num_heads // (2 ** i))  # Reduce heads in deeper layers
            
            self.attention_layers.append(
                GATConv(
                    hidden_dim, 
                    hidden_dim // heads, 
                    heads=heads,
                    dropout=0.1,
                    edge_dim=hidden_dim // 4,
                    concat=True
                )
            )
            
            self.batch_norms.append(BatchNorm(hidden_dim))
            
            # Feed-forward network
            self.feed_forwards.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
            )
        
        # Global attention for topological features
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Topological feature extractors
        self.local_topo_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.global_topo_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Multi-scale pooling
        self.scale_weights = nn.Parameter(torch.ones(3))  # For mean, max, add pooling
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Input projections
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # Store intermediate representations for multi-scale
        scale_representations = []
        
        # Multi-scale attention layers
        for i, (attn, bn, ff) in enumerate(zip(self.attention_layers, self.batch_norms, self.feed_forwards)):
            # Attention with residual connection
            x_attn = attn(x, edge_index, edge_attr)
            x = x + x_attn
            x = bn(x)
            
            # Feed-forward with residual connection
            x_ff = ff(x)
            x = x + x_ff
            
            # Store representation for this scale
            scale_representations.append(x.clone())
        
        # Extract local topological features
        local_features = self.local_topo_extractor(x)
        
        # Global attention for long-range topological interactions
        # Reshape for global attention (batch_size, num_nodes, hidden_dim)
        batch_size = batch.max().item() + 1
        max_nodes = torch.bincount(batch).max().item()
        
        # Create padded tensor for global attention
        x_global = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=x.device)
        attention_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=x.device)
        
        for b in range(batch_size):
            mask = batch == b
            nodes_in_batch = mask.sum().item()
            x_global[b, :nodes_in_batch] = x[mask]
            attention_mask[b, :nodes_in_batch] = False
        
        # Apply global attention
        x_global_attn, _ = self.global_attention(
            x_global, x_global, x_global, 
            key_padding_mask=attention_mask
        )
        
        # Extract global topological features
        global_features = self.global_topo_extractor(x_global_attn.mean(dim=1))
        
        # Multi-scale pooling
        pooled_features = []
        
        # Mean pooling
        x_mean = global_mean_pool(x, batch)
        pooled_features.append(self.scale_weights[0] * x_mean)
        
        # Max pooling  
        x_max = global_max_pool(x, batch)
        pooled_features.append(self.scale_weights[1] * x_max)
        
        # Add pooling
        x_add = global_add_pool(x, batch)
        pooled_features.append(self.scale_weights[2] * x_add)
        
        # Combine multi-scale features
        x_pooled = torch.stack(pooled_features).mean(dim=0)
        
        # Combine local and global topological features
        topo_features = torch.cat([
            global_mean_pool(local_features, batch),
            global_features
        ], dim=1)
        
        # Final feature combination
        final_features = torch.cat([x_pooled, topo_features], dim=1)
        
        return self.output_proj(final_features)


class TopologicalFeatureExtractor(nn.Module):
    """
    Specialized module for extracting topological invariant-related features
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # Chern number related features
        self.chern_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),  # Bounded activation for Chern-like features
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Z2 invariant related features
        self.z2_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),  # Binary-like features for Z2
            nn.Linear(hidden_dim // 2, 4)  # 4 Z2 invariants
        )
        
        # Band gap related features
        self.gap_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Symmetry-based features
        self.symmetry_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32)  # Space group related features
        )
    
    def forward(self, x):
        """Extract topological features"""
        chern_features = self.chern_extractor(x)
        z2_features = self.z2_extractor(x)
        gap_features = self.gap_extractor(x)
        symmetry_features = self.symmetry_extractor(x)
        
        return {
            'chern': chern_features,
            'z2': z2_features,
            'gap': gap_features,
            'symmetry': symmetry_features,
            'combined': torch.cat([chern_features, z2_features, gap_features], dim=1)
        }


class EnhancedTopologicalClassifier(nn.Module):
    """
    Enhanced classifier combining multi-scale attention with topological feature extraction
    """
    
    def __init__(self, node_dim, edge_dim, num_classes=3, hidden_dim=256):
        super().__init__()
        
        # Multi-scale graph attention
        self.graph_encoder = MultiScaleGraphAttention(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim
        )
        
        # Topological feature extractor
        self.topo_extractor = TopologicalFeatureExtractor(hidden_dim)
        
        # Classification heads
        self.main_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Auxiliary topological consistency head
        self.topo_consistency_head = nn.Sequential(
            nn.Linear(6, 32),  # chern(1) + z2(4) + gap(1) = 6
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Extract graph features
        graph_features = self.graph_encoder(x, edge_index, edge_attr, batch)
        
        # Extract topological features
        topo_features = self.topo_extractor(graph_features)
        
        # Main classification
        main_logits = self.main_classifier(graph_features)
        
        # Auxiliary topological classification
        aux_logits = self.topo_consistency_head(topo_features['combined'])
        
        # Confidence estimation
        confidence = self.confidence_head(graph_features)
        
        return {
            'logits': main_logits,
            'aux_logits': aux_logits,
            'confidence': confidence,
            'topo_features': topo_features,
            'graph_features': graph_features
        }