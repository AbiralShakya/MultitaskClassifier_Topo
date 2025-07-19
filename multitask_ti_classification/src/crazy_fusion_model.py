"""
Crazy Fusion Model - State-of-the-art multi-branch transformer fusion architecture
for topological insulator classification with cross-modal attention and deep residuals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
import math
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper.config import *
from helper.gpu_spectral_encoder import GPUSpectralEncoder


class MultiHeadCrossAttention(nn.Module):
    """Cross-modal attention mechanism for fusing different modalities."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection and residual connection
        output = self.w_o(context)
        output = self.layer_norm(output + query)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x


class DeepResidualBlock(nn.Module):
    """Deep residual block with multiple layers and skip connections."""
    
    def __init__(self, dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            x = layer(x) + residual
            residual = x
        return x


class CrystalGraphEncoder(nn.Module):
    """Deep crystal graph encoder with multiple GNN layers and residual connections."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multiple GNN layers with different architectures
        self.gnn_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1) if i % 2 == 0 
            else GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1)
            for i in range(num_layers)
        ])
        
        # Layer normalization and residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Multiple GNN layers with residuals
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            residual = x
            x = gnn_layer(x, edge_index)
            x = layer_norm(x + residual)
            x = F.relu(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class KSpaceEncoder(nn.Module):
    """Enhanced k-space encoder with multiple GNN variants and attention."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 gnn_type: str = "transformer", num_layers: int = 3):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers based on type
        if gnn_type == "transformer":
            self.gnn_layers = nn.ModuleList([
                TransformerConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1)
                for _ in range(num_layers)
            ])
        elif gnn_type == "gat":
            self.gnn_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1)
                for _ in range(num_layers)
            ])
        elif gnn_type == "gcn":
            self.gnn_layers = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
        elif gnn_type == "sage":
            self.gnn_layers = nn.ModuleList([
                SAGEConv(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers with residuals
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            residual = x
            x = gnn_layer(x, edge_index)
            x = layer_norm(x + residual)
            x = F.relu(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class ScalarFeatureEncoder(nn.Module):
    """Deep scalar feature encoder with residual blocks."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int = 3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Deep residual blocks
        self.residual_blocks = nn.ModuleList([
            DeepResidualBlock(hidden_dim, num_layers=3, dropout=0.1)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.output_proj(x)
        return x


class DecompositionEncoder(nn.Module):
    """Decomposition feature encoder with deep MLP."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # Deep MLP for decomposition features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SpectralEncoder(nn.Module):
    """Spectral encoder using GPU-accelerated computation."""
    
    def __init__(self, k_eigs: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.gpu_spectral = GPUSpectralEncoder(k_eigs, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, edge_index: torch.Tensor, num_nodes: int, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.gpu_spectral(edge_index, num_nodes, batch)
        x = self.output_proj(x)
        return x


class CrossModalFusion(nn.Module):
    """Cross-modal fusion using transformer blocks and attention."""
    
    def __init__(self, modality_dims: Dict[str, int], fusion_dim: int = 512, 
                 num_blocks: int = 3, n_heads: int = 8):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        
        # Project all modalities to same dimension
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, fusion_dim)
            for name, dim in modality_dims.items()
        })
        
        # Cross-modal attention layers
        self.cross_attention_layers = nn.ModuleList([
            MultiHeadCrossAttention(fusion_dim, n_heads, dropout=0.1)
            for _ in range(num_blocks)
        ])
        
        # Transformer blocks for final fusion
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(fusion_dim, n_heads, fusion_dim * 4, dropout=0.1)
            for _ in range(num_blocks)
        ])
        
        # Final fusion MLP - will be created dynamically based on actual modalities
        self.fusion_dim = fusion_dim
        
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Project all modalities to same dimension
        projected_features = {}
        for name, features in modality_features.items():
            if name in self.projections:
                projected_features[name] = self.projections[name](features)
        
        # Ensure all tensors have the same shape [batch_size, fusion_dim]
        batch_size = None
        for name, features in projected_features.items():
            if batch_size is None:
                batch_size = features.size(0) if features.dim() > 1 else 1
            if features.dim() == 1:
                # If tensor is 1D, expand to [1, fusion_dim] then repeat to batch_size
                features = features.unsqueeze(0).expand(batch_size, -1)
                projected_features[name] = features
        
        # Stack features for cross-modal attention
        stacked_features = torch.stack(list(projected_features.values()), dim=1)  # [batch, num_modalities, fusion_dim]
        
        # Apply cross-modal attention
        fused_features = stacked_features
        for cross_attn in self.cross_attention_layers:
            fused_features = cross_attn(fused_features, fused_features, fused_features)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            fused_features = transformer_block(fused_features)
        
        # Global pooling across modalities
        fused_features = torch.mean(fused_features, dim=1)  # [batch, fusion_dim]
        
        # Concatenate original features for final fusion
        concatenated = torch.cat(list(projected_features.values()), dim=1)
        
        # Create fusion MLP dynamically based on actual input size
        input_dim = concatenated.size(1)
        fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, self.fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim, self.fusion_dim // 2)
        ).to(concatenated.device)
        
        # Final fusion MLP
        output = fusion_mlp(concatenated)
        
        return output


class CrazyFusionModel(nn.Module):
    """State-of-the-art multi-branch transformer fusion model."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Configuration
        self.config = config
        self.hidden_dim = config.get('HIDDEN_DIM', 256)
        self.fusion_dim = config.get('FUSION_DIM', 512)
        self.num_classes = config.get('NUM_CLASSES', 2)
        
        # Modality flags
        self.use_crystal = config.get('USE_CRYSTAL', True)
        self.use_kspace = config.get('USE_KSPACE', True)
        self.use_scalar = config.get('USE_SCALAR', True)
        self.use_decomposition = config.get('USE_DECOMPOSITION', True)
        self.use_spectral = config.get('USE_SPECTRAL', True)
        
        # Initialize encoders
        self.encoders = nn.ModuleDict()
        
        if self.use_crystal:
            self.encoders['crystal'] = CrystalGraphEncoder(
                input_dim=config.get('CRYSTAL_INPUT_DIM', 92),
                hidden_dim=self.hidden_dim,
                output_dim=self.fusion_dim,
                num_layers=config.get('CRYSTAL_LAYERS', 4)
            )
        
        if self.use_kspace:
            self.encoders['kspace'] = KSpaceEncoder(
                input_dim=config.get('KSPACE_INPUT_DIM', 2),
                hidden_dim=self.hidden_dim,
                output_dim=self.fusion_dim,
                gnn_type=config.get('KSPACE_GNN_TYPE', 'transformer'),
                num_layers=config.get('KSPACE_LAYERS', 3)
            )
        
        if self.use_scalar:
            self.encoders['scalar'] = ScalarFeatureEncoder(
                input_dim=config.get('SCALAR_INPUT_DIM', 200),
                hidden_dim=self.hidden_dim,
                output_dim=self.fusion_dim,
                num_blocks=config.get('SCALAR_BLOCKS', 3)
            )
        
        if self.use_decomposition:
            self.encoders['decomposition'] = DecompositionEncoder(
                input_dim=config.get('DECOMPOSITION_INPUT_DIM', 100),
                hidden_dim=self.hidden_dim,
                output_dim=self.fusion_dim
            )
        
        if self.use_spectral:
            self.encoders['spectral'] = SpectralEncoder(
                k_eigs=config.get('K_EIGS', 64),
                hidden_dim=self.hidden_dim,
                output_dim=self.fusion_dim
            )
        
        # Cross-modal fusion
        modality_dims = {name: self.fusion_dim for name in self.encoders.keys()}
        self.fusion = CrossModalFusion(
            modality_dims=modality_dims,
            fusion_dim=self.fusion_dim,
            num_blocks=config.get('FUSION_BLOCKS', 3),
            n_heads=config.get('FUSION_HEADS', 8)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim // 2, self.fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim // 4, self.num_classes)
        )
        
        # Attention weights for visualization
        self.attention_weights = {}
        
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        modality_features = {}
        
        # Process each modality
        if self.use_crystal and 'crystal_x' in batch_data and 'crystal_edge_index' in batch_data:
            crystal_features = self.encoders['crystal'](
                batch_data['crystal_x'],
                batch_data['crystal_edge_index'],
                batch_data.get('crystal_batch')
            )
            modality_features['crystal'] = crystal_features
        
        if self.use_kspace and 'kspace_x' in batch_data and 'kspace_edge_index' in batch_data:
            kspace_features = self.encoders['kspace'](
                batch_data['kspace_x'],
                batch_data['kspace_edge_index'],
                batch_data.get('kspace_batch')
            )
            modality_features['kspace'] = kspace_features
        
        if self.use_scalar and 'scalar_features' in batch_data:
            scalar_features = self.encoders['scalar'](batch_data['scalar_features'])
            modality_features['scalar'] = scalar_features
        
        if self.use_decomposition and 'decomposition_features' in batch_data:
            decomposition_features = self.encoders['decomposition'](batch_data['decomposition_features'])
            modality_features['decomposition'] = decomposition_features
        
        if self.use_spectral and 'spectral_edge_index' in batch_data:
            spectral_features = self.encoders['spectral'](
                batch_data['spectral_edge_index'],
                batch_data.get('spectral_num_nodes', 100),
                batch_data.get('spectral_batch')
            )
            modality_features['spectral'] = spectral_features
        
        # Cross-modal fusion
        if modality_features:
            fused_features = self.fusion(modality_features)
            logits = self.classifier(fused_features)
        else:
            # Fallback if no modalities are enabled
            batch_size = next(iter(batch_data.values())).size(0) if batch_data else 1
            logits = torch.zeros(batch_size, self.num_classes, device=next(iter(batch_data.values())).device)
        
        return logits
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization."""
        return self.attention_weights.copy()
    
    def clear_attention_weights(self):
        """Clear stored attention weights."""
        self.attention_weights.clear()


def create_crazy_fusion_model(config: dict) -> CrazyFusionModel:
    """Factory function to create the crazy fusion model."""
    return CrazyFusionModel(config)


if __name__ == "__main__":
    # Test the model
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
        'KSPACE_GNN_TYPE': 'transformer'
    }
    
    model = create_crazy_fusion_model(test_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters") 