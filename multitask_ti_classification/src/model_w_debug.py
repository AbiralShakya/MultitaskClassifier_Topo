import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, TransformerConv
import torch_geometric
from torch_geometric.data import Data as PyGData
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear, Irreps
from e3nn.nn import Gate
from typing import List, Dict, Any, Tuple
import math
import numpy as np

import helper.config as config
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# --- Multi-Head Attention Mechanism ---
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for better feature fusion."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + output)
        
        return output

# --- Data Augmentation Functions ---
def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def add_feature_noise(features, noise_std=0.01):
    """Add small noise to features for regularization."""
    noise = torch.randn_like(features) * noise_std
    return features + noise

# --- Focal Loss ---
class FocalLoss(nn.Module):
    """Focal Loss for better handling of class imbalance."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- 1. Crystal Graph Encoder (RealSpaceEGNN) ---

class EGNNLayer(nn.Module):
    """
    A simplified EGNN-like layer using e3nn components.
    This assumes `node_features` are e3nn.o3.Irreps-wrapped tensors and `edge_attr_e3nn` too.
    """
    def __init__(self, node_irreps_in: Irreps, edge_irreps_in: Irreps, hidden_irreps: Irreps):
        super().__init__()
        self.node_irreps_in = node_irreps_in
        self.edge_irreps_in = edge_irreps_in
        self.hidden_irreps = hidden_irreps

        # TP for messages (from node_j and edge_attr) -> message irreps
        self.tp_messages_ij = FullyConnectedTensorProduct(
            node_irreps_in, edge_irreps_in, hidden_irreps,
        )

        # Direct linear path for messages (no tensor product)
        self.linear_messages_direct = Linear(node_irreps_in, hidden_irreps)

        # TP for update (from node_i and aggregated_messages) -> new node irreps
        self.tp_update = FullyConnectedTensorProduct(
            node_irreps_in, hidden_irreps, hidden_irreps,
        )

        # Direct linear path for update
        self.linear_update_direct = Linear(node_irreps_in, hidden_irreps)
        
    def forward(self, node_features, edge_index: torch.Tensor, edge_attr_tensor: torch.Tensor, 
                node_attr_scalar_raw: torch.Tensor):
        
        row, col = edge_index

        # FIX 1: Properly convert edge attributes to e3nn format
        # Extract components: [r_x, r_y, r_z, distance]
        r_vec = edge_attr_tensor[:, :3]  # (num_edges, 3) - vector part (1o)
        dist = edge_attr_tensor[:, 3:4]  # (num_edges, 1) - scalar part (0e)
        
        # Create proper e3nn edge attributes: concatenate scalar + vector
        edge_attr_e3nn = torch.cat([dist, r_vec], dim=-1)  # (num_edges, 4)

        # 1. Message passing with both tensor product and direct paths
        messages_tp_output = self.tp_messages_ij(node_features[col], edge_attr_e3nn)
        messages_direct = self.linear_messages_direct(node_features[col])
        messages_from_j = messages_tp_output + messages_direct

        # 2. Aggregation (sum messages for each node)
        aggregated_messages = torch_geometric.utils.scatter(
            messages_from_j, row, dim=0, dim_size=node_features.size(0), reduce="sum"
        )

        # 3. Update with both tensor product and direct paths
        updated_node_features_tp_output = self.tp_update(node_features, aggregated_messages)
        updated_node_features_direct = self.linear_update_direct(node_features)
        updated_node_features = updated_node_features_tp_output + updated_node_features_direct
        
        # Residual connection
        result = node_features + updated_node_features
        
        return result


class RealSpaceEGNNEncoder(nn.Module):
    """
    EGNN encoder for real-space atomic crystal graphs.
    Handles atomic features (Z, period, group) and spatial coordinates.
    """
    def __init__(self, 
                 node_input_scalar_dim: int,
                 hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
                 n_layers: int = 6,  # Increased from 3
                 radius: float = 4.0  # Updated to match config
                ):
        super().__init__()
        
        self.node_input_scalar_dim = node_input_scalar_dim
        
        self.input_node_irreps = Irreps(f"{node_input_scalar_dim}x0e")
        # FIX 2: Correct edge irreps order - scalar first, then vector
        self.edge_irreps = Irreps("1x0e + 1x1o")  # Changed order: scalar + vector
        self.hidden_irreps = Irreps(hidden_irreps_str)
        
        self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EGNNLayer(
                node_irreps_in=self.hidden_irreps,
                edge_irreps_in=self.edge_irreps,
                hidden_irreps=self.hidden_irreps
            ))

        # Extract only the scalar (0e) irreps from hidden_irreps for final projection
        scalar_irreps_list = []
        for mul, irrep in self.hidden_irreps:
            if irrep.l == 0:  # scalar irrep
                scalar_irreps_list.append((mul, irrep))
        
        if scalar_irreps_list:
            self.scalar_irreps = Irreps(scalar_irreps_list)
        else:
            # Fallback if no scalars found
            self.scalar_irreps = Irreps("1x0e")
            
        output_irreps = Irreps(f"{config.LATENT_DIM_GNN}x0e")
        self.final_linear_0e = Linear(self.scalar_irreps, output_irreps)

        self.radius = radius

    def forward(self, data: PyGData) -> torch.Tensor:
        x_raw_scalars = data.x  # (N_atoms, node_input_scalar_dim)
        
        # 1. Project input scalar features to e3nn's structure
        x_e3nn = self.initial_projection(x_raw_scalars) 

        # 2. Recompute edge_index and edge_attr (relative positions)
        edge_index = torch_geometric.nn.radius_graph(data.pos, self.radius, data.batch)
        row, col = edge_index
        
        r_vec = data.pos[row] - data.pos[col]
        dist = r_vec.norm(dim=-1, keepdim=True)
        
        # FIX 3: Handle zero distances to avoid division by zero
        normalized_r_vec = r_vec / (dist.clamp(min=1e-8))
        
        # Create edge attributes tensor: [r_x, r_y, r_z, distance]
        edge_attr_tensor = torch.cat([normalized_r_vec, dist], dim=-1)

        # 3. Pass through EGNN layers
        current_node_features_e3nn = x_e3nn
        for layer in self.layers:
            current_node_features_e3nn = layer(
                current_node_features_e3nn, 
                edge_index, 
                edge_attr_tensor,
                x_raw_scalars
            )

        # 4. Extract scalar features for global pooling
        # FIX 4: Properly extract scalar features using irreps structure
        scalar_features = []
        start_idx = 0
        for mul, irrep in self.hidden_irreps:
            end_idx = start_idx + mul * irrep.dim
            if irrep.l == 0:  # scalar irrep
                scalar_features.append(current_node_features_e3nn[:, start_idx:end_idx])
            start_idx = end_idx
        
        if scalar_features:
            invariant_features_per_node = torch.cat(scalar_features, dim=1)
        else:
            # Fallback: use first 64 features as scalars
            invariant_features_per_node = current_node_features_e3nn[:, :64]
        
        # Global mean pool on the invariant features
        graph_embedding_tensor = global_mean_pool(invariant_features_per_node, data.batch)
        
        # 5. Final projection to LATENT_DIM_GNN
        final_embedding = self.final_linear_0e(graph_embedding_tensor)
        
        return final_embedding
    
# --- 2. K-space Graph Encoder (TransformerConv) ---
class KSpaceTransformerGNNEncoder(nn.Module):
    """
    Simplified TransformerConv-based encoder for k-space graphs.
    """
    def __init__(self, node_feature_dim: int, hidden_dim: int, out_channels: int, # out_channels is LATENT_DIM_GNN
                 n_layers: int = 8, num_heads: int = 8):  # Reduced complexity
        super().__init__()
        
        self.initial_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        current_in_channels = hidden_dim
        
        for i in range(n_layers):
            self.layers.append(TransformerConv(
                in_channels=current_in_channels,
                out_channels=hidden_dim // num_heads,  # Ensure output is hidden_dim
                heads=num_heads,
                dropout=config.DROPOUT_RATE,
                beta=True
            ))
            self.bns.append(nn.LayerNorm(hidden_dim))  # Fixed dimension

            current_in_channels = hidden_dim # Update for next layer

        # Final projection to desired output dimension
        self.final_projection = nn.Linear(current_in_channels, out_channels)


    def forward(self, data: PyGData) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.initial_projection(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)

        # Global mean pool
        pooled_x = global_mean_pool(x, batch)

        # Final projection
        final_embedding = self.final_projection(pooled_x)

        return final_embedding

class GCNEncoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_channels, n_layers=8, num_heads=8):
        super().__init__()
        self.initial_projection = nn.Linear(node_feature_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.LayerNorm(hidden_dim))
        self.final_projection = nn.Linear(hidden_dim, out_channels)
    def forward(self, data: PyGData) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.initial_projection(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)
        pooled_x = global_mean_pool(x, batch)
        return self.final_projection(pooled_x)

class GATEncoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_channels, n_layers=8, num_heads=8):
        super().__init__()
        self.initial_projection = nn.Linear(node_feature_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=config.DROPOUT_RATE))
            self.bns.append(nn.LayerNorm(hidden_dim))
        self.final_projection = nn.Linear(hidden_dim, out_channels)
    def forward(self, data: PyGData) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.initial_projection(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)
        pooled_x = global_mean_pool(x, batch)
        return self.final_projection(pooled_x)

class GraphSAGEEncoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_channels, n_layers=8, num_heads=8):
        super().__init__()
        self.initial_projection = nn.Linear(node_feature_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.LayerNorm(hidden_dim))
        self.final_projection = nn.Linear(hidden_dim, out_channels)
    def forward(self, data: PyGData) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.initial_projection(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)
        pooled_x = global_mean_pool(x, batch)
        return self.final_projection(pooled_x)

# --- 3. Scalar Features Encoder ---

class ScalarFeatureEncoder(nn.Module):
    """
    FFNN encoder for combined scalar features (band_rep_features + metadata features).
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], out_channels: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))  # Changed to LayerNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=config.DROPOUT_RATE))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, out_channels))
        layers.append(nn.LayerNorm(out_channels))  # Changed to LayerNorm
        layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

# --- 4. Decomposition Features Encoder ---

class DecompositionFeatureEncoder(nn.Module):
    """
    FFNN encoder for decomposition branches features.
    """
    def __init__(self, input_dim: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, out_channels * 2), 
            nn.LayerNorm(out_channels * 2),  # Changed to LayerNorm
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(out_channels * 2, out_channels),
            nn.LayerNorm(out_channels),  # Changed to LayerNorm
            nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

# --- 5. Main Multi-Modal Classifier ---

class GatedFusion(nn.Module):
    def __init__(self, num_modalities):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(num_modalities))
    def forward(self, features):
        # features: list of tensors [batch, dim]
        gated = [f * torch.sigmoid(self.gate[i]) for i, f in enumerate(features)]
        return torch.cat(gated, dim=-1)

class MultiModalMaterialClassifier(nn.Module):
    """
    Enhanced multi-modal classifier for materials with attention, ensemble, and data augmentation.
    Combines Real-space EGNN, K-space Transformer GNN, and Scalar features.
    """
    def __init__(
        self,
        crystal_node_feature_dim: int,
        kspace_node_feature_dim: int,
        scalar_feature_dim: int,
        decomposition_feature_dim: int,
        num_topology_classes: int,
        num_magnetism_classes: int,
        
        # Encoder specific params
        egnn_hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
        egnn_num_layers: int = 6,  # Increased from 3
        egnn_radius: float = 4.0,  # Updated to match config
        
        kspace_gnn_hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers: int = 8,  # Reduced from 12
        kspace_gnn_num_heads: int = 8,  # Reduced from 16
        
        ffnn_hidden_dims_scalar: List[int] = config.FFNN_HIDDEN_DIMS_SCALAR,
        
        # Shared fusion params
        latent_dim_gnn: int = config.LATENT_DIM_GNN,
        latent_dim_ffnn: int = config.LATENT_DIM_FFNN,
        fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,
        
        # NEW: Attention and ensemble parameters
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        num_ensemble_heads: int = 3,
        use_mixup: bool = True,
        mixup_alpha: float = 0.2,
        feature_noise_std: float = 0.01,
    ):
        super().__init__()

        self.crystal_encoder = RealSpaceEGNNEncoder(
            node_input_scalar_dim=crystal_node_feature_dim,
            hidden_irreps_str=egnn_hidden_irreps_str,
            n_layers=egnn_num_layers,
            radius=egnn_radius,
        )
        self.active_modalities = []
        self.modalities = []
        if config.USE_CRYSTAL:
            self.active_modalities.append('crystal')
            self.modalities.append('crystal')
        if config.USE_KSPACE:
            self.active_modalities.append('kspace')
            self.modalities.append('kspace')
        if config.USE_SCALAR:
            self.active_modalities.append('scalar')
            self.modalities.append('scalar')
        if config.USE_DECOMPOSITION:
            self.active_modalities.append('decomposition')
            self.modalities.append('decomposition')
        print(f"[Model] Active modalities: {self.active_modalities}")
        # K-space GNN selection
        if config.KSPACE_GNN_TYPE == 'transformer':
            self.kspace_encoder = KSpaceTransformerGNNEncoder(
                node_feature_dim=kspace_node_feature_dim,
                hidden_dim=kspace_gnn_hidden_channels,
                out_channels=latent_dim_gnn,
                n_layers=kspace_gnn_num_layers,
                num_heads=kspace_gnn_num_heads
            )
        elif config.KSPACE_GNN_TYPE == 'gcn':
            self.kspace_encoder = GCNEncoder(
                node_feature_dim=kspace_node_feature_dim,
                hidden_dim=kspace_gnn_hidden_channels,
                out_channels=latent_dim_gnn,
                n_layers=kspace_gnn_num_layers,
                num_heads=kspace_gnn_num_heads
            )
        elif config.KSPACE_GNN_TYPE == 'gat':
            self.kspace_encoder = GATEncoder(
                node_feature_dim=kspace_node_feature_dim,
                hidden_dim=kspace_gnn_hidden_channels,
                out_channels=latent_dim_gnn,
                n_layers=kspace_gnn_num_layers,
                num_heads=kspace_gnn_num_heads
            )
        elif config.KSPACE_GNN_TYPE == 'sage':
            self.kspace_encoder = GraphSAGEEncoder(
                node_feature_dim=kspace_node_feature_dim,
                hidden_dim=kspace_gnn_hidden_channels,
                out_channels=latent_dim_gnn,
                n_layers=kspace_gnn_num_layers,
                num_heads=kspace_gnn_num_heads
            )
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=scalar_feature_dim,
            hidden_dims=ffnn_hidden_dims_scalar,
            out_channels=latent_dim_ffnn
        )
        self.decomposition_encoder = DecompositionFeatureEncoder(
            input_dim=decomposition_feature_dim,
            out_channels=latent_dim_ffnn
        )

        # Store dimensions for fusion
        self._crystal_dim = latent_dim_gnn
        self._kspace_dim = latent_dim_gnn
        self._scalar_dim = latent_dim_ffnn
        self._decomp_dim = latent_dim_ffnn
        
        # NEW: Attention mechanism
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.attention = None  # Will be created dynamically
        
        # NEW: Ensemble heads
        # self.ensemble_heads = []
        # self.num_ensemble_heads = num_ensemble_heads
        
        # NEW: Data augmentation parameters
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.feature_noise_std = feature_noise_std
        self.training = True

        # Fusion network will be created dynamically
        self.fusion_hidden_dims = fusion_hidden_dims
        self.fusion_network = None

        # Output heads
        self.topology_head = nn.Linear(1, num_topology_classes)  # Placeholder
        self.magnetism_head = nn.Linear(1, num_magnetism_classes)  # Placeholder

        # Gated fusion
        self.gated_fusion = GatedFusion(len(self.active_modalities))

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        features = []
        if 'crystal' in self.active_modalities:
            features.append(self.crystal_encoder(inputs['crystal_graph']))
        if 'kspace' in self.active_modalities:
            features.append(self.kspace_encoder(inputs['kspace_graph']))
        if 'scalar' in self.active_modalities:
            features.append(self.scalar_encoder(inputs['scalar_features']))
        if 'decomposition' in self.active_modalities:
            features.append(self.decomposition_encoder(inputs['kspace_physics_features']['decomposition_features']))
        # Gated fusion
        fused = self.gated_fusion(features)
        # Store for visualization
        self.last_fused = fused.detach().cpu()
        # Attention
        x_reshaped = fused.unsqueeze(1)
        x_attended = self.attention(x_reshaped)
        self.last_attention = x_attended.detach().cpu()
        x = x_attended.squeeze(1)
        
        # Feature normalization
        x = F.layer_norm(x, x.shape[1:])
        
        # Data augmentation
        if self.training:
            x = add_feature_noise(x, self.feature_noise_std)

        # Dynamically create fusion network if not exists
        if self.fusion_network is None:
            actual_input_dim = x.shape[1]
            print(f"Creating fusion network with input dimension: {actual_input_dim}")
            
            # Create attention mechanism
            self.attention = MultiHeadAttention(
                d_model=actual_input_dim,
                num_heads=self.attention_heads,
                dropout=self.attention_dropout
            ).to(x.device)
            
            # Create fusion MLP
            layers = []
            in_dim = actual_input_dim
            for h in self.fusion_hidden_dims:
                layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(config.DROPOUT_RATE)]
                in_dim = h
            self.fusion_network = nn.Sequential(*layers).to(x.device)
            
            # Comment out ensemble heads
            # self.ensemble_heads = []
            # for i in range(self.num_ensemble_heads):
            #     head = nn.Sequential(
            #         nn.Linear(in_dim, in_dim // 2),
            #         nn.ReLU(),
            #         nn.Dropout(0.2),
            #         nn.Linear(in_dim // 2, self.topology_head.out_features)
            #     ).to(x.device)
            #     self.ensemble_heads.append(head)
            
            # Update output heads
            self.topology_head = nn.Linear(in_dim, self.topology_head.out_features).to(x.device)
            self.magnetism_head = nn.Linear(in_dim, self.magnetism_head.out_features).to(x.device)

        # Apply attention mechanism
        x_reshaped = x.unsqueeze(1)  # Add sequence dimension for attention
        x_attended = self.attention(x_reshaped)
        x = x_attended.squeeze(1)  # Remove sequence dimension
        
        fused = self.fusion_network(x)
        
        # Gradient clipping
        fused = torch.clamp(fused, -10, 10)

        # Main heads
        topology_logits = self.topology_head(fused)
        magnetism_logits = self.magnetism_head(fused)
        
        # Comment out ensemble logic
        # ensemble_logits = []
        # for head in self.ensemble_heads:
        #     ensemble_logits.append(head(fused))
        # ensemble_logits = torch.stack(ensemble_logits)
        # ensemble_logits = torch.mean(ensemble_logits, dim=0)
        final_topology_logits = topology_logits

        return {
            'topology_logits': final_topology_logits,
            'magnetism_logits': magnetism_logits,
            'main_topology_logits': topology_logits,
            # 'ensemble_logits': ensemble_logits
        }

    def compute_loss(self, predictions: Dict[str, torch.Tensor], topology_targets: torch.Tensor, magnetism_targets: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_mixup:
            # Apply mixup to topology
            mixed_features, targets_a, targets_b, lam = mixup_data(
                predictions['topology_logits'], topology_targets, self.mixup_alpha
            )
            topology_loss = mixup_criterion(F.cross_entropy, mixed_features, targets_a, targets_b, lam)
            # Comment out ensemble loss
            # ensemble_loss = 0
            # for i in range(self.num_ensemble_heads):
            #     ensemble_loss += F.cross_entropy(predictions['ensemble_logits'], topology_targets, label_smoothing=0.1)
            # ensemble_loss /= self.num_ensemble_heads
            # Magnetism loss (no mixup)
            magnetism_loss = F.cross_entropy(predictions['magnetism_logits'], magnetism_targets, label_smoothing=0.1)
            return topology_loss + 0.5 * magnetism_loss
        else:
            topology_loss = F.cross_entropy(predictions['topology_logits'], topology_targets, label_smoothing=0.1)
            # Comment out ensemble loss
            # ensemble_loss = 0
            # for i in range(self.num_ensemble_heads):
            #     ensemble_loss += F.cross_entropy(predictions['ensemble_logits'], topology_targets, label_smoothing=0.1)
            # ensemble_loss /= self.num_ensemble_heads
            # Magnetism loss
            magnetism_loss = F.cross_entropy(predictions['magnetism_logits'], magnetism_targets, label_smoothing=0.1)
            return topology_loss + 0.5 * magnetism_loss