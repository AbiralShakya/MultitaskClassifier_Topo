# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool, TransformerConv
# import torch_geometric
# from torch_geometric.data import Data as PyGData
# from e3nn import o3
# from e3nn.o3 import FullyConnectedTensorProduct, Linear, Irreps
# from e3nn.nn import Gate
# from typing import List, Dict, Any, Tuple

# import helper.config as config

# # --- 1. Crystal Graph Encoder (RealSpaceEGNN) ---

# class EGNNLayer(nn.Module):
#     """
#     A simplified EGNN-like layer using e3nn components.
#     This assumes `node_features` are e3nn.o3.Irreps-wrapped tensors and `edge_attr_e3nn` too.
#     """
#     def __init__(self, node_irreps_in: Irreps, edge_irreps_in: Irreps, hidden_irreps: Irreps):
#         super().__init__()
#         self.node_irreps_in = node_irreps_in
#         self.edge_irreps_in = edge_irreps_in
#         self.hidden_irreps = hidden_irreps

#         # TP for messages (from node_j and edge_attr) -> message irreps
#         self.tp_messages_ij = FullyConnectedTensorProduct(
#             node_irreps_in, edge_irreps_in, hidden_irreps,
#         )

#         # Direct linear path for messages (no tensor product)
#         self.linear_messages_direct = Linear(node_irreps_in, hidden_irreps)

#         # TP for update (from node_i and aggregated_messages) -> new node irreps
#         self.tp_update = FullyConnectedTensorProduct(
#             node_irreps_in, hidden_irreps, hidden_irreps,
#         )

#         # Direct linear path for update
#         self.linear_update_direct = Linear(node_irreps_in, hidden_irreps)

#         print(f"EGNNLayer init:")
#         print(f"  node_irreps_in: {node_irreps_in}")
#         print(f"  edge_irreps_in: {edge_irreps_in}")
#         print(f"  hidden_irreps: {hidden_irreps}")
#         print(f"  tp_messages_ij.irreps_out: {self.tp_messages_ij.irreps_out}")
#         print(f"  tp_update.irreps_out: {self.tp_update.irreps_out}")
        
#     def forward(self, node_features, edge_index: torch.Tensor, edge_attr_tensor: torch.Tensor, 
#                 node_attr_scalar_raw: torch.Tensor):
        
#         print(f"Forward pass debug:")
#         print(f"  node_features.shape: {node_features.shape}")
#         print(f"  edge_attr_tensor.shape: {edge_attr_tensor.shape}")
#         print(f"  edge_index.shape: {edge_index.shape}")
        
#         row, col = edge_index

#         # FIX 1: Properly convert edge attributes to e3nn format
#         # Extract components: [r_x, r_y, r_z, distance]
#         r_vec = edge_attr_tensor[:, :3]  # (num_edges, 3) - vector part (1o)
#         dist = edge_attr_tensor[:, 3:4]  # (num_edges, 1) - scalar part (0e)
        
#         # Create proper e3nn edge attributes: concatenate scalar + vector
#         edge_attr_e3nn = torch.cat([dist, r_vec], dim=-1)  # (num_edges, 4)
#         print(f"  edge_attr_e3nn.shape: {edge_attr_e3nn.shape}")

#         # 1. Message passing with both tensor product and direct paths
#         messages_tp_output = self.tp_messages_ij(node_features[col], edge_attr_e3nn)
#         print(f"  messages_tp_output.shape: {messages_tp_output.shape}")
        
#         messages_direct = self.linear_messages_direct(node_features[col])
#         messages_from_j = messages_tp_output + messages_direct
#         print(f"  messages_from_j.shape: {messages_from_j.shape}")

#         # 2. Aggregation (sum messages for each node)
#         aggregated_messages = torch_geometric.utils.scatter(
#             messages_from_j, row, dim=0, dim_size=node_features.size(0), reduce="sum"
#         )
#         print(f"  aggregated_messages.shape: {aggregated_messages.shape}")

#         # 3. Update with both tensor product and direct paths
#         updated_node_features_tp_output = self.tp_update(node_features, aggregated_messages)
#         print(f"  updated_node_features_tp_output.shape: {updated_node_features_tp_output.shape}")
        
#         updated_node_features_direct = self.linear_update_direct(node_features)
#         updated_node_features = updated_node_features_tp_output + updated_node_features_direct
#         print(f"  updated_node_features.shape: {updated_node_features.shape}")
        
#         # Residual connection
#         result = node_features + updated_node_features
#         print(f"  result.shape: {result.shape}")
        
#         return result


# class RealSpaceEGNNEncoder(nn.Module):
#     """
#     EGNN encoder for real-space atomic crystal graphs.
#     Handles atomic features (Z, period, group) and spatial coordinates.
#     """
#     def __init__(self, 
#                  node_input_scalar_dim: int,
#                  hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
#                  n_layers: int = 3,  # Reduced from 6 for debugging
#                  radius: float = 5.0
#                 ):
#         super().__init__()
        
#         self.node_input_scalar_dim = node_input_scalar_dim
        
#         self.input_node_irreps = Irreps(f"{node_input_scalar_dim}x0e")
#         # FIX 2: Correct edge irreps order - scalar first, then vector
#         self.edge_irreps = Irreps("1x0e + 1x1o")  # Changed order: scalar + vector
#         self.hidden_irreps = Irreps(hidden_irreps_str)
        
#         self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)

#         self.layers = nn.ModuleList()
#         for _ in range(n_layers):
#             self.layers.append(EGNNLayer(
#                 node_irreps_in=self.hidden_irreps,
#                 edge_irreps_in=self.edge_irreps,
#                 hidden_irreps=self.hidden_irreps
#             ))

#         # Extract only the scalar (0e) irreps from hidden_irreps for final projection
#         scalar_irreps_list = []
#         for mul, irrep in self.hidden_irreps:
#             if irrep.l == 0:  # scalar irrep
#                 scalar_irreps_list.append((mul, irrep))
        
#         if scalar_irreps_list:
#             self.scalar_irreps = Irreps(scalar_irreps_list)
#         else:
#             # Fallback if no scalars found
#             self.scalar_irreps = Irreps("1x0e")
            
#         output_irreps = Irreps(f"{config.LATENT_DIM_GNN}x0e")
#         self.final_linear_0e = Linear(self.scalar_irreps, output_irreps)

#         self.radius = radius

#     def forward(self, data: PyGData) -> torch.Tensor:
#         x_raw_scalars = data.x  # (N_atoms, node_input_scalar_dim)
        
#         # 1. Project input scalar features to e3nn's structure
#         x_e3nn = self.initial_projection(x_raw_scalars) 

#         # 2. Recompute edge_index and edge_attr (relative positions)
#         edge_index = torch_geometric.nn.radius_graph(data.pos, self.radius, data.batch)
#         row, col = edge_index
        
#         r_vec = data.pos[row] - data.pos[col]
#         dist = r_vec.norm(dim=-1, keepdim=True)
        
#         # FIX 3: Handle zero distances to avoid division by zero
#         normalized_r_vec = r_vec / (dist.clamp(min=1e-8))
        
#         # Create edge attributes tensor: [r_x, r_y, r_z, distance]
#         edge_attr_tensor = torch.cat([normalized_r_vec, dist], dim=-1)

#         # 3. Pass through EGNN layers
#         current_node_features_e3nn = x_e3nn
#         for layer in self.layers:
#             current_node_features_e3nn = layer(
#                 current_node_features_e3nn, 
#                 edge_index, 
#                 edge_attr_tensor,
#                 x_raw_scalars
#             )

#         # 4. Extract scalar features for global pooling
#         # FIX 4: Properly extract scalar features using irreps structure
#         scalar_features = []
#         start_idx = 0
#         for mul, irrep in self.hidden_irreps:
#             end_idx = start_idx + mul * irrep.dim
#             if irrep.l == 0:  # scalar irrep
#                 scalar_features.append(current_node_features_e3nn[:, start_idx:end_idx])
#             start_idx = end_idx
        
#         if scalar_features:
#             invariant_features_per_node = torch.cat(scalar_features, dim=1)
#         else:
#             # Fallback: use first 64 features as scalars
#             invariant_features_per_node = current_node_features_e3nn[:, :64]
        
#         # Global mean pool on the invariant features
#         graph_embedding_tensor = global_mean_pool(invariant_features_per_node, data.batch)
        
#         # 5. Final projection to LATENT_DIM_GNN
#         final_embedding = self.final_linear_0e(graph_embedding_tensor)
        
#         return final_embedding
    
# # --- 2. K-space Graph Encoder (TransformerConv) ---

# class KSpaceTransformerGNNEncoder(nn.Module):
#     """
#     Graph Transformer (TransformerConv) encoder for k-space topology graphs.
#     """
#     def __init__(self, node_feature_dim: int, hidden_dim: int, out_channels: int,
#                  n_layers: int = 3, num_heads: int = 8):
#         super().__init__()
#         # Initial projection takes node_feature_dim and outputs hidden_dim
#         self.initial_projection = nn.Linear(node_feature_dim, hidden_dim)

#         self.layers = nn.ModuleList()
#         self.bns = nn.ModuleList()

#         # The input to the first TransformerConv layer is hidden_dim
#         current_in_channels = hidden_dim

#         for i in range(n_layers):
#             # Output of TransformerConv will be hidden_dim * num_heads
#             transformer_out_channels = hidden_dim * num_heads

#             self.layers.append(TransformerConv(
#                 in_channels=current_in_channels,  # This will be `hidden_dim` for the first layer,
#                                                   # and `transformer_out_channels` for subsequent layers
#                 out_channels=hidden_dim,          # This is the 'per head' output dimension
#                 heads=num_heads,
#                 dropout=config.DROPOUT_RATE,
#                 beta=True
#             ))
#             self.bns.append(nn.BatchNorm1d(transformer_out_channels)) # Corrected BatchNorm

#             # Update current_in_channels for the next iteration
#             current_in_channels = transformer_out_channels

#     def forward(self, data: PyGData) -> torch.Tensor:
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = self.initial_projection(x) # Shape: (N, hidden_dim)

#         for i, layer in enumerate(self.layers):
#             # On first iteration, x is (N, hidden_dim) -> correct for TransformerConv(in_channels=hidden_dim, ...)
#             # On subsequent iterations, x is (N, hidden_dim * num_heads) -> now correct for TransformerConv(in_channels=hidden_dim * num_heads, ...)
#             x = layer(x, edge_index)
#             x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)

#         return global_mean_pool(x, batch)

# # --- 3. ASPH Encoder ---

# class PHTokenEncoder(nn.Module):
#     """
#     Encoder for Atom-Specific Persistent Homology (ASPH) features.
#     A simple FFNN to process the feature vector.
#     """
#     def __init__(self, input_dim: int, out_channels: int):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, out_channels * 2),
#             nn.BatchNorm1d(out_channels * 2),
#             nn.ReLU(),
#             nn.Dropout(p=config.DROPOUT_RATE),
#             nn.Linear(out_channels * 2, out_channels),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU()
#         )

#     def forward(self, features: torch.Tensor) -> torch.Tensor:
#         return self.network(features)

# # --- 4. Scalar Features Encoder ---

# class ScalarFeatureEncoder(nn.Module):
#     """
#     FFNN encoder for combined scalar features (band_rep_features + metadata features).
#     """
#     def __init__(self, input_dim: int, hidden_dims: List[int], out_channels: int):
#         super().__init__()
#         layers = []
#         in_dim = input_dim
#         for h_dim in hidden_dims:
#             layers.append(nn.Linear(in_dim, h_dim))
#             layers.append(nn.BatchNorm1d(h_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(p=config.DROPOUT_RATE))
#             in_dim = h_dim
        
#         layers.append(nn.Linear(in_dim, out_channels))
#         layers.append(nn.BatchNorm1d(out_channels))
#         layers.append(nn.ReLU())
#         self.network = nn.Sequential(*layers)

#     def forward(self, features: torch.Tensor) -> torch.Tensor:
#         return self.network(features)

# # --- 5. Decomposition Features Encoder ---

# class DecompositionFeatureEncoder(nn.Module):
#     """
#     FFNN encoder for decomposition branches features.
#     """
#     def __init__(self, input_dim: int, out_channels: int):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, out_channels * 2), 
#             nn.BatchNorm1d(out_channels * 2),
#             nn.ReLU(),
#             nn.Dropout(p=config.DROPOUT_RATE),
#             nn.Linear(out_channels * 2, out_channels),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU()
#         )

#     def forward(self, features: torch.Tensor) -> torch.Tensor:
#         return self.network(features)

# # --- 6. Main Multi-Modal Classifier ---

# class MultiModalMaterialClassifier(nn.Module):
#     """
#     Multi-modal, multi-task classifier for materials.
#     Combines Real-space EGNN, K-space Transformer GNN, ASPH, and Scalar features.
#     """
#     def __init__(
#         self,
#         crystal_node_feature_dim: int,
#         kspace_node_feature_dim: int,
#         asph_feature_dim: int,
#         scalar_feature_dim: int,
#         decomposition_feature_dim: int,
#         num_topology_classes: int,
#         num_magnetism_classes: int,
        
#         # Encoder specific params
#         egnn_hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
#         egnn_num_layers: int = 3,  # Reduced for debugging
#         egnn_radius: float = 5.0,
        
#         kspace_gnn_hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
#         kspace_gnn_num_layers: int = 3,  # Reduced for debugging
#         kspace_gnn_num_heads: int = 8,
        
#         ffnn_hidden_dims_asph: List[int] = config.FFNN_HIDDEN_DIMS_ASPH,
#         ffnn_hidden_dims_scalar: List[int] = config.FFNN_HIDDEN_DIMS_SCALAR,
        
#         # Shared fusion params
#         latent_dim_gnn: int = config.LATENT_DIM_GNN,
#         latent_dim_ffnn: int = config.LATENT_DIM_FFNN,
#         fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,
#     ):
#         super().__init__()

#         self.crystal_encoder = RealSpaceEGNNEncoder(
#             node_input_scalar_dim=crystal_node_feature_dim,
#             hidden_irreps_str=egnn_hidden_irreps_str,
#             n_layers=egnn_num_layers,
#             radius=egnn_radius,
#         )
#         self.kspace_encoder = KSpaceTransformerGNNEncoder(
#             node_feature_dim=kspace_node_feature_dim,
#             hidden_dim=kspace_gnn_hidden_channels,
#             out_channels=latent_dim_gnn,
#             n_layers=kspace_gnn_num_layers,
#             num_heads=kspace_gnn_num_heads
#         )
#         self.asph_encoder = PHTokenEncoder(
#             input_dim=asph_feature_dim,
#             out_channels=latent_dim_ffnn 
#         )
#         self.scalar_encoder = ScalarFeatureEncoder(
#             input_dim=scalar_feature_dim,
#             hidden_dims=ffnn_hidden_dims_scalar,
#             out_channels=latent_dim_ffnn
#         )
#         self.decomposition_encoder = DecompositionFeatureEncoder(
#             input_dim=decomposition_feature_dim,
#             out_channels=latent_dim_ffnn
#         )

#         total_fused_dim = (latent_dim_gnn * 2) + (latent_dim_ffnn * 3) 

#         fusion_layers = []
#         in_dim_fusion = total_fused_dim
#         for h_dim in fusion_hidden_dims:
#             fusion_layers.append(nn.Linear(in_dim_fusion, h_dim))
#             fusion_layers.append(nn.BatchNorm1d(h_dim))
#             fusion_layers.append(nn.ReLU())
#             fusion_layers.append(nn.Dropout(p=config.DROPOUT_RATE))
#             in_dim_fusion = h_dim 
#         self.fusion_network = nn.Sequential(*fusion_layers)

#         self.topology_head = nn.Linear(in_dim_fusion, num_topology_classes)
#         self.magnetism_head = nn.Linear(in_dim_fusion, num_magnetism_classes)

#     def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
#         crystal_emb = self.crystal_encoder(inputs['crystal_graph'])
#         kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
#         asph_emb = self.asph_encoder(inputs['asph_features'])
#         scalar_emb = self.scalar_encoder(inputs['scalar_features'])
#         decomposition_emb = self.decomposition_encoder(inputs['kspace_physics_features']['decomposition_features']) 

#         combined_emb = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, decomposition_emb], dim=-1)

#         fused_output = self.fusion_network(combined_emb)

#         topology_logits = self.topology_head(fused_output)
#         magnetism_logits = self.magnetism_head(fused_output)

#         return {
#             'topology_logits': topology_logits,
#             'magnetism_logits': magnetism_logits
#         }


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

import helper.config as config

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

        print(f"EGNNLayer init:")
        print(f"  node_irreps_in: {node_irreps_in}")
        print(f"  edge_irreps_in: {edge_irreps_in}")
        print(f"  hidden_irreps: {hidden_irreps}")
        print(f"  tp_messages_ij.irreps_out: {self.tp_messages_ij.irreps_out}")
        print(f"  tp_update.irreps_out: {self.tp_update.irreps_out}")
        
    def forward(self, node_features, edge_index: torch.Tensor, edge_attr_tensor: torch.Tensor, 
                node_attr_scalar_raw: torch.Tensor):
        
        print(f"Forward pass debug:")
        print(f"  node_features.shape: {node_features.shape}")
        print(f"  edge_attr_tensor.shape: {edge_attr_tensor.shape}")
        print(f"  edge_index.shape: {edge_index.shape}")
        
        row, col = edge_index

        # FIX 1: Properly convert edge attributes to e3nn format
        # Extract components: [r_x, r_y, r_z, distance]
        r_vec = edge_attr_tensor[:, :3]  # (num_edges, 3) - vector part (1o)
        dist = edge_attr_tensor[:, 3:4]  # (num_edges, 1) - scalar part (0e)
        
        # Create proper e3nn edge attributes: concatenate scalar + vector
        edge_attr_e3nn = torch.cat([dist, r_vec], dim=-1)  # (num_edges, 4)
        print(f"  edge_attr_e3nn.shape: {edge_attr_e3nn.shape}")

        # 1. Message passing with both tensor product and direct paths
        messages_tp_output = self.tp_messages_ij(node_features[col], edge_attr_e3nn)
        print(f"  messages_tp_output.shape: {messages_tp_output.shape}")
        
        messages_direct = self.linear_messages_direct(node_features[col])
        messages_from_j = messages_tp_output + messages_direct
        print(f"  messages_from_j.shape: {messages_from_j.shape}")

        # 2. Aggregation (sum messages for each node)
        aggregated_messages = torch_geometric.utils.scatter(
            messages_from_j, row, dim=0, dim_size=node_features.size(0), reduce="sum"
        )
        print(f"  aggregated_messages.shape: {aggregated_messages.shape}")

        # 3. Update with both tensor product and direct paths
        updated_node_features_tp_output = self.tp_update(node_features, aggregated_messages)
        print(f"  updated_node_features_tp_output.shape: {updated_node_features_tp_output.shape}")
        
        updated_node_features_direct = self.linear_update_direct(node_features)
        updated_node_features = updated_node_features_tp_output + updated_node_features_direct
        print(f"  updated_node_features.shape: {updated_node_features.shape}")
        
        # Residual connection
        result = node_features + updated_node_features
        print(f"  result.shape: {result.shape}")
        
        return result


class RealSpaceEGNNEncoder(nn.Module):
    """
    EGNN encoder for real-space atomic crystal graphs.
    Handles atomic features (Z, period, group) and spatial coordinates.
    """
    def __init__(self, 
                 node_input_scalar_dim: int,
                 hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
                 n_layers: int = 3,  # Reduced from 6 for debugging
                 radius: float = 5.0
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
    Graph Transformer (TransformerConv) encoder for k-space topology graphs.
    """
    def __init__(self, node_feature_dim: int, hidden_dim: int, out_channels: int, # out_channels is LATENT_DIM_GNN
                 n_layers: int = 3, num_heads: int = 8):
        super().__init__()
        self.initial_projection = nn.Linear(node_feature_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        current_in_channels = hidden_dim # Input to the first TransformerConv layer

        for i in range(n_layers):
            transformer_out_channels = hidden_dim * num_heads

            self.layers.append(TransformerConv(
                in_channels=current_in_channels,
                out_channels=hidden_dim, # This is 'per head' output dimension
                heads=num_heads,
                dropout=config.DROPOUT_RATE,
                beta=True
            ))
            self.bns.append(nn.BatchNorm1d(transformer_out_channels))

            current_in_channels = transformer_out_channels # Update for next layer

        # FIX: Add a final linear layer to project to out_channels (LATENT_DIM_GNN)
        self.final_projection = nn.Linear(current_in_channels, out_channels)


    def forward(self, data: PyGData) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.initial_projection(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)

        # Global mean pool first
        pooled_x = global_mean_pool(x, batch) # Shape: (batch_size, current_in_channels) after loop

        # FIX: Project pooled features to the desired output dimension (LATENT_DIM_GNN)
        final_embedding = self.final_projection(pooled_x)

        return final_embedding
# --- 3. ASPH Encoder ---

class PHTokenEncoder(nn.Module):
    """
    Encoder for Atom-Specific Persistent Homology (ASPH) features.
    A simple FFNN to process the feature vector.
    """
    def __init__(self, input_dim: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, out_channels * 2),
            nn.BatchNorm1d(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

# --- 4. Scalar Features Encoder ---

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
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=config.DROPOUT_RATE))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, out_channels))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

# --- 5. Decomposition Features Encoder ---

class DecompositionFeatureEncoder(nn.Module):
    """
    FFNN encoder for decomposition branches features.
    """
    def __init__(self, input_dim: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, out_channels * 2), 
            nn.BatchNorm1d(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

# --- 6. Main Multi-Modal Classifier ---

class MultiModalMaterialClassifier(nn.Module):
    """
    Multi-modal, multi-task classifier for materials.
    Combines Real-space EGNN, K-space Transformer GNN, ASPH, and Scalar features.
    """
    def __init__(
        self,
        crystal_node_feature_dim: int,
        kspace_node_feature_dim: int,
        asph_feature_dim: int,
        scalar_feature_dim: int,
        decomposition_feature_dim: int,
        num_topology_classes: int,
        num_magnetism_classes: int,
        
        # Encoder specific params
        egnn_hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
        egnn_num_layers: int = 3,  # Reduced for debugging
        egnn_radius: float = 5.0,
        
        kspace_gnn_hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers: int = 3,  # Reduced for debugging
        kspace_gnn_num_heads: int = 8,
        
        ffnn_hidden_dims_asph: List[int] = config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar: List[int] = config.FFNN_HIDDEN_DIMS_SCALAR,
        
        # Shared fusion params
        latent_dim_gnn: int = config.LATENT_DIM_GNN,
        latent_dim_ffnn: int = config.LATENT_DIM_FFNN,
        fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,
    ):
        super().__init__()

        self.crystal_encoder = RealSpaceEGNNEncoder(
            node_input_scalar_dim=crystal_node_feature_dim,
            hidden_irreps_str=egnn_hidden_irreps_str,
            n_layers=egnn_num_layers,
            radius=egnn_radius,
        )
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=kspace_node_feature_dim,
            hidden_dim=kspace_gnn_hidden_channels,
            out_channels=latent_dim_gnn,
            n_layers=kspace_gnn_num_layers,
            num_heads=kspace_gnn_num_heads
        )
        self.asph_encoder = PHTokenEncoder(
            input_dim=asph_feature_dim,
            out_channels=latent_dim_ffnn 
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

        total_fused_dim = (latent_dim_gnn * 2) + (latent_dim_ffnn * 3) 

        fusion_layers = []
        in_dim_fusion = total_fused_dim
        for h_dim in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(in_dim_fusion, h_dim))
            fusion_layers.append(nn.BatchNorm1d(h_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(p=config.DROPOUT_RATE))
            in_dim_fusion = h_dim 
        self.fusion_network = nn.Sequential(*fusion_layers)

        self.topology_head = nn.Linear(in_dim_fusion, num_topology_classes)
        self.magnetism_head = nn.Linear(in_dim_fusion, num_magnetism_classes)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        crystal_emb = self.crystal_encoder(inputs['crystal_graph'])
        print(f"DEBUG: crystal_emb shape: {crystal_emb.shape}") # Add this
        kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
        print(f"DEBUG: kspace_emb shape: {kspace_emb.shape}") # Add this
        asph_emb = self.asph_encoder(inputs['asph_features'])
        print(f"DEBUG: asph_emb shape: {asph_emb.shape}") # Add this
        scalar_emb = self.scalar_encoder(inputs['scalar_features'])
        print(f"DEBUG: scalar_emb shape: {scalar_emb.shape}") # Add this
        decomposition_emb = self.decomposition_encoder(inputs['kspace_physics_features']['decomposition_features'])
        print(f"DEBUG: decomposition_emb shape: {decomposition_emb.shape}") # Add this

        combined_emb = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, decomposition_emb], dim=-1)
        print(f"DEBUG: combined_emb shape before fusion: {combined_emb.shape}") # Add this

        fused_output = self.fusion_network(combined_emb)

        topology_logits = self.topology_head(fused_output)
        magnetism_logits = self.magnetism_head(fused_output)

        return {
            'topology_logits': topology_logits,
            'magnetism_logits': magnetism_logits
        }