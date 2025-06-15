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
            node_irreps_in, edge_irreps_in, hidden_irreps, # input_1, input_2, output
        )
        self.gate_messages = Gate(hidden_irreps)
        self.linear_messages_out = Linear(self.gate_messages.irreps_out, hidden_irreps)

        # TP for update (from node_i and aggregated_messages) -> new node irreps
        self.tp_update = FullyConnectedTensorProduct(
            node_irreps_in, hidden_irreps, hidden_irreps, # input_1, input_2, output
        )
        self.gate_update = Gate(hidden_irreps)
        self.linear_update_out = Linear(self.gate_update.irreps_out, node_irreps_in) # Output same as input for skip connection
        
    def forward(self, node_features: Irreps, edge_index: torch.Tensor, edge_attr_e3nn: Irreps, 
                node_attr_scalar_raw: torch.Tensor): # Type hint is Irreps, not IrrepsArray
        
        row, col = edge_index

        # 1. Message passing
        messages_from_j = self.linear_messages_out(self.gate_messages(
            self.tp_messages_ij(node_features[col], edge_attr_e3nn)
        ))

        # 2. Aggregation (sum messages for each node)
        aggregated_messages = torch_geometric.utils.scatter(messages_from_j.array, row, dim=0, dim_size=node_features.size(0), reduce="sum")
        # Need to wrap aggregated_messages back into Irreps, or ensure `scatter` returns IrrepsArray if it's e3nn's scatter
        # For PyG's scatter, you get a raw tensor, so you need to re-wrap it with its irreps
        aggregated_messages_e3nn = self.hidden_irreps(aggregated_messages) # Wrap back into IrrepsArray

        # 3. Update (combine current node features with aggregated messages)
        updated_node_features_temp = self.linear_update_out(self.gate_update(
            self.tp_update(node_features, aggregated_messages_e3nn) # Use the re-wrapped tensor
        ))
        
        # Residual connection: node_features and updated_node_features_temp are both IrrepsArray implicitly
        return node_features + updated_node_features_temp


class RealSpaceEGNNEncoder(nn.Module):
    """
    EGNN encoder for real-space atomic crystal graphs.
    Handles atomic features (Z, period, group) and spatial coordinates.
    """
    def __init__(self, 
                 node_input_scalar_dim: int,
                 hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
                 n_layers: int = 6,
                 radius: float = 5.0
                ):
        super().__init__()
        
        self.node_input_scalar_dim = node_input_scalar_dim
        
        self.input_node_irreps = Irreps(f"{node_input_scalar_dim}x0e")
        self.edge_irreps = Irreps("1x1o + 1x0e") 
        self.hidden_irreps = Irreps(hidden_irreps_str)
        
        self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EGNNLayer(
                node_irreps_in=self.hidden_irreps,
                edge_irreps_in=self.edge_irreps,
                hidden_irreps=self.hidden_irreps
            ))

        self.final_linear_0e = Linear(self.hidden_irreps.filter("0e"), config.LATENT_DIM_GNN)

        self.radius = radius

    def forward(self, data: PyGData) -> torch.Tensor: # PyG Data object
        x_raw_scalars = data.x # (N_atoms, node_input_scalar_dim) - original scalar features
        
        # 1. Project input scalar features to e3nn's Irreps structure (this is where the IrrepsArray is created implicitly)
        x_e3nn = self.initial_projection(x_raw_scalars) # x_e3nn is now an e3nn tensor (IrrepsArray)

        # 2. Recompute edge_index and edge_attr (relative positions)
        edge_index = torch_geometric.nn.radius_graph(data.pos, self.radius, data.batch)
        row, col = edge_index
        
        r_vec = data.pos[row] - data.pos[col]
        dist = r_vec.norm(dim=-1, keepdim=True)
        normalized_r_vec = r_vec / (dist + 1e-8) 
        
        # Create the e3nn edge attributes from raw tensors
        # `Irreps` object itself is callable to wrap a tensor.
        # Ensure the order matches self.edge_irreps = Irreps("1x1o + 1x0e") -> first 1o then 0e
        edge_attr_e3nn = self.edge_irreps(torch.cat([normalized_r_vec, dist], dim=-1))


        # 3. Pass through EGNN layers
        current_node_features_e3nn = x_e3nn
        for layer in self.layers:
            current_node_features_e3nn = layer(
                current_node_features_e3nn, 
                edge_index, 
                edge_attr_e3nn,
                x_raw_scalars # Pass raw scalar features if GNNLayer needs it (e.g. for CT-UAE)
            )

        # 4. Global Pooling: Extract INVARIANT (0e) part of node features
        invariant_features_per_node = current_node_features_e3nn.filter("0e")
        
        # Global mean pool on the underlying tensor array of these invariant features
        graph_embedding = global_mean_pool(invariant_features_per_node.array, data.batch) # .array extracts the torch.Tensor

        # 5. Final projection to LATENT_DIM_GNN
        final_embedding = self.final_linear_0e(graph_embedding)
        
        return final_embedding
    
# --- 2. K-space Graph Encoder (TransformerConv) ---

class KSpaceTransformerGNNEncoder(nn.Module):
    """
    Graph Transformer (TransformerConv) encoder for k-space topology graphs.
    """
    def __init__(self, node_feature_dim: int, hidden_dim: int, out_channels: int, 
                 n_layers: int = 4, num_heads: int = 8):
        super().__init__()
        # Initial projection of node features (e.g., k-point coords + irrep encodings)
        self.initial_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() # Batch normalization layers

        for i in range(n_layers):
            self.layers.append(TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads if (i < n_layers - 1) else hidden_dim, # Output dim per head
                heads=num_heads,
                dropout=config.DROPOUT_RATE,
                beta=True # Uses skip connection
            ))
            # BatchNorm after each TransformerConv layer
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
    def forward(self, data: PyGData) -> torch.Tensor:
        # data.x: node features, data.edge_index: graph connectivity, data.batch: batch assignment
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.initial_projection(x) # Project to hidden_dim
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerConv):
                x = layer(x, edge_index)
            else: # BatchNorm, ReLU
                x = layer(x)
            x = self.bns[i](x) # Apply BatchNorm
            x = F.relu(x) # Apply ReLU
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)

        # Global pooling to get a graph-level embedding
        return global_mean_pool(x, batch) # (B, out_channels)

# --- 3. ASPH Encoder ---

class PHTokenEncoder(nn.Module):
    """
    Encoder for Atom-Specific Persistent Homology (ASPH) features.
    A simple FFNN to process the feature vector.
    """
    def __init__(self, input_dim: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, out_channels * 2), # Expand a bit
            nn.BatchNorm1d(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(out_channels * 2, out_channels), # Project to final output dim
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features expected shape: (B, input_dim)
        return self.network(features) # (B, out_channels)

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
        layers.append(nn.BatchNorm1d(out_channels)) # BatchNorm for the final output
        layers.append(nn.ReLU()) # Final activation
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features expected shape: (B, input_dim)
        return self.network(features) # (B, out_channels)

# --- 5. Main Multi-Modal Classifier ---

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
        num_topology_classes: int,
        num_magnetism_classes: int,
        
        # Encoder specific params
        egnn_hidden_irreps_str: str = "64x0e + 32x1o + 16x2e", # Default for EGNN
        egnn_num_layers: int = 6,
        egnn_radius: float = 5.0, # Atomic interaction radius for EGNN
        
        kspace_gnn_hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers: int = config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads: int = 8,
        
        ffnn_hidden_dims_asph: List[int] = config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar: List[int] = config.FFNN_HIDDEN_DIMS_SCALAR,
        
        # Shared fusion params
        latent_dim_gnn: int = config.LATENT_DIM_GNN, # Output dim for both GNN encoders
        latent_dim_ffnn: int = config.LATENT_DIM_FFNN, # Output dim for both FFNN encoders
        fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,
    ):
        super().__init__()

        # Encoders for each modality
        self.crystal_encoder = RealSpaceEGNNEncoder(
            node_input_scalar_dim=crystal_node_feature_dim,
            hidden_irreps_str=egnn_hidden_irreps_str,
            n_layers=egnn_num_layers,
            radius=egnn_radius,
            # Note: The output dimension of RealSpaceEGNNEncoder (config.LATENT_DIM_GNN)
            # is controlled by its final_linear_0e layer.
        )
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=kspace_node_feature_dim,
            hidden_dim=kspace_gnn_hidden_channels,
            out_channels=latent_dim_gnn, # Output dimension for k-space GNN
            n_layers=kspace_gnn_num_layers,
            num_heads=kspace_gnn_num_heads
        )
        self.asph_encoder = PHTokenEncoder(
            input_dim=asph_feature_dim,
            out_channels=latent_dim_ffnn # Output dimension for ASPH FFNN
        )
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=scalar_feature_dim,
            hidden_dims=ffnn_hidden_dims_scalar,
            out_channels=latent_dim_ffnn # Output dimension for scalar FFNN
        )

        # Calculate the total dimension after concatenating all encoder outputs
        total_fused_dim = (latent_dim_gnn * 2) + (latent_dim_ffnn * 2)

        # Shared fusion layers (MLP)
        fusion_layers = []
        in_dim_fusion = total_fused_dim
        for h_dim in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(in_dim_fusion, h_dim))
            fusion_layers.append(nn.BatchNorm1d(h_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(p=config.DROPOUT_RATE))
            in_dim_fusion = h_dim 
        self.fusion_network = nn.Sequential(*fusion_layers)

        # Output heads for each task
        self.topology_head = nn.Linear(in_dim_fusion, num_topology_classes)
        self.magnetism_head = nn.Linear(in_dim_fusion, num_magnetism_classes)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Encode each modality
        crystal_emb = self.crystal_encoder(inputs['crystal_graph'])
        kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
        asph_emb = self.asph_encoder(inputs['asph_features'])
        scalar_emb = self.scalar_encoder(inputs['scalar_features'])

        # Concatenate all embeddings into a single comprehensive material embedding
        combined_emb = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb], dim=-1)

        # Pass through shared fusion layers
        fused_output = self.fusion_network(combined_emb)

        # Predict for each task using separate heads
        topology_logits = self.topology_head(fused_output)
        magnetism_logits = self.magnetism_head(fused_output)

        return {
            'topology_logits': topology_logits,
            'magnetism_logits': magnetism_logits
        }