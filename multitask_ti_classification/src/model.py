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

        # --- FIX: Create irreps components as strings, not tuples ---
        # 1. Separate scalars (l=0) from non-scalars (l>0)
        irreps_scalars_parts = []
        for mul, irrep_obj in hidden_irreps:
            if irrep_obj.l == 0:
                # Convert numerical parity (irrep_obj.p) to string 'e' or 'o'
                parity_str = 'e' if irrep_obj.p == 1 else 'o'
                irreps_scalars_parts.append(f"{mul}x{irrep_obj.l}{parity_str}")
        
        # Join with " + " to create a valid irreps string
        irreps_scalars = Irreps(" + ".join(irreps_scalars_parts)) if irreps_scalars_parts else Irreps("")

        irreps_gated_parts = []
        for mul, irrep_obj in hidden_irreps:
            if irrep_obj.l > 0:
                # Convert numerical parity (irrep_obj.p) to string 'e' or 'o'
                parity_str = 'e' if irrep_obj.p == 1 else 'o'
                irreps_gated_parts.append(f"{mul}x{irrep_obj.l}{parity_str}")
        
        irreps_gated = Irreps(" + ".join(irreps_gated_parts)) if irreps_gated_parts else Irreps("")
        
        # 2. irreps_gates: A scalar (0e) for each component in irreps_gated.
        irreps_gates_parts = []
        for mul, irrep_obj in irreps_gated: # Iterate over the (mul, Irrep_object) pairs for gated parts
            irreps_gates_parts.append(f"{mul}x0e") # Each gated mul-l irrep gets mul-0e gate
        
        irreps_gates = Irreps(" + ".join(irreps_gates_parts)) if irreps_gates_parts else Irreps("")

        self.gate_messages = Gate(
            irreps_in=hidden_irreps,    # The full input irreps to the gate
            act_scalars=F.silu,         # Activation for the scalar (0e) part
            irreps_gates=irreps_gates,  # The scalar gates themselves
            act_gates=F.sigmoid,        # Activation for the gates (often sigmoid for scaling)
            irreps_gated=irreps_gated   # The non-scalar (l>0) parts to be gated
        )
        self.linear_messages_out = Linear(self.gate_messages.irreps_out, hidden_irreps)

        # TP for update (from node_i and aggregated_messages) -> new node irreps
        self.tp_update = FullyConnectedTensorProduct(
            node_irreps_in, hidden_irreps, hidden_irreps, # input_1, input_2, output
        )
        
        # --- FIX: Correct Gate initialization for update gate ---
        # Reuse the same derived irreps for the update gate
        self.gate_update = Gate(
            irreps_in=hidden_irreps,
            act_scalars=F.silu,
            irreps_gates=irreps_gates,
            act_gates=F.sigmoid,
            irreps_gated=irreps_gated
        )
        self.linear_update_out = Linear(self.gate_update.irreps_out, node_irreps_in) # Output same as input for skip connection
        
    def forward(self, node_features: Irreps, edge_index: torch.Tensor, edge_attr_e3nn: Irreps, 
                node_attr_scalar_raw: torch.Tensor):
        
        row, col = edge_index

        # 1. Message passing
        messages_tp_output = self.tp_messages_ij(node_features[col], edge_attr_e3nn)
        messages_gated = self.gate_messages(messages_tp_output)
        messages_from_j = self.linear_messages_out(messages_gated)

        # 2. Aggregation (sum messages for each node)
        aggregated_messages = torch_geometric.utils.scatter(messages_from_j.array, row, dim=0, dim_size=node_features.size(0), reduce="sum")
        aggregated_messages_e3nn = self.hidden_irreps(aggregated_messages)

        # 3. Update (combine current node features with aggregated messages)
        updated_node_features_tp_output = self.tp_update(node_features, aggregated_messages_e3nn)
        updated_node_features_gated = self.gate_update(updated_node_features_tp_output)
        updated_node_features_temp = self.linear_update_out(updated_node_features_gated)
        
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
        
        # 1. Project input scalar features to e3nn's Irreps structure
        x_e3nn = self.initial_projection(x_raw_scalars) 

        # 2. Recompute edge_index and edge_attr (relative positions)
        edge_index = torch_geometric.nn.radius_graph(data.pos, self.radius, data.batch)
        row, col = edge_index
        
        r_vec = data.pos[row] - data.pos[col]
        dist = r_vec.norm(dim=-1, keepdim=True)
        normalized_r_vec = r_vec / (dist + 1e-8) 
        
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
        graph_embedding = global_mean_pool(invariant_features_per_node.array, data.batch)

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
        self.initial_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=config.DROPOUT_RATE,
                beta=True
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
    def forward(self, data: PyGData) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.initial_projection(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index) 
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)

        return global_mean_pool(x, batch)

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
        decomposition_feature_dim: int,
        num_topology_classes: int,
        num_magnetism_classes: int,
        
        # Encoder specific params
        egnn_hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
        egnn_num_layers: int = 6,
        egnn_radius: float = 5.0,
        
        kspace_gnn_hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers: int = config.GNN_NUM_LAYERS,
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
        kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
        asph_emb = self.asph_encoder(inputs['asph_features'])
        scalar_emb = self.scalar_encoder(inputs['scalar_features'])
        decomposition_emb = self.decomposition_encoder(inputs['kspace_physics_features']['decomposition_features']) 

        combined_emb = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, decomposition_emb], dim=-1)

        fused_output = self.fusion_network(combined_emb)

        topology_logits = self.topology_head(fused_output)
        magnetism_logits = self.magnetism_head(fused_output)

        return {
            'topology_logits': topology_logits,
            'magnetism_logits': magnetism_logits
        }
    

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