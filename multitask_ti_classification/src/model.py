import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, TransformerConv, MessagePassing 
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data as PyGData
from e3nn_jax import IrrepsArray
from e3nn import o3  # Keep for other parts if needed, but RealSpaceEGNNEncoder won't be used now
from e3nn.o3 import FullyConnectedTensorProduct, Linear, Irreps, Irrep
from e3nn.nn import Gate 
import torch_geometric
from typing import List, Dict, Any, Tuple, Optional
import warnings

import helper.config as config
from helper.kspace_physics_encoders import PhysicsInformedKSpaceEncoder, EnhancedKSpacePhysicsFeatures
from helper.topological_crystal_encoder import TopologicalCrystalEncoder

# --- 1. Crystal Graph Encoder (RealSpaceEGNN - now fixed for e3nn 0.5.6) ---

class EGNNLayer(nn.Module): # Reverted to standard e3nn.nn.Module and its expected logic
    """
    A simplified EGNN-like layer using e3nn components.
    Designed for e3nn 0.5.6+ where IrrepsArray and Gate work as expected.
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

        # --- Correct Gate initialization for messages ---
        tp_out_irreps_msg = self.tp_messages_ij.irreps_out
        
        scalars_msg = []
        gated_msg = []
        for mul, irrep in tp_out_irreps_msg:
            if irrep.l == 0:
                scalars_msg.append((mul, irrep))
            else:
                gated_msg.append((mul, irrep))
        
        irreps_scalars_msg = Irreps(scalars_msg) if scalars_msg else Irreps("0x0e")
        irreps_gated_msg = Irreps(gated_msg) if gated_msg else Irreps("0x0e")
        
        gates_msg = []
        for mul, irrep in irreps_gated_msg:
            if irrep.l > 0: 
                gates_msg.append((mul, o3.Irrep(0, 1))) 
        irreps_gates_msg = Irreps(gates_msg) if gates_msg else Irreps("0x0e")
        
        act_scalars_msg = [F.silu] * len(irreps_scalars_msg) if len(irreps_scalars_msg) > 0 else []
        act_gates_msg = [F.sigmoid] * len(irreps_gates_msg) if len(irreps_gates_msg) > 0 else []
        
        self.gate_messages = Gate(
            irreps_scalars=irreps_scalars_msg,
            act_scalars=act_scalars_msg,
            irreps_gates=irreps_gates_msg,
            act_gates=act_gates_msg,
            irreps_gated=irreps_gated_msg
        )
        self.linear_messages_out = Linear(self.gate_messages.irreps_out, hidden_irreps)

        # TP for update
        self.tp_update = FullyConnectedTensorProduct(
            node_irreps_in, hidden_irreps, hidden_irreps,
        )
        
        # --- Correct Gate initialization for update ---
        tp_out_irreps_update = self.tp_update.irreps_out
        
        scalars_update = []
        gated_update = []
        for mul, irrep in tp_out_irreps_update:
            if irrep.l == 0:
                scalars_update.append((mul, irrep))
            else:
                gated_update.append((mul, irrep))
        
        irreps_scalars_update = Irreps(scalars_update) if scalars_update else Irreps("0x0e")
        irreps_gated_update = Irreps(gated_update) if gated_update else Irreps("0x0e")
        
        gates_update = []
        for mul, irrep in irreps_gated_update:
            if irrep.l > 0:
                gates_update.append((mul, o3.Irrep(0, 1)))
        irreps_gates_update = Irreps(gates_update) if gates_update else Irreps("0x0e")
        
        act_scalars_update = [F.silu] * len(irreps_scalars_update) if len(irreps_scalars_update) > 0 else []
        act_gates_update = [F.sigmoid] * len(irreps_gates_update) if len(irreps_gates_update) > 0 else []
        
        self.gate_update = Gate(
            irreps_scalars=irreps_scalars_update,
            act_scalars=act_scalars_update,
            irreps_gates=irreps_gates_update,
            act_gates=act_gates_update,
            irreps_gated=irreps_gated_update
        )
        self.linear_update_out = Linear(self.gate_update.irreps_out, node_irreps_in)
        
    def forward(self, node_features: IrrepsArray, edge_index: torch.Tensor, edge_attr_e3nn: IrrepsArray) -> IrrepsArray:
        """
        Inputs and outputs are expected to be e3nn.IrrepsArray.
        """
        row, col = edge_index

        # 1. Message passing
        messages_tp_output = self.tp_messages_ij(node_features[col], edge_attr_e3nn)
        messages_gated = self.gate_messages(messages_tp_output) 
        messages_from_j = self.linear_messages_out(messages_gated)

        # 2. Aggregation (sum messages for each node)
        aggregated_messages = torch_geometric.utils.scatter(messages_from_j.array, row, dim=0, dim_size=node_features.size(0), reduce="sum")
        # Need to re-attach irreps after torch_geometric.utils.scatter
        aggregated_messages = IrrepsArray(self.hidden_irreps, aggregated_messages)


        # 3. Update
        updated_node_features_tp_output = self.tp_update(node_features, aggregated_messages)
        updated_node_features_gated = self.gate_update(updated_node_features_tp_output)
        updated_node_features_temp = self.linear_update_out(updated_node_features_gated)
            
        return node_features + updated_node_features_temp


class RealSpaceEGNNEncoder(nn.Module):
    """
    Real-space EGNN encoder. Uses idiomatic e3nn 0.5.6+ features.
    Explicitly ensures IrrepsArray inputs.
    """
    def __init__(self, 
                 node_input_scalar_dim: int,
                 hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
                 n_layers: int = 6,
                 radius: float = 5.0
                ):
        super().__init__()
        
        self.node_input_scalar_dim = node_input_scalar_dim
        self.radius = radius
        
        # Define irreps
        self.input_node_irreps = Irreps(f"{node_input_scalar_dim}x0e")
        self.edge_irreps = Irreps("1x1o + 1x0e")
        self.hidden_irreps = Irreps(hidden_irreps_str)
        
        # Initial projection using e3nn.o3.Linear
        self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EGNNLayer(
                node_irreps_in=self.hidden_irreps,
                edge_irreps_in=self.edge_irreps,
                hidden_irreps=self.hidden_irreps
            ))
        
        scalar_irreps_list = [(mul, irrep) for mul, irrep in self.hidden_irreps if irrep.l == 0]
        self.scalar_irreps = Irreps(scalar_irreps_list) if scalar_irreps_list else Irreps("0x0e")
        
        output_latent_dim = config.LATENT_DIM_GNN

        # Final projection
        self.final_projection = Linear(self.scalar_irreps, Irreps(f"{output_latent_dim}x0e"))
    
    def extract_scalar_features(self, x: IrrepsArray) -> torch.Tensor:
        """Extract scalar (l=0) features from an IrrepsArray."""
        if not hasattr(x, 'irreps') or not isinstance(x.irreps, Irreps):
            warnings.warn("extract_scalar_features: Input tensor has no valid irreps. Assuming full tensor is scalar and returning it.")
            return x 

        scalar_features = []
        start_idx = 0
        
        for mul, irrep in x.irreps:
            end_idx = start_idx + mul * irrep.dim
            if irrep.l == 0:  # scalar
                scalar_features.append(x[:, start_idx:end_idx])
            start_idx = end_idx
        
        if scalar_features:
            return torch.cat(scalar_features, dim=-1)
        else:
            warnings.warn("No scalar features found based on irreps. Returning zero tensor of expected scalar dim.")
            return torch.zeros(x.shape[0], self.scalar_irreps.dim, device=x.device)

    def forward(self, data: PyGData) -> torch.Tensor:
        x_raw_scalars = data.x
        pos = data.pos
        batch = data.batch
        
        # CRITICAL: Ensure initial input and edge attributes are IrrepsArray objects
        # This is where the irreps attribute is injected or validated.
        x_e3nn = IrrepsArray(self.input_node_irreps, x_raw_scalars)
        
        # Build radius graph
        edge_index = torch_geometric.nn.radius_graph(pos, self.radius, batch)
        
        if edge_index.size(1) == 0:
            batch_size = batch.max().item() + 1 if batch is not None else 1
            return torch.zeros(batch_size, config.LATENT_DIM_GNN, device=x_raw_scalars.device)
        
        row, col = edge_index
        
        # Compute edge attributes: relative vector and distance
        r_vec = pos[row] - pos[col]
        dist = r_vec.norm(dim=-1, keep_backdim=True) # Changed keepdim to keep_backdim for e3nn.o3.Linear compatibility
        normalized_r_vec = r_vec / (dist + 1e-8) 
        edge_attr_tensor = torch.cat([normalized_r_vec, dist], dim=-1)
        
        # CRITICAL: Ensure edge attributes are also IrrepsArray objects
        edge_attr_e3nn = IrrepsArray(self.edge_irreps, edge_attr_tensor)
        
        # Apply EGNN layers
        current_node_features_e3nn = x_e3nn
        for i, layer in enumerate(self.layers):
            current_node_features_e3nn = layer(
                current_node_features_e3nn, 
                edge_index, 
                edge_attr_e3nn 
            )
        
        # Extract scalar features for pooling
        scalar_features = self.extract_scalar_features(current_node_features_e3nn)
        
        # Global pooling
        pooled = global_mean_pool(scalar_features, batch)
        # pooled is a plain tensor from global_mean_pool. Re-attach scalar_irreps for final_projection
        pooled_e3nn = IrrepsArray(self.scalar_irreps, pooled)
        
        # Final projection
        output_e3nn = self.final_projection(pooled_e3nn)
        
        # Return as regular tensor (extract array from IrrepsArray)
        return output_e3nn.array

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

        current_in_channels = hidden_dim # Input channels for the first layer

        for i in range(n_layers):
            out_channels_per_head = hidden_dim # Keep per-head dimension consistent

            self.layers.append(TransformerConv(
                in_channels=current_in_channels, # Use dynamic in_channels
                out_channels=out_channels_per_head,
                heads=num_heads,
                dropout=config.DROPOUT_RATE,
                beta=True
            ))
            # BatchNorm1d should match total output features of TransformerConv (out_channels_per_head * num_heads)
            self.bns.append(nn.BatchNorm1d(out_channels_per_head * num_heads)) # Corrected
            
            # Update current_in_channels for the next layer
            current_in_channels = out_channels_per_head * num_heads

        # Add a final linear layer to project to the desired `out_channels` (LATENT_DIM_GNN)
        self.final_projection = nn.Linear(current_in_channels, out_channels)

    def forward(self, data: PyGData) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.initial_projection(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index) # Output of TransformerConv is (N_nodes, hidden_dim * num_heads)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)

        pooled_x = global_mean_pool(x, batch)
        final_embedding = self.final_projection(pooled_x) # Apply final projection
        
        return final_embedding

# --- 3. ASPH Encoder ---

class PHTokenEncoder(nn.Module):
    """
    Encoder for Atom-Specific Persistent Homology (ASPH) features.
    A simple FFNN to process the feature vector.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], out_channels: int): # Added hidden_dims parameter
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=config.DROPOUT_RATE))
            in_dim = h_dim
        
        # The final layer should map from the last hidden_dim to out_channels
        if in_dim != out_channels: # Only add if needed
             layers.append(nn.Linear(in_dim, out_channels))
             layers.append(nn.BatchNorm1d(out_channels))
             layers.append(nn.ReLU())
        elif hidden_dims: # If hidden_dims provided and last hidden is out_channels, just ensure consistency
            pass
        else: # No hidden_dims, direct input to out_channels
            layers.append(nn.Linear(in_dim, out_channels))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

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
    

class SimpleEGNNConv(MessagePassing):
    """Simplified EGNN convolution that's more reliable than full E3NN"""
    def __init__(self, in_channels, out_channels, edge_dim=4):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Message network
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Update network
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        # Add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Pad edge attributes for self loops
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        
        # Residual connection if dimensions match
        if x.size(-1) == out.size(-1):
            out = out + x
            
        return self.norm(out)
    
    def message(self, x_i, x_j, edge_attr):
        # Create message from source node, target node, and edge attributes
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(message_input)

class SimplifiedCrystalEncoder(nn.Module):
    """Simplified crystal encoder without E3NN complexity"""
    def __init__(self, 
                 node_feature_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 output_dim: int = 64,
                 radius: float = 5.0):
        super().__init__()
        
        self.radius = radius
        
        # Initial projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # EGNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(SimpleEGNNConv(hidden_dim, hidden_dim))
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Build radius graph
        edge_index = torch_geometric.nn.radius_graph(pos, self.radius, batch)
        
        if edge_index.size(1) == 0:
            # No edges - return zero embedding
            batch_size = batch.max().item() + 1 if batch is not None else 1
            return torch.zeros(batch_size, self.output_proj[-1].out_features, device=x.device)
        
        # Compute edge attributes
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)
        edge_dir = edge_vec / (edge_dist + 1e-8)
        edge_attr = torch.cat([edge_dir, edge_dist], dim=-1)
        
        # Apply EGNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final projection
        x = self.output_proj(x)
        
        return x

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
        kspace_gnn_num_heads: int = config.KSPACE_GNN_NUM_HEADS, # Using config value
        
        ffnn_hidden_dims_asph: List[int] = config.FFNN_HIDDEN_DIMS_ASPH, # Added as input param
        ffnn_hidden_dims_scalar: List[int] = config.FFNN_HIDDEN_DIMS_SCALAR,
        
        # Shared fusion params
        latent_dim_gnn: int = config.LATENT_DIM_GNN,
        latent_dim_asph: int = config.LATENT_DIM_ASPH,
        latent_dim_other_ffnn: int = config.LATENT_DIM_OTHER_FFNN,
        
        fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,

        crystal_encoder_hidden_dim: int = 256,
        crystal_encoder_num_layers: int = 6,
        crystal_encoder_output_dim: int = 128,
        crystal_encoder_radius: float = 8.0,
        crystal_encoder_num_scales: int = 3,
        crystal_encoder_use_topological_features: bool = True
    ):
        super().__init__()
        
        # self.crystal_encoder = SimplifiedCrystalEncoder(
        #     node_feature_dim=crystal_node_feature_dim, 
        #     hidden_dim=kspace_gnn_hidden_channels,
        #     num_layers=kspace_gnn_num_layers, 
        #     output_dim=latent_dim_gnn 
        # )
        
        self.crystal_encoder = TopologicalCrystalEncoder(
            node_feature_dim=crystal_node_feature_dim,
            hidden_dim=crystal_encoder_hidden_dim,
            num_layers=crystal_encoder_num_layers,
            output_dim=crystal_encoder_output_dim, 
            radius=crystal_encoder_radius,
            num_scales=crystal_encoder_num_scales,
            use_topological_features=crystal_encoder_use_topological_features
        )

        self.kspace_encoder = PhysicsInformedKSpaceEncoder(
            node_feature_dim=kspace_node_feature_dim,
            hidden_dim=kspace_gnn_hidden_channels,
            num_layers=kspace_gnn_num_layers, 
            output_dim=latent_dim_gnn 
        )
        self.asph_encoder = PHTokenEncoder(
            input_dim=asph_feature_dim,
            hidden_dims=ffnn_hidden_dims_asph, 
            out_channels=latent_dim_asph
        )
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=scalar_feature_dim,
            hidden_dims=ffnn_hidden_dims_scalar,
            out_channels=latent_dim_other_ffnn
        )
        self.decomposition_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=decomposition_feature_dim,
            gap_features_dim=config.BAND_GAP_SCALAR_DIM,
            dos_features_dim=config.DOS_FEATURE_DIM,
            fermi_features_dim=config.FERMI_FEATURE_DIM,
            output_dim=latent_dim_other_ffnn 
        )

       # total_fused_dim = (latent_dim_gnn * 2) + latent_dim_asph + (latent_dim_other_ffnn * 2)
        total_fused_dim = (
            crystal_encoder_output_dim +      # Output of crystal_encoder
            latent_dim_gnn +                  # Output of kspace_encoder
            latent_dim_asph +                 # Output of asph_encoder (if its out_channels is latent_dim_asph)
            latent_dim_other_ffnn +           # Output of scalar_encoder
            latent_dim_other_ffnn             # Output of decomposition_encoder
        ) 

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
        
        decomposition_emb = self.decomposition_encoder(
            decomposition_features=inputs['kspace_physics_features']['decomposition_features'],
            gap_features=inputs['kspace_physics_features'].get('gap_features'),
            dos_features=inputs['kspace_physics_features'].get('dos_features'),
            fermi_features=inputs['kspace_physics_features'].get('fermi_features')
        ) 
        
        combined_emb = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, decomposition_emb], dim=-1)

        fused_output = self.fusion_network(combined_emb)

        topology_logits = self.topology_head(fused_output)
        magnetism_logits = self.magnetism_head(fused_output)

        return {
            'topology_logits': topology_logits,
            'magnetism_logits': magnetism_logits
        }