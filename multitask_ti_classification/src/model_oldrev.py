import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, TransformerConv, MessagePassing 
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data as PyGData
from e3nn import o3
from e3nn_jax import IrrepsArray
from e3nn.o3 import FullyConnectedTensorProduct, Linear, Irreps, Irrep
from e3nn.nn import Gate 
from typing import List, Dict, Any, Tuple, Optional
import warnings
import torch_geometric

import helper.config as config
from helper.kspace_physics_encoders import PhysicsInformedKSpaceEncoder, EnhancedKSpacePhysicsFeatures

# --- 1. Crystal Graph Encoder (RealSpaceEGNN - now fixed for e3nn 0.5.6) ---

class EGNNLayer(nn.Module):
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
        irreps_gated_msg = Irreps(gated_msg) if gated_msg else Irreps("0x0e") # Use 0x0e for empty non-scalars
        
        gates_msg = []
        for mul, irrep in irreps_gated_msg:
            if irrep.l > 0: # Only create scalar gates for non-scalar components
                gates_msg.append((mul, o3.Irrep(0, 1))) # 0e gates
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
        
        # Define irreps for input, hidden features, and edge attributes
        self.input_node_irreps = Irreps(f"{node_input_scalar_dim}x0e")
        self.edge_irreps = Irreps("1x1o + 1x0e")  # normalized vector + distance
        self.hidden_irreps = Irreps(hidden_irreps_str)
        
        # Initial projection using e3nn.o3.Linear
        self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)
        
        # EGNN layers (now use the fixed EGNNLayer above)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EGNNLayer(
                node_irreps_in=self.hidden_irreps,
                edge_irreps_in=self.edge_irreps,
                hidden_irreps=self.hidden_irreps
            ))
        
        # Scalar irreps for final output (only l=0 components)
        scalar_irreps_list = [(mul, irrep) for mul, irrep in self.hidden_irreps if irrep.l == 0]
        self.scalar_irreps = Irreps(scalar_irreps_list) if scalar_irreps_list else Irreps("0x0e")
        
        output_latent_dim = config.LATENT_DIM_GNN

        # Final projection using e3nn.o3.Linear
        self.final_projection = Linear(self.scalar_irreps, Irreps(f"{output_latent_dim}x0e"))

    def extract_scalar_features(self, x: IrrepsArray) -> torch.Tensor:
        """
        Extract scalar (l=0) features from an IrrepsArray.
        """
        if not hasattr(x, 'irreps') or not isinstance(x.irreps, Irreps):
            warnings.warn("extract_scalar_features: Input tensor has no valid irreps. Assuming full tensor is scalar and returning it.")
            return x # Should not happen if previous steps ensure irreps

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
        dist = r_vec.norm(dim=-1, keepdim=True)
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
                edge_attr_e3nn # Pass the IrrepsArray edge attributes
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
                 node_input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 output_dim: int = 64,
                 radius: float = 5.0):
        super().__init__()
        
        self.radius = radius
        
        # Initial projection
        self.input_proj = nn.Linear(node_input_dim, hidden_dim)
        
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
    

class E3NNTensorWrapper:
    """
    Wrapper to ensure tensors have proper irreps attribute for e3nn operations.
    This solves the batching issue where irreps get lost.
    """
    @staticmethod
    def wrap_tensor(tensor: torch.Tensor, irreps: Irreps) -> torch.Tensor:
        """Attach irreps to tensor if not present"""
        if not hasattr(tensor, 'irreps'):
            # Create a new tensor with irreps attribute
            wrapped = tensor.clone()
            wrapped.irreps = irreps
            return wrapped
        return tensor
    
    @staticmethod
    def ensure_irreps(tensor: torch.Tensor, expected_irreps: Irreps) -> torch.Tensor:
        """Ensure tensor has the expected irreps"""
        if not hasattr(tensor, 'irreps'):
            tensor.irreps = expected_irreps
        elif tensor.irreps != expected_irreps:
            print(f"Warning: tensor irreps {tensor.irreps} != expected {expected_irreps}")
        return tensor

class RobustGate(nn.Module):
    """
    Robust gate implementation that handles e3nn Gate initialization issues
    """
    def __init__(self, irreps_in: Irreps, activation_scalars=F.silu, activation_gates=F.sigmoid):
        super().__init__()
        self.irreps_in = irreps_in
        
        # Separate scalars and non-scalars
        self.scalar_irreps = Irreps([(mul, irrep) for mul, irrep in irreps_in if irrep.l == 0])
        self.nonscalar_irreps = Irreps([(mul, irrep) for mul, irrep in irreps_in if irrep.l > 0])
        
        # Create gate irreps (one scalar gate per non-scalar irrep multiplicity)
        gate_irreps_list = []
        for mul, irrep in self.nonscalar_irreps:
            if irrep.l > 0:
                gate_irreps_list.append((mul, o3.Irrep(0, 1)))  # scalar gates
        self.gate_irreps = Irreps(gate_irreps_list) if gate_irreps_list else Irreps("0x0e")
        
        # Output irreps (scalars + gated non-scalars)
        self.irreps_out = self.scalar_irreps + self.nonscalar_irreps
        
        # Try to use e3nn Gate, fallback to manual implementation
        self.use_e3nn_gate = True
        try:
            if len(self.scalar_irreps) > 0 and len(self.nonscalar_irreps) > 0:
                self.gate = Gate(
                    irreps_scalars=self.scalar_irreps,
                    act_scalars=[activation_scalars] * len(self.scalar_irreps),
                    irreps_gates=self.gate_irreps,
                    act_gates=[activation_gates] * len(self.gate_irreps),
                    irreps_gated=self.nonscalar_irreps
                )
            elif len(self.scalar_irreps) > 0:
                # Only scalars, no gating needed
                self.scalar_activation = activation_scalars
                self.use_e3nn_gate = False
            else:
                # Only non-scalars, create simple gates
                self.gate_linear = nn.Linear(self.gate_irreps.dim, self.nonscalar_irreps.dim)
                self.gate_activation = activation_gates
                self.use_e3nn_gate = False
        except Exception as e:
            print(f"Warning: e3nn Gate failed, using manual implementation: {e}")
            self.use_e3nn_gate = False
            self.scalar_activation = activation_scalars
            if len(self.nonscalar_irreps) > 0:
                self.gate_linear = nn.Linear(self.gate_irreps.dim if self.gate_irreps.dim > 0 else 1, 
                                           self.nonscalar_irreps.dim)
                self.gate_activation = activation_gates
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_e3nn_gate and hasattr(self, 'gate'):
            result = self.gate(x)
            return E3NNTensorWrapper.ensure_irreps(result, self.irreps_out)
        
        # Manual gate implementation
        outputs = []
        start_idx = 0
        
        # Process scalars
        for mul, irrep in self.scalar_irreps:
            end_idx = start_idx + mul * irrep.dim
            scalar_part = x[:, start_idx:end_idx]
            if hasattr(self, 'scalar_activation'):
                scalar_part = self.scalar_activation(scalar_part)
            outputs.append(scalar_part)
            start_idx = end_idx
        
        # Process non-scalars with gating
        if len(self.nonscalar_irreps) > 0:
            gate_start = start_idx
            for mul, irrep in self.nonscalar_irreps:
                end_idx = start_idx + mul * irrep.dim
                nonscalar_part = x[:, start_idx:end_idx]
                
                if hasattr(self, 'gate_linear'):
                    # Create gates from the non-scalar features themselves (simplified)
                    gate_input = nonscalar_part.mean(dim=-1, keepdim=True)
                    gates = self.gate_activation(self.gate_linear(gate_input))
                    nonscalar_part = nonscalar_part * gates
                
                outputs.append(nonscalar_part)
                start_idx = end_idx
        
        result = torch.cat(outputs, dim=-1) if outputs else x
        return E3NNTensorWrapper.ensure_irreps(result, self.irreps_out)

class RobustEGNNLayer(nn.Module):
    """
    Robust EGNN layer with proper IrrepsArray wrapping and TorchScript-friendly fallbacks.
    """
    def __init__(self, node_irreps_in: Irreps, edge_irreps_in: Irreps, hidden_irreps: Irreps):
        super().__init__()
        self.node_irreps_in = node_irreps_in
        self.edge_irreps_in = edge_irreps_in
        self.hidden_irreps = hidden_irreps

        # Message path
        self.message_tp = FullyConnectedTensorProduct(node_irreps_in, edge_irreps_in, hidden_irreps)
        self.message_gate = RobustGate(self.message_tp.irreps_out)
        self.message_linear = (
            Linear(self.message_gate.irreps_out, hidden_irreps)
            if self.message_gate.irreps_out != hidden_irreps else nn.Identity()
        )

        # Update path
        self.update_tp = FullyConnectedTensorProduct(node_irreps_in, hidden_irreps, hidden_irreps)
        self.update_gate = RobustGate(self.update_tp.irreps_out)
        self.update_linear = (
            Linear(self.update_gate.irreps_out, node_irreps_in)
            if self.update_gate.irreps_out != node_irreps_in else nn.Identity()
        )

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # 1) Wrap and assert shapes
        x = IrrepsArray(self.node_irreps_in, node_features)
        e = IrrepsArray(self.edge_irreps_in, edge_attr)

        assert x.shape[-1] == self.node_irreps_in.dim, (
            f"node_features dim={x.shape[-1]}, expected={self.node_irreps_in.dim}"
        )
        assert e.shape[-1] == self.edge_irreps_in.dim, (
            f"edge_attr dim={e.shape[-1]}, expected={self.edge_irreps_in.dim}"
        )

        row, col = edge_index

        # 2) Message passing
        try:
            m = self.message_tp(x[col], e)
            m = self.message_gate(m)
            m = self.message_linear(m)
        except Exception:
            # fallback only in Python; TorchScript will ignore this entirely
            m = self.message_fallback(x[col].array, e.array)

        # aggregate
        m_agg = torch_geometric.utils.scatter(
            m.array if isinstance(m, IrrepsArray) else m,
            row, dim=0, dim_size=x.shape[0], reduce="sum"
        )
        m_agg = IrrepsArray(self.hidden_irreps, m_agg)

        # 3) Update
        try:
            u = self.update_tp(x, m_agg)
            u = self.update_gate(u)
            u = self.update_linear(u)
        except Exception:
            u = self.update_fallback(x.array, m_agg.array)

        # 4) Residual & return
        out = x + (u if isinstance(u, IrrepsArray) else IrrepsArray(self.node_irreps_in, u))
        return out.array

    @torch.jit.unused
    def message_fallback(self, x_col: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # Simple linear fallback
        combined = torch.cat([x_col, e], dim=-1)
        fallback = nn.Linear(combined.shape[-1], self.hidden_irreps.dim).to(combined.device)
        return fallback(combined)

    @torch.jit.unused
    def update_fallback(self, x: torch.Tensor, m_agg: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, m_agg], dim=-1)
        fallback = nn.Linear(combined.shape[-1], self.node_irreps_in.dim).to(combined.device)
        return fallback(combined)
    
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
        
        # Initial projection
        self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(RobustEGNNLayer(
                node_irreps_in=self.hidden_irreps,
                edge_irreps_in=self.edge_irreps,
                hidden_irreps=self.hidden_irreps
            ))
        
        scalar_irreps_list = [(mul, irrep) for mul, irrep in self.hidden_irreps if irrep.l == 0]
        self.scalar_irreps = Irreps(scalar_irreps_list) if scalar_irreps_list else Irreps("0x0e")
        
        output_latent_dim = config.LATENT_DIM_GNN

        # Final projection
        self.final_projection = Linear(self.scalar_irreps, Irreps(f"{output_latent_dim}x0e"))
    
    def extract_scalar_features(self, x: torch.Tensor) -> torch.Tensor: # Type hint should be IrrepsArray
        """Extract scalar (l=0) features from tensor with irreps"""
        if not hasattr(x, 'irreps') or not isinstance(x.irreps, Irreps):
            warnings.warn("extract_scalar_features: Input tensor has no valid irreps. Assuming full tensor is scalar and returning it.")
            return x # Should not happen if previous steps ensure irreps

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
        x_raw = data.x
        pos = data.pos
        batch = data.batch
        
        # Initial projection
        x = self.initial_projection(x_raw)
        x = E3NNTensorWrapper.ensure_irreps(x, self.hidden_irreps)
        
        # Build radius graph
        edge_index = torch_geometric.nn.radius_graph(pos, self.radius, batch)
        
        if edge_index.size(1) == 0:
            # No edges case
            batch_size = batch.max().item() + 1 if batch is not None else 1
            return torch.zeros(batch_size, config.LATENT_DIM_GNN, device=x_raw.device)
        
        # Compute edge attributes
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)
        edge_dir = edge_vec / (edge_dist + 1e-8)
        
        # Combine to match edge_irreps: "1x1o + 1x0e"
        edge_attr = torch.cat([edge_dir, edge_dist], dim=-1)
        edge_attr = E3NNTensorWrapper.ensure_irreps(edge_attr, self.edge_irreps)
        
        # Apply EGNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Extract scalar features for pooling
        scalar_features = self.extract_scalar_features(x)
        
        # Global pooling
        pooled = global_mean_pool(scalar_features, batch)
        pooled = E3NNTensorWrapper.ensure_irreps(pooled, self.scalar_irreps)
        
        # Final projection
        output = self.final_projection(pooled)
        
        # Return as regular tensor (extract array from IrrepsArray)
        if hasattr(output, 'array'): # Check if it's an IrrepsArray
            return output.array
        return output # If not, return as is (should be a plain tensor)

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
                 node_input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 output_dim: int = 64,
                 radius: float = 5.0):
        super().__init__()
        
        self.radius = radius
        
        # Initial projection
        self.input_proj = nn.Linear(node_input_dim, hidden_dim)
        
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
    

class E3NNTensorWrapper:
    """
    Wrapper to ensure tensors have proper irreps attribute for e3nn operations.
    This solves the batching issue where irreps get lost.
    """
    @staticmethod
    def wrap_tensor(tensor: torch.Tensor, irreps: Irreps) -> torch.Tensor:
        """Attach irreps to tensor if not present"""
        if not hasattr(tensor, 'irreps'):
            # Create a new tensor with irreps attribute
            wrapped = tensor.clone()
            wrapped.irreps = irreps
            return wrapped
        return tensor
    
    @staticmethod
    def ensure_irreps(tensor: torch.Tensor, expected_irreps: Irreps) -> torch.Tensor:
        """Ensure tensor has the expected irreps"""
        if not hasattr(tensor, 'irreps'):
            tensor.irreps = expected_irreps
        elif tensor.irreps != expected_irreps:
            print(f"Warning: tensor irreps {tensor.irreps} != expected {expected_irreps}")
        return tensor

class RobustGate(nn.Module):
    """
    Robust gate implementation that handles e3nn Gate initialization issues
    """
    def __init__(self, irreps_in: Irreps, activation_scalars=F.silu, activation_gates=F.sigmoid):
        super().__init__()
        self.irreps_in = irreps_in
        
        # Separate scalars and non-scalars
        self.scalar_irreps = Irreps([(mul, irrep) for mul, irrep in irreps_in if irrep.l == 0])
        self.nonscalar_irreps = Irreps([(mul, irrep) for mul, irrep in irreps_in if irrep.l > 0])
        
        # Create gate irreps (one scalar gate per non-scalar irrep multiplicity)
        gate_irreps_list = []
        for mul, irrep in self.nonscalar_irreps:
            if irrep.l > 0:
                gate_irreps_list.append((mul, o3.Irrep(0, 1)))  # scalar gates
        self.gate_irreps = Irreps(gate_irreps_list) if gate_irreps_list else Irreps("0x0e")
        
        # Output irreps (scalars + gated non-scalars)
        self.irreps_out = self.scalar_irreps + self.nonscalar_irreps
        
        # Try to use e3nn Gate, fallback to manual implementation
        self.use_e3nn_gate = True
        try:
            if len(self.scalar_irreps) > 0 and len(self.nonscalar_irreps) > 0:
                self.gate = Gate(
                    irreps_scalars=self.scalar_irreps,
                    act_scalars=[activation_scalars] * len(self.scalar_irreps),
                    irreps_gates=self.gate_irreps,
                    act_gates=[activation_gates] * len(self.gate_irreps),
                    irreps_gated=self.nonscalar_irreps
                )
            elif len(self.scalar_irreps) > 0:
                # Only scalars, no gating needed
                self.scalar_activation = activation_scalars
                self.use_e3nn_gate = False
            else:
                # Only non-scalars, create simple gates
                self.gate_linear = nn.Linear(self.gate_irreps.dim, self.nonscalar_irreps.dim)
                self.gate_activation = activation_gates
                self.use_e3nn_gate = False
        except Exception as e:
            print(f"Warning: e3nn Gate failed, using manual implementation: {e}")
            self.use_e3nn_gate = False
            self.scalar_activation = activation_scalars
            if len(self.nonscalar_irreps) > 0:
                self.gate_linear = nn.Linear(self.gate_irreps.dim if self.gate_irreps.dim > 0 else 1, 
                                           self.nonscalar_irreps.dim)
                self.gate_activation = activation_gates
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_e3nn_gate and hasattr(self, 'gate'):
            result = self.gate(x)
            return E3NNTensorWrapper.ensure_irreps(result, self.irreps_out)
        
        # Manual gate implementation
        outputs = []
        start_idx = 0
        
        # Process scalars
        for mul, irrep in self.scalar_irreps:
            end_idx = start_idx + mul * irrep.dim
            scalar_part = x[:, start_idx:end_idx]
            if hasattr(self, 'scalar_activation'):
                scalar_part = self.scalar_activation(scalar_part)
            outputs.append(scalar_part)
            start_idx = end_idx
        
        # Process non-scalars with gating
        if len(self.nonscalar_irreps) > 0:
            gate_start = start_idx
            for mul, irrep in self.nonscalar_irreps:
                end_idx = start_idx + mul * irrep.dim
                nonscalar_part = x[:, start_idx:end_idx]
                
                if hasattr(self, 'gate_linear'):
                    # Create gates from the non-scalar features themselves (simplified)
                    gate_input = nonscalar_part.mean(dim=-1, keepdim=True)
                    gates = self.gate_activation(self.gate_linear(gate_input))
                    nonscalar_part = nonscalar_part * gates
                
                outputs.append(nonscalar_part)
                start_idx = end_idx
        
        result = torch.cat(outputs, dim=-1) if outputs else x
        return E3NNTensorWrapper.ensure_irreps(result, self.irreps_out)

class RobustEGNNLayer(nn.Module):
    """
    Robust EGNN layer with better error handling and tensor management
    """
    def __init__(self, node_irreps_in: Irreps, edge_irreps_in: Irreps, hidden_irreps: Irreps):
        super().__init__()
        self.node_irreps_in = node_irreps_in
        self.edge_irreps_in = edge_irreps_in
        self.hidden_irreps = hidden_irreps
        
        # Message tensor product
        self.message_tp = FullyConnectedTensorProduct(
            node_irreps_in, edge_irreps_in, hidden_irreps
        )
        
        # Message gate
        self.message_gate = RobustGate(self.message_tp.irreps_out)
        
        # Message output projection
        if self.message_gate.irreps_out != hidden_irreps:
            self.message_linear = Linear(self.message_gate.irreps_out, hidden_irreps)
        else:
            self.message_linear = nn.Identity()
        
        # Update tensor product
        self.update_tp = FullyConnectedTensorProduct(
            node_irreps_in, hidden_irreps, hidden_irreps
        )
        
        # Update gate
        self.update_gate = RobustGate(self.update_tp.irreps_out)
        
        # Update output projection
        if self.update_gate.irreps_out != node_irreps_in:
            self.update_linear = Linear(self.update_gate.irreps_out, node_irreps_in)
        else:
            self.update_linear = nn.Identity()
    
    def forward(self, node_features, edge_index, edge_attr):
        # wrap into IrrepsArray
        x = IrrepsArray(self.node_irreps_in, node_features)
        e = IrrepsArray(self.edge_irreps_in, edge_attr)

        # Python-only debug prints + asserts
        if not torch.jit.is_scripting():
            print(f"[DEBUG] node_features.shape={node_features.shape}, irreps.dim={self.node_irreps_in.dim}")
            print(f"[DEBUG] edge_attr.shape={edge_attr.shape}, irreps.dim={self.edge_irreps_in.dim}")
            assert node_features.shape[-1] == self.node_irreps_in.dim, (
                f"invalid node_features dim: got {node_features.shape[-1]}, "
                f"expected {self.node_irreps_in.dim}"
            )
            assert edge_attr.shape[-1] == self.edge_irreps_in.dim, (
                f"invalid edge_attr dim: got {edge_attr.shape[-1]}, "
                f"expected {self.edge_irreps_in.dim}"
            )

        row, col = edge_index

        # message passing (same as before) …
        m = self.message_tp(x[col], e)
        m = self.message_gate(m)
        m = self.message_linear(m)

        m_agg = torch_geometric.utils.scatter(
            m.array, row, dim=0, dim_size=x.shape[0], reduce="sum"
        )
        m_agg = IrrepsArray(self.hidden_irreps, m_agg)

        # update (same as before) …
        u = self.update_tp(x, m_agg)
        u = self.update_gate(u)
        u = self.update_linear(u)

        out = x + u
        return out.array

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
        
        # Initial projection
        self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(RobustEGNNLayer(
                node_irreps_in=self.hidden_irreps,
                edge_irreps_in=self.edge_irreps,
                hidden_irreps=self.hidden_irreps
            ))
        
        scalar_irreps_list = [(mul, irrep) for mul, irrep in self.hidden_irreps if irrep.l == 0]
        self.scalar_irreps = Irreps(scalar_irreps_list) if scalar_irreps_list else Irreps("0x0e")
        
        output_latent_dim = config.LATENT_DIM_GNN

        # Final projection
        self.final_projection = Linear(self.scalar_irreps, Irreps(f"{output_latent_dim}x0e"))
    
    def extract_scalar_features(self, x: torch.Tensor) -> torch.Tensor: # Type hint should be IrrepsArray
        """Extract scalar (l=0) features from tensor with irreps"""
        if not hasattr(x, 'irreps') or not isinstance(x.irreps, Irreps):
            warnings.warn("extract_scalar_features: Input tensor has no valid irreps. Assuming full tensor is scalar and returning it.")
            return x # Should not happen if previous steps ensure irreps

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
        x_raw = data.x
        pos = data.pos
        batch = data.batch
        
        # Initial projection
        x = self.initial_projection(x_raw)
        x = E3NNTensorWrapper.ensure_irreps(x, self.hidden_irreps)
        
        # Build radius graph
        edge_index = torch_geometric.nn.radius_graph(pos, self.radius, batch)
        
        if edge_index.size(1) == 0:
            # No edges case
            batch_size = batch.max().item() + 1 if batch is not None else 1
            return torch.zeros(batch_size, config.LATENT_DIM_GNN, device=x_raw.device)
        
        # Compute edge attributes
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)
        edge_dir = edge_vec / (edge_dist + 1e-8)
        
        # Combine to match edge_irreps: "1x1o + 1x0e"
        edge_attr = torch.cat([edge_dir, edge_dist], dim=-1)
        edge_attr = E3NNTensorWrapper.ensure_irreps(edge_attr, self.edge_irreps)
        
        # Apply EGNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Extract scalar features for pooling
        scalar_features = self.extract_scalar_features(x)
        
        # Global pooling
        pooled = global_mean_pool(scalar_features, batch)
        pooled = E3NNTensorWrapper.ensure_irreps(pooled, self.scalar_irreps)
        
        # Final projection
        output = self.final_projection(pooled)
        
        # Return as regular tensor (remove irreps for compatibility)
        if hasattr(output, 'array'): # Check if it's an IrrepsArray
            return output.array
        return output # If not, return as is (should be a plain tensor)

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
                 node_input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 output_dim: int = 64,
                 radius: float = 5.0):
        super().__init__()
        
        self.radius = radius
        
        # Initial projection
        self.input_proj = nn.Linear(node_input_dim, hidden_dim)
        
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
    

class E3NNTensorWrapper:
    """
    Wrapper to ensure tensors have proper irreps attribute for e3nn operations.
    This solves the batching issue where irreps get lost.
    """
    @staticmethod
    def wrap_tensor(tensor: torch.Tensor, irreps: Irreps) -> torch.Tensor:
        """Attach irreps to tensor if not present"""
        if not hasattr(tensor, 'irreps'):
            # Create a new tensor with irreps attribute
            wrapped = tensor.clone()
            wrapped.irreps = irreps
            return wrapped
        return tensor
    
    @staticmethod
    def ensure_irreps(tensor: torch.Tensor, expected_irreps: Irreps) -> torch.Tensor:
        """Ensure tensor has the expected irreps"""
        if not hasattr(tensor, 'irreps'):
            tensor.irreps = expected_irreps
        elif tensor.irreps != expected_irreps:
            print(f"Warning: tensor irreps {tensor.irreps} != expected {expected_irreps}")
        return tensor

class RobustGate(nn.Module):
    """
    Robust gate implementation that handles e3nn Gate initialization issues
    """
    def __init__(self, irreps_in: Irreps, activation_scalars=F.silu, activation_gates=F.sigmoid):
        super().__init__()
        self.irreps_in = irreps_in
        
        # Separate scalars and non-scalars
        self.scalar_irreps = Irreps([(mul, irrep) for mul, irrep in irreps_in if irrep.l == 0])
        self.nonscalar_irreps = Irreps([(mul, irrep) for mul, irrep in irreps_in if irrep.l > 0])
        
        # Create gate irreps (one scalar gate per non-scalar irrep multiplicity)
        gate_irreps_list = []
        for mul, irrep in self.nonscalar_irreps:
            if irrep.l > 0:
                gate_irreps_list.append((mul, o3.Irrep(0, 1)))  # scalar gates
        self.gate_irreps = Irreps(gate_irreps_list) if gate_irreps_list else Irreps("0x0e")
        
        # Output irreps (scalars + gated non-scalars)
        self.irreps_out = self.scalar_irreps + self.nonscalar_irreps
        
        # Try to use e3nn Gate, fallback to manual implementation
        self.use_e3nn_gate = True
        try:
            if len(self.scalar_irreps) > 0 and len(self.nonscalar_irreps) > 0:
                self.gate = Gate(
                    irreps_scalars=self.scalar_irreps,
                    act_scalars=[activation_scalars] * len(self.scalar_irreps),
                    irreps_gates=self.gate_irreps,
                    act_gates=[activation_gates] * len(self.gate_irreps),
                    irreps_gated=self.nonscalar_irreps
                )
            elif len(self.scalar_irreps) > 0:
                # Only scalars, no gating needed
                self.scalar_activation = activation_scalars
                self.use_e3nn_gate = False
            else:
                # Only non-scalars, create simple gates
                self.gate_linear = nn.Linear(self.gate_irreps.dim, self.nonscalar_irreps.dim)
                self.gate_activation = activation_gates
                self.use_e3nn_gate = False
        except Exception as e:
            print(f"Warning: e3nn Gate failed, using manual implementation: {e}")
            self.use_e3nn_gate = False
            self.scalar_activation = activation_scalars
            if len(self.nonscalar_irreps) > 0:
                self.gate_linear = nn.Linear(self.gate_irreps.dim if self.gate_irreps.dim > 0 else 1, 
                                           self.nonscalar_irreps.dim)
                self.gate_activation = activation_gates
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_e3nn_gate and hasattr(self, 'gate'):
            result = self.gate(x)
            return E3NNTensorWrapper.ensure_irreps(result, self.irreps_out)
        
        # Manual gate implementation
        outputs = []
        start_idx = 0
        
        # Process scalars
        for mul, irrep in self.scalar_irreps:
            end_idx = start_idx + mul * irrep.dim
            scalar_part = x[:, start_idx:end_idx]
            if hasattr(self, 'scalar_activation'):
                scalar_part = self.scalar_activation(scalar_part)
            outputs.append(scalar_part)
            start_idx = end_idx
        
        # Process non-scalars with gating
        if len(self.nonscalar_irreps) > 0:
            gate_start = start_idx
            for mul, irrep in self.nonscalar_irreps:
                end_idx = start_idx + mul * irrep.dim
                nonscalar_part = x[:, start_idx:end_idx]
                
                if hasattr(self, 'gate_linear'):
                    # Create gates from the non-scalar features themselves (simplified)
                    gate_input = nonscalar_part.mean(dim=-1, keepdim=True)
                    gates = self.gate_activation(self.gate_linear(gate_input))
                    nonscalar_part = nonscalar_part * gates
                
                outputs.append(nonscalar_part)
                start_idx = end_idx
        
        result = torch.cat(outputs, dim=-1) if outputs else x
        return E3NNTensorWrapper.ensure_irreps(result, self.irreps_out)

class RobustEGNNLayer(nn.Module):
    """
    Robust EGNN layer with better error handling and tensor management
    """
    def __init__(self, node_irreps_in: Irreps, edge_irreps_in: Irreps, hidden_irreps: Irreps):
        super().__init__()
        self.node_irreps_in = node_irreps_in
        self.edge_irreps_in = edge_irreps_in
        self.hidden_irreps = hidden_irreps
        
        # Message tensor product
        self.message_tp = FullyConnectedTensorProduct(
            node_irreps_in, edge_irreps_in, hidden_irreps
        )
        
        # Message gate
        self.message_gate = RobustGate(self.message_tp.irreps_out)
        
        # Message output projection
        if self.message_gate.irreps_out != hidden_irreps:
            self.message_linear = Linear(self.message_gate.irreps_out, hidden_irreps)
        else:
            self.message_linear = nn.Identity()
        
        # Update tensor product
        self.update_tp = FullyConnectedTensorProduct(
            node_irreps_in, hidden_irreps, hidden_irreps
        )
        
        # Update gate
        self.update_gate = RobustGate(self.update_tp.irreps_out)
        
        # Update output projection
        if self.update_gate.irreps_out != node_irreps_in:
            self.update_linear = Linear(self.update_gate.irreps_out, node_irreps_in)
        else:
            self.update_linear = nn.Identity()
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        
        # Ensure tensors have proper irreps
        node_features = E3NNTensorWrapper.ensure_irreps(node_features, self.node_irreps_in)
        edge_attr = E3NNTensorWrapper.ensure_irreps(edge_attr, self.edge_irreps_in)
        
        row, col = edge_index
        
        # Message passing
        try:
            # Compute messages
            messages = self.message_tp(node_features[col], edge_attr)
            messages = E3NNTensorWrapper.ensure_irreps(messages, self.message_tp.irreps_out)
            
            # Apply gate
            messages = self.message_gate(messages)
            
            # Apply output projection
            if not isinstance(self.message_linear, nn.Identity):
                messages = self.message_linear(messages)
            messages = E3NNTensorWrapper.ensure_irreps(messages, self.hidden_irreps)
            
        except Exception as e:
         #   print(f"Error in message computation: {e}")
            # Fallback: simple linear combination
            combined = torch.cat([node_features[col], edge_attr], dim=-1)
            if not hasattr(self, 'message_fallback'):
                self.message_fallback = nn.Linear(combined.shape[-1], self.hidden_irreps.dim).to(combined.device)
            messages = self.message_fallback(combined)
            messages = E3NNTensorWrapper.ensure_irreps(messages, self.hidden_irreps)
        
        # Aggregate messages
        aggregated = torch_geometric.utils.scatter(
            messages, row, dim=0, dim_size=node_features.size(0), reduce="sum"
        )
        aggregated = E3NNTensorWrapper.ensure_irreps(aggregated, self.hidden_irreps)
        
        # Update nodes
        try:
            # Compute updates
            updates = self.update_tp(node_features, aggregated)
            updates = E3NNTensorWrapper.ensure_irreps(updates, self.update_tp.irreps_out)
            
            # Apply gate
            updates = self.update_gate(updates)
            
            # Apply output projection
            if not isinstance(self.update_linear, nn.Identity):
                updates = self.update_linear(updates)
            updates = E3NNTensorWrapper.ensure_irreps(updates, self.node_irreps_in)
            
        except Exception as e:
         #   print(f"Error in update computation: {e}")
            # Fallback: simple linear combination
            combined = torch.cat([node_features, aggregated], dim=-1)
            if not hasattr(self, 'update_fallback'):
                self.update_fallback = nn.Linear(combined.shape[-1], self.node_irreps_in.dim).to(combined.device)
            updates = self.update_fallback(combined)
            updates = E3NNTensorWrapper.ensure_irreps(updates, self.node_irreps_in)
        
        # Residual connection
        result = node_features + updates
        return E3NNTensorWrapper.ensure_irreps(result, self.node_irreps_in)

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
        
        # Initial projection
        self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(RobustEGNNLayer(
                node_irreps_in=self.hidden_irreps,
                edge_irreps_in=self.edge_irreps,
                hidden_irreps=self.hidden_irreps
            ))
        
        scalar_irreps_list = [(mul, irrep) for mul, irrep in self.hidden_irreps if irrep.l == 0]
        self.scalar_irreps = Irreps(scalar_irreps_list) if scalar_irreps_list else Irreps("0x0e")
        
        output_latent_dim = config.LATENT_DIM_GNN

        # Final projection
        self.final_projection = Linear(self.scalar_irreps, Irreps(f"{output_latent_dim}x0e"))
    
    def extract_scalar_features(self, x: torch.Tensor) -> torch.Tensor: # Type hint should be IrrepsArray
        """Extract scalar (l=0) features from tensor with irreps"""
        if not hasattr(x, 'irreps') or not isinstance(x.irreps, Irreps):
            warnings.warn("extract_scalar_features: Input tensor has no valid irreps. Assuming full tensor is scalar and returning it.")
            return x # Should not happen if previous steps ensure irreps

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
        x_raw = data.x
        pos = data.pos
        batch = data.batch
        
        # Initial projection
        x = self.initial_projection(x_raw)
        x = E3NNTensorWrapper.ensure_irreps(x, self.hidden_irreps)
        
        # Build radius graph
        edge_index = torch_geometric.nn.radius_graph(pos, self.radius, batch)
        
        if edge_index.size(1) == 0:
            # No edges case
            batch_size = batch.max().item() + 1 if batch is not None else 1
            return torch.zeros(batch_size, config.LATENT_DIM_GNN, device=x_raw.device)
        
        # Compute edge attributes
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)
        edge_dir = edge_vec / (edge_dist + 1e-8)
        
        # Combine to match edge_irreps: "1x1o + 1x0e"
        edge_attr = torch.cat([edge_dir, edge_dist], dim=-1)
        edge_attr = E3NNTensorWrapper.ensure_irreps(edge_attr, self.edge_irreps)
        
        # Apply EGNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Extract scalar features for pooling
        scalar_features = self.extract_scalar_features(x)
        
        # Global pooling
        pooled = global_mean_pool(scalar_features, batch)
        pooled = E3NNTensorWrapper.ensure_irreps(pooled, self.scalar_irreps)
        
        # Final projection
        output = self.final_projection(pooled)
        
        # Return as regular tensor (extract array from IrrepsArray)
        if hasattr(output, 'array'): # Check if it's an IrrepsArray
            return output.array
        return output # If not, return as is (should be a plain tensor)

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
                 node_input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 output_dim: int = 64,
                 radius: float = 5.0):
        super().__init__()
        
        self.radius = radius
        
        # Initial projection
        self.input_proj = nn.Linear(node_input_dim, hidden_dim)
        
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
        
        ffnn_hidden_dims_asph: List[int] = config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar: List[int] = config.FFNN_HIDDEN_DIMS_SCALAR,
        
        # Shared fusion params
        latent_dim_gnn: int = config.LATENT_DIM_GNN,
        latent_dim_asph: int = config.LATENT_DIM_ASPH,
        latent_dim_other_ffnn: int = config.LATENT_DIM_OTHER_FFNN,
        
        fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,
    ):
        super().__init__()

        self.crystal_encoder = RealSpaceEGNNEncoder(
            node_input_scalar_dim=crystal_node_feature_dim,
            hidden_irreps_str=egnn_hidden_irreps_str,
            n_layers=egnn_num_layers,
            radius=egnn_radius,
        )
        
        self.kspace_encoder = PhysicsInformedKSpaceEncoder(
            node_feature_dim=kspace_node_feature_dim,
            hidden_dim=kspace_gnn_hidden_channels,
            num_layers=kspace_gnn_num_layers, 
            output_dim=latent_dim_gnn 
        )
        self.asph_encoder = PHTokenEncoder(
            input_dim=asph_feature_dim,
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

        total_fused_dim = (latent_dim_gnn * 2) + latent_dim_asph + (latent_dim_other_ffnn * 2) 

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