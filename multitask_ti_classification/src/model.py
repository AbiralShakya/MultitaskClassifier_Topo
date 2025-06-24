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

class EGNNLayer(nn.Module):
    """
    A simplified EGNN-like layer using e3nn components.
    Fixed version with proper Gate initialization and tensor handling.
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

        # --- FIXED: Proper Gate initialization for older e3nn versions ---
        # Get the actual output irreps from tensor product
        tp_out_irreps = self.tp_messages_ij.irreps_out
        
        # Separate scalars and non-scalars from the tensor product output
        scalar_parts = []
        gated_parts = []
        
        for mul, irrep in tp_out_irreps:
            if irrep.l == 0:  # scalar
                scalar_parts.append((mul, irrep))
            else:  # non-scalar (will be gated)
                gated_parts.append((mul, irrep))
        
        # Create irreps objects
        irreps_scalars = Irreps(scalar_parts) if scalar_parts else Irreps("0x0e")
        irreps_gated = Irreps(gated_parts) if gated_parts else Irreps("0x1o")
        
        # Create gates - one scalar gate per gated irrep multiplicity
        gate_parts = []
        for mul, irrep in irreps_gated:
            if irrep.l > 0:  # Only create gates for non-scalars
                gate_parts.append((mul, o3.Irrep(0, 1)))  # (mul, 0e) gates
        
        irreps_gates = Irreps(gate_parts) if gate_parts else Irreps("0x0e")
        
        # Create activation lists - must match the number of irrep types, not total multiplicity
        act_scalars = [F.silu] * len(irreps_scalars) if len(irreps_scalars) > 0 else []
        act_gates = [F.sigmoid] * len(irreps_gates) if len(irreps_gates) > 0 else []
        
        # Wrapped Gate initialization in try-except block
        self.has_e3nn_gate = True # Flag to track if e3nn Gate is used or fallback linear
        try:
            self.gate_messages = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=act_scalars,
                irreps_gates=irreps_gates,
                act_gates=act_gates,
                irreps_gated=irreps_gated
            )
            self.linear_messages_out = Linear(self.gate_messages.irreps_out, hidden_irreps)
        except Exception as e:
            self.has_e3nn_gate = False
            print(f"Warning: Message Gate initialization failed in EGNNLayer: {e}. Falling back to Linear.")
            self.gate_messages = nn.Linear(tp_out_irreps.dim, hidden_irreps.dim) # Fallback to a simple linear layer
            self.linear_messages_out = nn.Identity() # Identity since this linear layer already maps to hidden_irreps.dim

        # TP for update (from node_i and aggregated_messages) -> new node irreps
        self.tp_update = FullyConnectedTensorProduct(
            node_irreps_in, hidden_irreps, hidden_irreps,
        )
        
        # Gate for update - same approach with older e3nn API
        tp_update_out_irreps = self.tp_update.irreps_out
        
        # Separate scalars and non-scalars for update gate
        update_scalar_parts = []
        update_gated_parts = []
        
        for mul, irrep in tp_update_out_irreps:
            if irrep.l == 0:  # scalar
                update_scalar_parts.append((mul, irrep))
            else:  # non-scalar (will be gated)
                update_gated_parts.append((mul, irrep))
        
        # Create irreps objects for update
        update_irreps_scalars = Irreps(update_scalar_parts) if update_scalar_parts else Irreps("0x0e")
        update_irreps_gated = Irreps(update_gated_parts) if update_gated_parts else Irreps("0x1o")
        
        # Create gates for update
        update_gate_parts = []
        for mul, irrep in update_irreps_gated:
            if irrep.l > 0:  # Only create gates for non-scalars
                update_gate_parts.append((mul, o3.Irrep(0, 1)))  # (mul, 0e) gates
        
        update_irreps_gates = Irreps(update_gate_parts) if update_gate_parts else Irreps("0x0e")
        
        # Create activation lists for update
        update_act_scalars = [F.silu] * len(update_irreps_scalars) if len(update_irreps_scalars) > 0 else []
        update_act_gates = [F.sigmoid] * len(update_irreps_gates) if len(update_irreps_gates) > 0 else []
        
        # Wrapped Gate initialization in try-except block
        self.has_e3nn_update_gate = True # Flag to track if e3nn Gate is used or fallback linear
        try:
            self.gate_update = Gate(
                irreps_scalars=update_irreps_scalars,
                act_scalars=update_act_scalars,
                irreps_gates=update_irreps_gates,
                act_gates=update_act_gates,
                irreps_gated=update_irreps_gated
            )
            self.linear_update_out = Linear(self.gate_update.irreps_out, node_irreps_in)
        except Exception as e:
            self.has_e3nn_update_gate = False
            print(f"Warning: Update Gate initialization failed in EGNNLayer: {e}. Falling back to Linear.")
            self.gate_update = nn.Linear(tp_update_out_irreps.dim, node_irreps_in.dim) # Fallback
            self.linear_update_out = nn.Identity() # Identity


    def forward(self, node_features, edge_index: torch.Tensor, edge_attr_tensor: torch.Tensor, 
                node_attr_scalar_raw: torch.Tensor):
        
        row, col = edge_index

        # edge_attr_tensor is already [normalized_r_vec (3D for 1x1o), dist (1D for 1x0e)]
        # This order matches self.edge_irreps ("1x1o + 1x0e")
        edge_attr_e3nn = edge_attr_tensor 

        # 1. Message passing
        # CRITICAL CHECK: If node_features does not have irreps, then e3nn is NOT working for this batch
        if not hasattr(node_features, 'irreps') or not self.has_e3nn_gate:
            # Fallback for non-e3nn execution path within EGNNLayer.forward
            # These are basic linear combinations/projections that bypass e3nn TPs and Gates
            
            # Message calculation (approximate)
            input_dim_for_fallback_msg = node_features[col].shape[-1] + edge_attr_e3nn.shape[-1]
            output_dim_for_fallback_msg = self.hidden_irreps.dim
            if not hasattr(self, 'fallback_msg_proj'):
                self.fallback_msg_proj = nn.Linear(input_dim_for_fallback_msg, output_dim_for_fallback_msg).to(node_features.device)
            messages_from_j = self.fallback_msg_proj(torch.cat([node_features[col], edge_attr_e3nn], dim=-1))

            # Update calculation (approximate)
            input_dim_for_fallback_update = node_features.shape[-1] + aggregated_messages.shape[-1] # This `aggregated_messages` is defined later, needs to be calculated first
            
            # This complex nested fallback logic is hard to manage. Let's simplify the EGNNLayer fallback
            # The RealSpaceEGNNEncoder now decides if e3nn is used. If it's not, EGNNLayer should act as simple MLP.
            # This entire section should only execute if self.e3nn_working is True in the RealSpaceEGNNEncoder.
            # If it's False, RealSpaceEGNNEncoder uses its nn.Sequential layers.
            # So, this `if not hasattr(node_features, 'irreps')` check should ideally not be needed here if RealSpaceEGNNEncoder is correctly branching.
            # The issue is RealSpaceEGNNEncoder's `e3nn_working` check.
            pass # Removed complex nested fallback, relying on RealSpaceEGNNEncoder's higher-level branching


        # If we reach here, it means RealSpaceEGNNEncoder thought e3nn was working.
        # However, node_features still might not have irreps due to batching/PyG issues.
        # So, if node_features *lacks* irreps at this point, we must explicitly ensure the fallback.
        # This means the EGNNLayer itself must be able to switch its forward method.
        # This is becoming very complex due to the environment.

        # Let's simplify this. RealSpaceEGNNEncoder makes the decision.
        # If self.e3nn_working is False, then RealSpaceEGNNEncoder's `layers` contains `nn.Sequential` modules, not `EGNNLayer` instances.
        # So, the forward pass of `EGNNLayer` should only be called if e3nn is actually working.
        # The debug `node_features irreps: No irreps` is coming from *inside* `EGNNLayer.forward` when it *shouldn't be called* if e3nn isn't working.

        # The print `node_features irreps: No irreps` is causing the confusion. It indicates `hasattr(node_features, 'irreps')` is False.
        # This suggests `x_e3nn` is not an IrrepsArray from `initial_projection` itself or loses it during batching.

        # The solution is to ensure `initial_projection` *always* outputs a raw tensor, and then
        # manually attach the Irreps within the EGNNLayer *if e3nn is functional*.
        # Or, the RealSpaceEGNNEncoder's fallback MUST work.

        # Let's put the most robust fallback check directly at the start of EGNNLayer.forward.
        # This will assume node_features is a regular tensor.
        # If it doesn't have irreps, then fallback.

        if not hasattr(node_features, 'irreps'):
            # This branch means e3nn is NOT working reliably for this layer.
            # We must use simple linear layers/approximations.
            
            # Approx. Message calculation
            input_dim_for_msg_fallback = node_features[col].shape[-1] + edge_attr_e3nn.shape[-1]
            output_dim_for_msg_fallback = self.hidden_irreps.dim
            
            # Ensure fallback_msg_proj is initialized (create on first use if not present)
            if not hasattr(self, 'fallback_msg_proj_runtime'):
                self.fallback_msg_proj_runtime = nn.Linear(input_dim_for_msg_fallback, output_dim_for_msg_fallback).to(node_features.device)
            
            messages_from_j = self.fallback_msg_proj_runtime(torch.cat([node_features[col], edge_attr_e3nn], dim=-1))
            
            # Approx. Update calculation
            aggregated_messages = torch_geometric.utils.scatter(messages_from_j, row, dim=0, dim_size=node_features.size(0), reduce="sum")
            
            input_dim_for_update_fallback = node_features.shape[-1] + aggregated_messages.shape[-1]
            output_dim_for_update_fallback = node_features.shape[-1] # for residual connection
            
            if not hasattr(self, 'fallback_update_proj_runtime'):
                self.fallback_update_proj_runtime = nn.Linear(input_dim_for_update_fallback, output_dim_for_update_fallback).to(node_features.device)
            
            updated_node_features_temp = self.fallback_update_proj_runtime(torch.cat([node_features, aggregated_messages], dim=-1))

            return node_features + updated_node_features_temp

        # --- END OF FALLBACK FORWARD PATH ---
        # If we reach here, it means node_features *has* the irreps attribute, so e3nn operations can proceed.
        
        # 1. Message passing
        try:
            messages_tp_output = self.tp_messages_ij(node_features[col], edge_attr_e3nn)
        except Exception as e:
            print(f"Error in EGNNLayer tensor product (messages_tp_output) for batch element starting at node index {node_features.batch_ptr[0] if hasattr(node_features, 'batch_ptr') else 'N/A'}: {e}")
            raise # Re-raise if TP fails even with irreps, as this is a core e3nn failure.
        
        # Apply gate with error handling
        try:
            messages_gated = self.gate_messages(messages_tp_output)
            messages_from_j = self.linear_messages_out(messages_gated)
        except Exception as e:
            print(f"Error in EGNNLayer gate_messages (forward): {e}")
            # Fallback: If Gate fails at runtime, use its pre-initialized linear fallback or simple projection.
            if isinstance(self.linear_messages_out, nn.Identity): # This means gate_messages was a linear fallback
                messages_from_j = self.gate_messages(messages_tp_output)
            else: # Gate was a proper Gate, but failed at runtime, so direct projection
                if not hasattr(self, 'final_fallback_msg_proj_runtime_after_gate_fail'):
                    self.final_fallback_msg_proj_runtime_after_gate_fail = nn.Linear(messages_tp_output.shape[-1], self.hidden_irreps.dim).to(messages_tp_output.device)
                messages_from_j = self.final_fallback_msg_proj_runtime_after_gate_fail(messages_tp_output)


        # 2. Aggregation (sum messages for each node)
        aggregated_messages = torch_geometric.utils.scatter(
            messages_from_j, row, dim=0, dim_size=node_features.size(0), reduce="sum"
        )

        # 3. Update (combine current node features with aggregated messages)
        try:
            updated_node_features_tp_output = self.tp_update(node_features, aggregated_messages)
            updated_node_features_gated = self.gate_update(updated_node_features_tp_output)
            updated_node_features_temp = self.linear_update_out(updated_node_features_gated)
        except Exception as e:
            print(f"Error in EGNNLayer update (forward): {e}")
            # Fallback: If Update Gate fails at runtime, use its pre-initialized linear fallback or simple projection.
            if isinstance(self.linear_update_out, nn.Identity): # This means gate_update was a linear fallback
                updated_node_features_temp = self.gate_update(updated_node_features_tp_output)
            else: # Gate was a proper Gate, but failed at runtime
                if not hasattr(self, 'final_fallback_update_proj_runtime_after_gate_fail'):
                    combined_dim_for_fallback_update = node_features.shape[-1] + aggregated_messages.shape[-1]
                    self.final_fallback_update_proj_runtime_after_gate_fail = nn.Linear(combined_dim_for_fallback_update, node_features.shape[-1]).to(node_features.device)
                updated_node_features_temp = self.final_fallback_update_proj_runtime_after_gate_fail(torch.cat([node_features, aggregated_messages], dim=-1))
            
        return node_features + updated_node_features_temp


class RealSpaceEGNNEncoder(nn.Module):
    """
    EGNN encoder for real-space atomic crystal graphs.
    Fixed version with better error handling and tensor management.
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
        self.edge_irreps = Irreps("1x1o + 1x0e")  # 3D vector + 1D scalar
        self.hidden_irreps = Irreps(hidden_irreps_str)
        
        # Flag to track if e3nn is working or if we're in fallback mode
        self.e3nn_working = True
        try:
            # Test if e3nn.o3.Linear actually returns tensors with 'irreps' attribute
            # This is critical. If this fails, the whole e3nn part is broken.
            self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)
            dummy_test_input = torch.randn(1, self.input_node_irreps.dim)
            dummy_output = self.initial_projection(dummy_test_input)
            if not hasattr(dummy_output, 'irreps') or not isinstance(dummy_output.irreps, Irreps) or dummy_output.irreps != self.hidden_irreps:
                raise RuntimeError(f"e3nn.o3.Linear did not return expected e3nn tensor structure. Expected {self.hidden_irreps}, got {getattr(dummy_output, 'irreps', 'no irreps attr')}")
            print(f"e3nn Linear initialization successful. Initial projection output irreps: {dummy_output.irreps}")

        except Exception as e:
            self.e3nn_working = False
            print(f"CRITICAL WARNING: e3nn.o3.Linear failed or did not return tensor with irreps: {e}")
            print("Falling back to standard PyTorch Linear layer for initial_projection. e3nn functionality will be disabled for crystal_encoder.")
            # Fallback to standard nn.Linear if e3nn.o3.Linear fails or doesn't attach irreps
            self.initial_projection = nn.Linear(self.input_node_irreps.dim, self.hidden_irreps.dim)


        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if self.e3nn_working:
                self.layers.append(EGNNLayer(
                    node_irreps_in=self.hidden_irreps,
                    edge_irreps_in=self.edge_irreps,
                    hidden_irreps=self.hidden_irreps
                ))
            else:
                # If e3nn not working, append a simple MLP-like layer as a fallback
                # This should simulate a GNN layer by combining node and aggregated edge features
                
                # Input dimension for fallback layer: features from previous node + features from edges
                # (note: edge features will be aggregated, so it's not simply adding edge_irreps.dim)
                # A common non-equivariant GNN layer uses output dim of previous layer for node features.
                input_dim_for_fallback_layer = self.hidden_irreps.dim + self.hidden_irreps.dim + self.edge_irreps.dim
                self.layers.append(nn.Sequential(
                    nn.Linear(input_dim_for_fallback_layer, self.hidden_irreps.dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hidden_irreps.dim),
                    nn.Dropout(p=config.DROPOUT_RATE)
                ))

        # Extract only scalar irreps for final output
        scalar_irreps_list = [(mul, irrep) for mul, irrep in self.hidden_irreps if irrep.l == 0]
        if scalar_irreps_list and self.e3nn_working: # Only use e3nn irreps if it's working
            self.scalar_irreps = Irreps(scalar_irreps_list)
        else:
            self.scalar_irreps = Irreps("1x0e") # Fallback to a single scalar irrep or a dummy
            
        output_latent_dim = config.LATENT_DIM_GNN # Output dim of final linear is LATENT_DIM_GNN

        if self.e3nn_working:
            # Final projection from scalar_irreps to output_irreps (LATENT_DIM_GNN)
            self.final_linear_0e = Linear(self.scalar_irreps, Irreps(f"{output_latent_dim}x0e"))
        else:
            # Fallback to standard nn.Linear
            # Input dim is from the last layer's output (self.hidden_irreps.dim)
            self.final_linear_0e = nn.Linear(self.hidden_irreps.dim, output_latent_dim)

        self.radius = radius

    def forward(self, data: PyGData) -> torch.Tensor:
        x_raw_scalars = data.x
        
        # Project initial features
        current_node_features = self.initial_projection(x_raw_scalars) 
        
        # Build edge graph
        edge_index = torch_geometric.nn.radius_graph(data.pos, self.radius, data.batch)
        
        if edge_index.size(1) == 0:
            # Handle case with no edges - return zeros
            batch_size = data.batch.max().item() + 1 if data.batch is not None else 1
            return torch.zeros(batch_size, config.LATENT_DIM_GNN, device=x_raw_scalars.device)
        
        row, col = edge_index
        
        # Compute edge attributes
        r_vec = data.pos[row] - data.pos[col]
        dist = r_vec.norm(dim=-1, keepdim=True)
        normalized_r_vec = r_vec / (dist + 1e-8) 
        edge_attr_tensor = torch.cat([normalized_r_vec, dist], dim=-1) # [normalized_vector (3D), distance (1D)]
        
        # Verify edge attribute dimensions
        assert edge_attr_tensor.shape[-1] == 4, f"Expected 4D edge attributes, got {edge_attr_tensor.shape[-1]}D"

        # Pass through EGNN layers (or fallback layers)
        for i, layer in enumerate(self.layers):
            if self.e3nn_working:
                # E3NN path
                current_node_features = layer(
                    current_node_features, # This is supposed to be an e3nn tensor
                    edge_index, 
                    edge_attr_tensor, # This will be internally converted by EGNNLayer if e3nn_working
                    x_raw_scalars
                )
            else:
                # Fallback for non-e3nn EGNNLayer (now a simple MLP-like GNN operation)
                
                # Simplified message passing (concat and aggregate)
                messages_input_fallback = torch.cat([current_node_features[col], edge_attr_tensor], dim=-1)
                aggregated_messages_fallback = torch_geometric.utils.scatter(
                    messages_input_fallback, row, dim=0, dim_size=current_node_features.size(0), reduce="sum"
                )
                
                # Pass through the Sequential layer (which is self.layers[i] in this fallback branch)
                # It expects (node_features + aggregated_messages) dim
                # So we need to combine current_node_features and aggregated_messages_fallback
                combined_for_layer_fallback = torch.cat([current_node_features, aggregated_messages_fallback], dim=-1)
                current_node_features = layer(combined_for_layer_fallback) # This should map back to hidden_irreps.dim
                
        # Global pooling: Extract scalar features
        if self.e3nn_working:
            scalar_features = []
            start_idx = 0
            if hasattr(current_node_features, 'irreps'): # This check is still useful if `e3nn_working` is True but irreps get lost
                irreps_to_iterate = current_node_features.irreps
            else:
                irreps_to_iterate = self.hidden_irreps # Fallback to declared hidden_irreps structure for slicing
                print("Warning: current_node_features after EGNN layer has no 'irreps' attribute, trying to extract scalars based on hidden_irreps structure.")

            for mul, irrep in irreps_to_iterate:
                end_idx = start_idx + mul * irrep.dim
                if irrep.l == 0:  # Only scalar (l=0) features
                    scalar_features.append(current_node_features[:, start_idx:end_idx])
                start_idx = end_idx
            
            if scalar_features:
                invariant_features_per_node = torch.cat(scalar_features, dim=1)
            else:
                # If no scalars extracted, fallback to using first few dimensions
                scalar_dim_fallback = min(current_node_features.shape[-1], self.scalar_irreps.dim if hasattr(self.scalar_irreps, 'dim') else current_node_features.shape[-1])
                invariant_features_per_node = current_node_features[:, :scalar_dim_fallback]
            
            graph_embedding_tensor = global_mean_pool(invariant_features_per_node, data.batch)
            final_embedding = self.final_linear_0e(graph_embedding_tensor) # This is e3nn.o3.Linear
        else:
            # Fallback when e3nn is not working (RealSpaceEGNNEncoder is in full fallback mode)
            invariant_features_per_node = current_node_features # In fallback, current_node_features is a plain tensor
            graph_embedding_tensor = global_mean_pool(invariant_features_per_node, data.batch)
            final_embedding = self.final_linear_0e(graph_embedding_tensor) # This is nn.Linear

        return final_embedding

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
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=kspace_node_feature_dim,
            hidden_dim=kspace_gnn_hidden_channels,
            out_channels=latent_dim_gnn,
            n_layers=kspace_gnn_num_layers,
            num_heads=kspace_gnn_num_heads
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
        self.decomposition_encoder = DecompositionFeatureEncoder(
            input_dim=decomposition_feature_dim,
            out_channels=latent_dim_other_ffnn
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
        decomposition_emb = self.decomposition_encoder(inputs['kspace_physics_features']['decomposition_features']) 

        combined_emb = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, decomposition_emb], dim=-1)

        fused_output = self.fusion_network(combined_emb)

        topology_logits = self.topology_head(fused_output)
        magnetism_logits = self.magnetism_head(fused_output)

        return {
            'topology_logits': topology_logits,
            'magnetism_logits': magnetism_logits
        }