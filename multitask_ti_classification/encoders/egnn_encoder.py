# encoders/egnn_encoder.py
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import global_mean_pool
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from e3nn.nn import Gate
from e3nn.util.jit import script

# A helper for creating standard e3nn blocks
class GNNLayer(nn.Module):
    def __init__(self, node_irreps_in, edge_irreps_in, node_irreps_out, hidden_irreps):
        super().__init__()
        self.node_irreps_in = o3.Irreps(node_irreps_in)
        self.edge_irreps_in = o3.Irreps(edge_irreps_in)
        self.node_irreps_out = o3.Irreps(node_irreps_out)
        self.hidden_irreps = o3.Irreps(hidden_irreps)

        # Message network: combines node features from neighbor, central node, and edge features
        # and outputs features that will be summed.
        # This is highly customizable. A common pattern is:
        # (node_i, node_j, edge_attr) -> hidden_irreps -> node_irreps_out (for message)
        self.tp_messages = FullyConnectedTensorProduct(
            self.node_irreps_in, self.node_irreps_in, self.edge_irreps_in, self.hidden_irreps,
        )
        self.gate_messages = Gate(self.hidden_irreps)
        self.lin_messages_out = Linear(self.gate_messages.irreps_out, self.node_irreps_out)

        # Update network: combines original node features with aggregated messages
        # (node_i, aggregated_messages) -> node_irreps_out (new node features)
        self.tp_update = FullyConnectedTensorProduct(
            self.node_irreps_in, self.node_irreps_out, self.hidden_irreps,
        )
        self.gate_update = Gate(self.hidden_irreps)
        self.lin_update_out = Linear(self.gate_update.irreps_out, self.node_irreps_out)


    def forward(self, node_features, pos, edge_index, edge_attr):
        row, col = edge_index

        # Edge features: normalized displacement vector + distance (scalar)
        # Note: You'll need to compute these edge_attr (r_ij vector and distance)
        # before passing to the layer.
        # e3nn expects input features to be properly structured with irreps.
        
        # Messages from neighbors to central node
        messages = self.lin_messages_out(self.gate_messages(
            self.tp_messages(node_features[row], node_features[col], edge_attr)
        ))

        # Aggregate messages: Sum messages for each node
        # e3nn uses scatter operations that respect irreps
        # This is typically done with e3nn's own scatter functions or PyG's for convenience
        # if the features are flat.
        # For simplicity, if features are flat tensors from e3nn, you can use PyG scatter.
        aggregated_messages = torch_geometric.utils.scatter(messages, col, dim=0, dim_size=node_features.size(0), reduce="sum")

        # Update node features
        updated_node_features = self.lin_update_out(self.gate_update(
            self.tp_update(node_features, aggregated_messages)
        ))

        return updated_node_features


class RealSpaceEGNN(nn.Module):
    def __init__(self, 
                 original_atom_features_dim, # e.g., 5 for [Z, group, row, etc.]
                 ct_uae_dim=128,
                 hidden_irreps_str="128x0e + 64x1o + 32x2e", # A common mix for general features
                 n_layers=6,
                 radius=5.0, # Radius for graph construction
                 num_neighbors=None # Optional, for fixed k-NN graph
                ):
        super().__init__()
        
        # 1. Define Irreps for inputs, hidden states, and outputs
        # Input features: all scalars initially. We treat ct-UAE as scalars for simplicity here.
        # If ct-UAE has inherent geometric meaning you want to leverage, you'd define different irreps.
        self.input_node_irreps = o3.Irreps(f"{original_atom_features_dim + ct_uae_dim}x0e")
        
        # Edge features: vector (displacement) and scalar (distance)
        self.edge_irreps = o3.Irreps("1x1o + 1x0e") # 1 vector (r_vec) and 1 scalar (distance)

        # Hidden features: defined by the string input
        self.hidden_irreps = o3.Irreps(hidden_irreps_str)
        
        # Output features: we want a graph-level scalar, so the final layer should project to scalars
        # For final pooling, it's common to only take the scalar part.
        self.output_node_irreps = self.hidden_irreps # The last layer's output will be this.

        # 2. Initial Linear layer to project input features to the hidden irreps
        # This linear layer will operate on the initial scalar features and transform them
        # into the mixed irreps of the hidden layer.
        self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)

        # 3. Stack GNN layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GNNLayer(
                node_irreps_in=self.hidden_irreps,
                edge_irreps_in=self.edge_irreps,
                node_irreps_out=self.hidden_irreps,
                hidden_irreps=self.hidden_irreps # Often, hidden irreps are the same as output
            ))
            # You might add non-linearities/normalization within GNNLayer or between layers
            # For e3nn, the Gate operation handles non-linearity.

        # 4. Global Pooling and Final Projection to 512-D
        # We need an *invariant* output for the global embedding.
        # This typically means selecting only the scalar (0e) components and then summing them.
        self.final_projection_to_512 = Linear(self.output_node_irreps.filter("0e"), 512)
        # The sum of dimensions for 0e should be >= 512 for this to work directly.
        # E.g., if output_node_irreps is "128x0e + 64x1o", then 0e has 128 dimensions.
        # In this case, the Linear layer would project 128 -> 512.
        # Make sure your hidden_irreps_str leads to enough 0e features if 512 is desired from 0e.
        # Or, if 512 is the total dimension, you might sum ALL irreps and then project, but that loses equivariance.
        # The typical practice for invariant graph-level features is to pool the 0e components.

        self.radius = radius
        self.num_neighbors = num_neighbors

    def forward(self, data):
        # data.x: (N_atoms, original_atom_features_dim + ct_uae_dim)
        # data.pos: (N_atoms, 3)
        # data.edge_index: (2, N_bonds) (from PyG's radius_graph or precomputed)
        # data.batch: (N_atoms)

        # 1. Project input features to e3nn's Irreps structure
        # Need to ensure data.x is of the correct type for e3nn or convert it.
        # If it's a flat tensor, Linear handles conversion from flat to Irreps.
        x = self.initial_projection(data.x) # x is now an e3nn.Irreps-compatible tensor

        # 2. Recompute/Load edge_index and edge_attr (relative positions)
        # PyG's radius_graph is good here.
        # Note: You can precompute and cache edge_index and edge_attr to disk if needed.
        # e3nn.o3.spherical_harmonics is used to create edge features (r_vec, dist, harmonics).
        # We need: r_vec (normalized position vector) and dist (scalar distance).

        if data.edge_index is None: # If not precomputed, compute on the fly
            # For simplicity, let's assume PyG's radius_graph is used
            edge_index = torch_geometric.nn.radius_graph(data.pos, self.radius, data.batch)
            row, col = edge_index
            r_vec = data.pos[row] - data.pos[col]
            dist = r_vec.norm(dim=-1, keepdim=True)
            normalized_r_vec = r_vec / (dist + 1e-8) # Avoid division by zero
            
            # Combine normalized_r_vec (1o) and dist (0e) into e3nn edge_attr
            # This expects `e3nn.o3.spherical_harmonics` if you want higher-order edge features.
            # For simple EGNN, just pass the vector and scalar.
            edge_attr = torch.cat([normalized_r_vec, dist], dim=-1) # (N_edges, 3+1)
            # You would then convert this flat tensor to e3nn.Irreps for edge_attr.
            # e3nn.to_e3nn_tensor(edge_attr, self.edge_irreps) needs to be done.
            # This is a bit tricky; e3nn.io.extract can help if you feed the raw data.
            
            # Simplest for GNNLayer: assume edge_attr for GNNLayer is (vector, scalar)
            # r_vec (1o), dist (0e)
            edge_attr_e3nn = o3.IrrepsArray(self.edge_irreps, torch.cat([normalized_r_vec, dist], dim=-1))
            # The above is a simplification. Typically, you'd pass the actual r_vec as the 'vector'
            # and dist as the 'scalar' components of the IrrepsArray directly, not concatenate flat.
        else:
            edge_index = data.edge_index
            # Assume data.edge_attr is already an e3nn.IrrepsArray or convert it
            # This is where your preprocessing step will need to be careful with edge_attr.
            # It needs to provide (normalized_r_vec, distance) pair.
            row, col = edge_index
            r_vec = data.pos[row] - data.pos[col]
            dist = r_vec.norm(dim=-1, keepdim=True)
            normalized_r_vec = r_vec / (dist + 1e-8)
            edge_attr_e3nn = o3.IrrepsArray(self.edge_irreps, torch.cat([normalized_r_vec, dist], dim=-1))


        # 3. Pass through GNN layers
        for layer in self.layers:
            x = layer(x, data.pos, edge_index, edge_attr_e3nn) # `x` is updated node features with Irreps

        # 4. Global Pooling:
        # We need to extract the INVARIANT (0e) part of the features for global pooling.
        # `x.real` accesses the underlying torch.Tensor
        invariant_features = x.chunks_by_irreps_dim[self.output_node_irreps.index_of_l[0]] # get all 0e chunks

        # Global mean pool on the invariant features
        h_real_pooled = global_mean_pool(invariant_features, data.batch)

        # 5. Final projection to 512-D
        h_real_final = self.final_projection_to_512(h_real_pooled)
        
        return h_real_final # (B, 512)