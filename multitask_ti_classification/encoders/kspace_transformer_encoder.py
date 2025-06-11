import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import TransformerConv # A basic GNN Transformer layer

class KSpaceCTGNN(nn.Module):
    def __init__(self, input_node_dim, hidden_dim=512, n_layers=4, num_heads=8):
        super().__init__()
        # Initial projection for k-point features (irrep_id, energy_rank, pos_encodings)
        self.initial_projection = nn.Linear(input_node_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            # TransformerConv is a good starting point for graph transformers
            # It performs self-attention over neighbors.
            self.layers.append(TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=0.1,
                beta=True # Uses skip connection from original node features
            ))
            # You might add LayerNorm and ReLU/GELU after each layer
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())

    def forward(self, data):
        # Expects a Data object with data.x, data.edge_index, data.batch
        x = self.initial_projection(data.x)
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerConv):
                x = layer(x, data.edge_index)
            else: # LayerNorm, ReLU
                x = layer(x)

        return global_mean_pool(x, data.batch) # Output size (B, hidden_dim)