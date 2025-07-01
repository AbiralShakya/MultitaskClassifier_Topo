import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class KSpaceTransformerEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim=64, model_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(model_dim, out_dim)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # x: (num_nodes, node_dim) - node features from graph
        # Transform to sequence format for transformer
        x = self.input_proj(x)  # (num_nodes, model_dim)
        
        # Group by batch to create sequences
        batch_size = batch.max().item() + 1
        sequences = []
        
        for i in range(batch_size):
            mask = (batch == i)
            seq = x[mask]  # (num_nodes_in_batch_i, model_dim)
            sequences.append(seq)
        
        # Pad sequences to same length for transformer
        max_len = max(seq.size(0) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            if seq.size(0) < max_len:
                padding = torch.zeros(max_len - seq.size(0), seq.size(1), device=seq.device)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        
        # Stack into batch: (batch_size, max_len, model_dim)
        x_batch = torch.stack(padded_sequences)
        
        # Apply transformer
        x_batch = self.transformer(x_batch)  # (batch_size, max_len, model_dim)
        
        # Global pooling: take mean across sequence length
        x_batch = x_batch.mean(dim=1)  # (batch_size, model_dim)
        
        # Final projection
        out = self.out_proj(x_batch)  # (batch_size, out_dim)
        
        return out