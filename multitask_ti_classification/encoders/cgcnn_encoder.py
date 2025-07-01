import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class CGCNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
    def forward(self, x, edge_index, edge_attr, batch):
        x = torch.sigmoid(self.bn1(self.conv1(x, edge_index)))
        x = torch.sigmoid(self.bn2(self.conv2(x, edge_index)))
        x = torch.sigmoid(self.bn3(self.conv3(x, edge_index)))
        return global_mean_pool(x, batch)  # (batch, 128) 