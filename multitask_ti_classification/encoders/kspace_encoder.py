import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class KSpaceEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.conv3 = GCNConv(out_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        return global_mean_pool(x, batch)  # (batch, 64) 