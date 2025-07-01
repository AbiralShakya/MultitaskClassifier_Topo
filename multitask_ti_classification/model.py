import torch
import torch.nn as nn
from encoders.cgcnn_encoder import CGCNNEncoder
from encoders.asph_encoder import ASPHEncoder
from encoders.kspace_encoder import KSpaceEncoder
from encoders.kspace_transformer_encoder import KSpaceTransformerEncoder
from encoders.kspace_physics_encoder import KSpacePhysicsEncoder

class HybridTopoClassifier(nn.Module):
    def __init__(self, cgcnn_node_dim, cgcnn_edge_dim, asph_dim=3115, kspace_node_dim=10, kspace_edge_dim=4, kspace_physics_dim=202):
        super().__init__()
        self.cgcnn = CGCNNEncoder(cgcnn_node_dim, cgcnn_edge_dim, hidden_dim=128)
        self.asph = ASPHEncoder(input_dim=asph_dim, out_dim=64)
        self.kspace_gnn = KSpaceEncoder(kspace_node_dim, kspace_edge_dim, out_dim=64)
        self.kspace_transformer = KSpaceTransformerEncoder(input_dim=kspace_node_dim, out_dim=64)
        self.kspace_physics = KSpacePhysicsEncoder(input_dim=kspace_physics_dim, out_dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(128+64+64+64+64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    def forward(self, crystal_graph, asph_features, kspace_graph, kspace_physics_features):
        cgcnn_out = self.cgcnn(crystal_graph.x, crystal_graph.edge_index, crystal_graph.edge_attr, crystal_graph.batch)
        asph_out = self.asph(asph_features)
        kspace_gnn_out = self.kspace_gnn(kspace_graph.x, kspace_graph.edge_index, kspace_graph.edge_attr, kspace_graph.batch)
        # Transformer expects (batch, num_kpoints, node_dim)
        if kspace_graph.x.dim() == 2:
            kspace_x_seq = kspace_graph.x.unsqueeze(0)  # (1, num_kpoints, node_dim)
        else:
            kspace_x_seq = kspace_graph.x
        kspace_transformer_out = self.kspace_transformer(kspace_x_seq)
        # Physics features: concatenate all vectors
        physics_vec = torch.cat([
            kspace_physics_features['decomposition_features'],
            kspace_physics_features['gap_features'],
            kspace_physics_features['dos_features'],
            kspace_physics_features['fermi_features']
        ], dim=-1)
        kspace_physics_out = self.kspace_physics(physics_vec)
        fused = torch.cat([cgcnn_out, asph_out, kspace_gnn_out, kspace_transformer_out, kspace_physics_out], dim=1)
        return self.classifier(fused) 