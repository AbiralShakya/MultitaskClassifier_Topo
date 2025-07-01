import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.cgcnn_encoder import CGCNNEncoder
from encoders.asph_encoder import ASPHEncoder
from encoders.kspace_encoder import KSpaceEncoder
from encoders.kspace_transformer_encoder import KSpaceTransformerEncoder
from encoders.kspace_physics_encoder import KSpacePhysicsEncoder

class ImprovedAttentionFusion(nn.Module):
    def __init__(self, feature_dim=128, num_modalities=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # Modality-specific projections to ensure consistent feature spaces
        self.modality_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_modalities)
        ])
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        
        # Self-attention within each modality
        self.self_attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feature fusion with gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, num_modalities),
            nn.Sigmoid()
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, features_list):
        # Project each modality to consistent feature space
        projected_features = []
        for i, (features, proj) in enumerate(zip(features_list, self.modality_projections)):
            projected = proj(features)
            projected_features.append(projected)
        
        # Stack features: (batch_size, num_modalities, feature_dim)
        stacked_features = torch.stack(projected_features, dim=1)
        
        # Cross-modal attention
        attended_features, _ = self.cross_attention(stacked_features, stacked_features, stacked_features)
        attended_features = self.norm1(attended_features + stacked_features)  # Residual connection
        
        # Self-attention within each modality
        self_attended, _ = self.self_attention(attended_features, attended_features, attended_features)
        self_attended = self.norm2(self_attended + attended_features)  # Residual connection
        
        # Global context
        global_context = self_attended.mean(dim=1)  # (batch_size, feature_dim)
        
        # Gated fusion
        concatenated = torch.cat(projected_features, dim=1)  # (batch_size, num_modalities * feature_dim)
        gates = self.gate(concatenated)  # (batch_size, num_modalities)
        
        # Apply gates to individual features
        gated_features = []
        for i, features in enumerate(projected_features):
            gate_i = gates[:, i:i+1]  # (batch_size, 1)
            gated = features * gate_i
            gated_features.append(gated)
        
        # Final fusion
        gated_concatenated = torch.cat(gated_features, dim=1)
        fused = self.fusion_mlp(gated_concatenated)
        
        # Combine with global context
        final_features = fused + global_context
        return final_features

class HybridTopoClassifier(nn.Module):
    def __init__(self, cgcnn_node_dim, cgcnn_edge_dim, asph_dim=3115, kspace_node_dim=10, kspace_edge_dim=4, kspace_physics_dim=202):
        super().__init__()
        # Encoders with optimized dimensions
        self.cgcnn = CGCNNEncoder(cgcnn_node_dim, cgcnn_edge_dim, hidden_dim=128)
        self.asph = ASPHEncoder(input_dim=asph_dim, out_dim=128)
        self.kspace_gnn = KSpaceEncoder(kspace_node_dim, kspace_edge_dim, out_dim=128)
        self.kspace_transformer = KSpaceTransformerEncoder(kspace_node_dim, kspace_edge_dim, out_dim=128)
        self.kspace_physics = KSpacePhysicsEncoder(input_dim=kspace_physics_dim, out_dim=128)
        
        # Improved attention-based fusion
        self.fusion = ImprovedAttentionFusion(feature_dim=128, num_modalities=5)
        
        # Improved classifier with better regularization
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Modality-specific classifiers for auxiliary losses
        self.cgcnn_classifier = nn.Linear(128, 3)
        self.asph_classifier = nn.Linear(128, 3)
        self.kspace_classifier = nn.Linear(128, 3)
        
    def forward(self, crystal_graph, asph_features, kspace_graph, kspace_physics_features):
        # Encode each modality with better preprocessing
        cgcnn_out = self.cgcnn(crystal_graph.x, crystal_graph.edge_index, crystal_graph.edge_attr, crystal_graph.batch)
        asph_out = self.asph(asph_features)
        kspace_gnn_out = self.kspace_gnn(kspace_graph.x, kspace_graph.edge_index, kspace_graph.edge_attr, kspace_graph.batch)
        kspace_transformer_out = self.kspace_transformer(kspace_graph.x, kspace_graph.edge_index, kspace_graph.edge_attr, kspace_graph.batch)
        
        # Better physics feature processing
        physics_vec = torch.cat([
            kspace_physics_features['decomposition_features'],
            kspace_physics_features['gap_features'],
            kspace_physics_features['dos_features'],
            kspace_physics_features['fermi_features']
        ], dim=-1)
        kspace_physics_out = self.kspace_physics(physics_vec)
        
        # Improved attention-based fusion
        fused = self.fusion([cgcnn_out, asph_out, kspace_gnn_out, kspace_transformer_out, kspace_physics_out])
        
        # Main classification
        main_logits = self.classifier(fused)
        
        # Auxiliary predictions for each modality
        cgcnn_logits = self.cgcnn_classifier(cgcnn_out)
        asph_logits = self.asph_classifier(asph_out)
        kspace_logits = self.kspace_classifier(kspace_gnn_out + kspace_transformer_out)
        
        return main_logits, cgcnn_logits, asph_logits, kspace_logits