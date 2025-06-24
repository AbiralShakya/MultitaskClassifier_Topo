import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, MessagePassing
from typing import Dict, Optional, List

# Assuming these are available from global config or dataset
# If these dimensions are not from config, you'll need to pass them or define
# a local config/import the main config.
import helper.config as config 

# Re-adding missing PyG Data for type hints
from torch_geometric.data import Data as PyGData


class BrillouinZoneEncoder(nn.Module):
    """
    Physics-informed encoder that respects Brillouin zone symmetries and high-symmetry points.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable embeddings for high-symmetry points (Γ, X, M, K, etc.)
        # Need to determine actual number of unique symmetry labels.
        # Max 10 symmetry points is a reasonable default if not known.
        self.symmetry_point_embedding = nn.Embedding(10, hidden_dim) 

        # MLP for k-point position encoding
        self.k_position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 3D k-space coordinates
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, k_points: torch.Tensor, symmetry_labels: Optional[torch.Tensor] = None):
        """
        Args:
            k_points: [N, 3] k-space coordinates
            symmetry_labels: [N] indices for high-symmetry points (optional)
        """
        # Position encoding for k-points
        k_pos_emb = self.k_position_encoder(k_points)

        # Add symmetry point embeddings if available
        if symmetry_labels is not None and symmetry_labels.numel() > 0: # Ensure labels are present and not empty
            sym_emb = self.symmetry_point_embedding(symmetry_labels)
            k_pos_emb = k_pos_emb + sym_emb # Element-wise addition assumes broadcasting works or shapes match

        return k_pos_emb

class BerryCurvatureLayer(MessagePassing):
    """
    Layer that incorporates Berry curvature physics for topological band structure analysis.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add') # Message aggregation: 'add', 'mean', 'max'
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Network to compute Berry connection-like features
        # Input: [x_i, x_j, k_diff] (in_channels + in_channels + 3)
        self.berry_connection_net = nn.Sequential(
            nn.Linear(2 * in_channels + 3, out_channels), 
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        # Network to compute band gap and energy difference features
        # This seems to be per-node, so it uses node features x
        self.energy_gap_net = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, out_channels) # Output should match message size for concatenation
        )

        # Combining network
        # Input: [berry_messages, energy_features] (out_channels + out_channels)
        self.combine_net = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, k_positions: torch.Tensor):
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            k_positions: K-space positions [N, 3]
        """
        # Compute Berry connection-like messages
        berry_messages = self.propagate(edge_index, x=x, k_pos=k_positions)

        # Compute energy gap features
        energy_features = self.energy_gap_net(x)

        # Combine features
        # Ensure dimensions match for concatenation. berry_messages and energy_features should be [N, out_channels]
        combined = torch.cat([berry_messages, energy_features], dim=-1)
        output = self.combine_net(combined)

        # Residual connection
        if x.size(-1) == output.size(-1):
            output = output + x
        else:
            # If dimensions don't match for residual, apply linear projection to match
            if not hasattr(self, 'residual_proj'):
                self.residual_proj = nn.Linear(output.size(-1), x.size(-1)).to(output.device)
            output = self.residual_proj(output) + x


        return output

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, k_pos_i: torch.Tensor, k_pos_j: torch.Tensor):
        """Compute messages incorporating k-space geometry."""
        # K-space distance vector (analogous to Berry connection)
        k_diff = k_pos_i - k_pos_j

        # Create message with Berry connection-like physics
        message_input = torch.cat([x_i, x_j, k_diff], dim=-1)
        return self.berry_connection_net(message_input)

class TopologicalInvariantExtractor(nn.Module):
    """
    Extracts topological invariants like Chern numbers from band structure representations.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Network to extract Chern number-like features
        self.chern_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single Chern-like invariant
        )

        # Network to extract Z2 topological invariant features
        self.z2_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 Z2 invariants for 3D
        )

        # Mirror Chern number extractor (for systems with mirror symmetry)
        self.mirror_chern_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Mirror Chern numbers
        )

    def forward(self, band_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract topological invariants from band structure features.

        Args:
            band_features: [batch_size, input_dim] global band structure features

        Returns:
            Dictionary of topological invariants
        """
        chern_features = self.chern_extractor(band_features)
        z2_features = self.z2_extractor(band_features)
        mirror_chern_features = self.mirror_chern_extractor(band_features)

        return {
            'chern_invariant': chern_features,
            'z2_invariants': z2_features,
            'mirror_chern': mirror_chern_features
        }

class PhysicsInformedKSpaceEncoder(nn.Module):
    """
    Complete physics-informed k-space encoder incorporating:
    1. Brillouin zone symmetries
    2. Berry curvature physics
    3. Topological invariant extraction
    4. Band structure connectivity
    """
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 output_dim: int = config.LATENT_DIM_GNN): # Match LATENT_DIM_GNN from config
        super().__init__()

        self.hidden_dim = hidden_dim

        # Initial projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # Brillouin zone encoder
        self.bz_encoder = BrillouinZoneEncoder(hidden_dim)

        # Berry curvature layers
        self.berry_layers = nn.ModuleList([
            BerryCurvatureLayer(hidden_dim, hidden_dim) # in_channels=hidden_dim, out_channels=hidden_dim
            for _ in range(num_layers)
        ])

        # Layer normalization for each Berry layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Topological invariant extractor
        self.topo_extractor = TopologicalInvariantExtractor(hidden_dim, hidden_dim // 2) # input_dim is graph_features (hidden_dim)

        # Band structure attention mechanism
        self.band_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Final projection
        # Total invariants: Chern (1) + Z2 (4) + Mirror Chern (2) = 7
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim + 7, output_dim),  # +7 for topological invariants
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, data: PyGData) -> torch.Tensor:
        """
        Forward pass incorporating physics-based k-space analysis.

        Args:
            data: PyG Data object with:
                - x: node features [N, node_feature_dim]
                - edge_index: edge connectivity [2, E]
                - pos: k-space positions [N, 3]
                - batch: batch assignment [N]
                - symmetry_labels: (optional) high-symmetry point labels [N]
        """
        x, edge_index, k_pos, batch = data.x, data.edge_index, data.pos, data.batch
        symmetry_labels = getattr(data, 'symmetry_labels', None) # Safely get symmetry labels

        # Initial feature projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Encode k-space positions with Brillouin zone awareness
        k_emb = self.bz_encoder(
            k_pos,
            symmetry_labels # Pass symmetry labels
        )

        # Combine node features with k-space embeddings
        x = x + k_emb

        # Apply Berry curvature layers
        for berry_layer, layer_norm in zip(self.berry_layers, self.layer_norms):
            x_new = berry_layer(x, edge_index, k_pos)
            x = layer_norm(x_new)

        # Global pooling to get per-graph features
        graph_features = global_mean_pool(x, batch)

        # Extract topological invariants
        topo_invariants = self.topo_extractor(graph_features)

        # Concatenate all topological features
        topo_features = torch.cat([
            topo_invariants['chern_invariant'],
            topo_invariants['z2_invariants'],
            topo_invariants['mirror_chern']
        ], dim=-1)

        # Apply band structure attention (self-attention on graph features)
        # MultiheadAttention expects input shape [batch_size, sequence_length, embed_dim]
        # For global graph features, sequence_length is 1
        if len(graph_features.shape) == 2:
            graph_features_seq = graph_features.unsqueeze(1)  # [batch, 1, hidden_dim]
            attended_features, _ = self.band_attention(
                graph_features_seq, graph_features_seq, graph_features_seq
            )
            attended_features = attended_features.squeeze(1)  # [batch, hidden_dim]
        else: # Handle cases where batch_size might be 1 and unsqueeze failed
            attended_features = graph_features


        # Combine attended features with topological invariants
        combined_features = torch.cat([attended_features, topo_features], dim=-1)

        # Final projection
        output = self.output_proj(combined_features)

        return output

class EnhancedKSpacePhysicsFeatures(nn.Module):
    """
    Enhanced encoder for additional k-space physics features beyond basic band structure.
    """
    def __init__(self,
                 decomposition_dim: int,
                 gap_features_dim: int = config.BAND_GAP_SCALAR_DIM, # From config
                 dos_features_dim: int = config.DOS_FEATURE_DIM,     # From config
                 fermi_features_dim: int = config.FERMI_FEATURE_DIM, # From config
                 output_dim: int = config.LATENT_DIM_OTHER_FFNN): # Match LATENT_DIM_OTHER_FFNN
        super().__init__()

        # Band gap analysis network
        # Input dim should match gap_features_dim
        self.gap_analyzer = nn.Sequential(
            nn.Linear(gap_features_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32), # Output 32
            nn.ReLU()
        )

        # Decomposition features (irreducible representations)
        # Input dim should match decomposition_dim
        self.decomp_encoder = nn.Sequential(
            nn.Linear(decomposition_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32) # Output 32
        )

        # Density of states features
        # Input dim should match dos_features_dim
        self.dos_encoder = nn.Sequential(
            nn.Linear(dos_features_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32) # Output 32
        )

        # Fermi surface features
        # Input dim should match fermi_features_dim
        self.fermi_encoder = nn.Sequential(
            nn.Linear(fermi_features_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16) # Output 16
        )

        # Final combination
        # Sum of all feature dims: 32 (decomp) + 32 (gap) + 32 (dos) + 16 (fermi) = 112
        self.final_proj = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 16, output_dim),  # Sum of all feature dims (112) -> output_dim (64)
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self,
                decomposition_features: torch.Tensor,
                gap_features: Optional[torch.Tensor] = None,
                dos_features: Optional[torch.Tensor] = None,
                fermi_features: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Process decomposition features (always available)
        decomp_emb = self.decomp_encoder(decomposition_features)

        # Process optional features with fallback to zeros
        if gap_features is not None and gap_features.numel() > 0:
            gap_emb = self.gap_analyzer(gap_features)
        else:
            gap_emb = torch.zeros(decomp_emb.size(0), 32, device=decomp_emb.device) # 32 is output dim of analyzer

        if dos_features is not None and dos_features.numel() > 0:
            dos_emb = self.dos_encoder(dos_features)
        else:
            dos_emb = torch.zeros(decomp_emb.size(0), 32, device=decomp_emb.device) # 32 is output dim of encoder

        if fermi_features is not None and fermi_features.numel() > 0:
            fermi_emb = self.fermi_encoder(fermi_features)
        else:
            fermi_emb = torch.zeros(decomp_emb.size(0), 16, device=decomp_emb.device) # 16 is output dim of encoder

        # Combine all features
        combined = torch.cat([decomp_emb, gap_emb, dos_emb, fermi_emb], dim=-1)

        return self.final_proj(combined)