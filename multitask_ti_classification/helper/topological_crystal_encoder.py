import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing
import torch_geometric
import math

class MultiScaleEGNNConv(MessagePassing):
    """
    Enhanced EGNN convolution with multiple interaction ranges for capturing
    both local and long-range topological features.
    """
    def __init__(self, in_channels, out_channels, edge_dim=4, num_scales=3):
        super().__init__(aggr='add') # Messages from each scale are aggregated (summed)
        
        self.in_channels = in_channels
        self.out_channels = out_channels # This is the target output dimension for the layer
        self.num_scales = num_scales
        
        # Each message network should produce a vector of `out_channels` dimension.
        # The input to each message network is (x_i || x_j || edge_attr || scale_info)
        # Input dim: (in_channels + in_channels + edge_dim + 1)
        message_input_dim = 2 * in_channels + edge_dim + 1

        self.message_networks = nn.ModuleList()
        for _ in range(num_scales):
            self.message_networks.append(nn.Sequential(
                nn.Linear(message_input_dim, out_channels * 2), # Expand for capacity
                nn.LayerNorm(out_channels * 2),
                nn.SiLU(),
                nn.Linear(out_channels * 2, out_channels) # Output dimension is out_channels
            ))

        # Attention mechanism for scale fusion.
        # It takes concatenated messages from all scales. Each message is `out_channels`.
        # So, input to attention is (out_channels * num_scales).
        self.scale_attention = nn.Sequential(
            nn.Linear(out_channels * num_scales, out_channels // 4), 
            nn.SiLU(),
            nn.Linear(out_channels // 4, num_scales), # Output 'num_scales' attention weights
            nn.Softmax(dim=-1) # Softmax across scales
        )
        
        # Update network with topological bias
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels * 2), # Input is (x_enhanced || weighted_messages)
            nn.LayerNorm(out_channels * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels)
        )
        
        # Layer normalization for the final output
        self.norm = nn.LayerNorm(out_channels)
        
        # Position encoding for topological awareness
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, in_channels // 2), # Input 3D position
            nn.SiLU(),
            nn.Linear(in_channels // 2, in_channels) # Output matches node features
        )
        
    def forward(self, x, edge_index, edge_attr, pos, scale_factors=None):
        if scale_factors is None:
            scale_factors = [1.0, 2.0, 4.0]  # Default multi-scale factors
            
        # Add positional encoding
        pos_enc = self.pos_encoder(pos)
        x_enhanced = x + pos_enc # Add position features to node features
        
        # Multi-scale message passing
        scale_messages = [] # List to hold messages from each scale. Each will be [N_nodes, out_channels]
        
        for i, scale in enumerate(scale_factors):
            # Concatenate scale information to edge attributes for the message network
            scale_info = torch.full((edge_attr.size(0), 1), scale, device=edge_attr.device)
            edge_attr_scaled = torch.cat([edge_attr, scale_info], dim=-1)
            
            # Propagate messages using the scale-specific network
            messages_for_this_scale = self.propagate(edge_index, x=x_enhanced, edge_attr=edge_attr_scaled, 
                                                     message_net=self.message_networks[i])
            scale_messages.append(messages_for_this_scale)
        
        # Combine all scale messages for attention calculation
        # This will be [N_nodes, out_channels * num_scales]
        combined_messages_for_attention = torch.cat(scale_messages, dim=-1)
        
        # Compute attention weights for each scale
        attention_weights = self.scale_attention(combined_messages_for_attention) # [N_nodes, num_scales]
        
        # Apply attention weights to each scale message and sum them up
        # Resulting weighted_messages will be [N_nodes, out_channels]
        weighted_messages = torch.zeros_like(scale_messages[0]) # Initialize with shape [N_nodes, out_channels]
        for i, msg in enumerate(scale_messages):
            # msg: [N_nodes, out_channels]
            # attention_weights[:, i].unsqueeze(-1): [N_nodes, 1]
            # Element-wise multiply, then add to weighted_messages
            weighted_messages += msg * attention_weights[:, i].unsqueeze(-1)
        
        # Update node features with residual connection
        # Input to update_mlp is [x_enhanced || weighted_messages]
        # x_enhanced: [N_nodes, in_channels]
        # weighted_messages: [N_nodes, out_channels]
        out = self.update_mlp(torch.cat([x_enhanced, weighted_messages], dim=-1))
        
        # Apply residual connection and layer normalization
        # Ensure residual connection is compatible (dimensions match)
        if x_enhanced.size(-1) == out.size(-1):
            out = out + x_enhanced
            
        return self.norm(out)
    
    def message(self, x_i, x_j, edge_attr, message_net):
        # x_i: Features of node i, x_j: Features of node j
        # edge_attr: Attributes of edge (j,i)
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return message_net(message_input)


class TopologicalFeatureExtractor(nn.Module):
    """
    Specialized module for extracting topological invariants and features
    that are relevant for TI/TSM/trivial classification.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Chern number and Z2 invariant estimators
        self.chern_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Bounded output for Chern-like features (e.g., -1 to 1)
        )
        
        self.z2_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 4),  # 4 Z2 invariants
            nn.Sigmoid()  # Binary-like features (0 to 1)
        )
        
        # Band gap and Dirac point estimators
        self.gap_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # Positive gap values
        )
        
        # Symmetry breaking indicators
        self.symmetry_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 8)  # Various symmetry indicators
        )
        
    def forward(self, x):
        chern_features = self.chern_estimator(x)
        z2_features = self.z2_estimator(x)
        gap_features = self.gap_estimator(x)
        symmetry_features = self.symmetry_estimator(x)
        
        # Concatenate all extracted topological features
        return torch.cat([chern_features, z2_features, gap_features, symmetry_features], dim=-1)


class AdaptivePooling(nn.Module):
    """
    Adaptive pooling that combines multiple pooling strategies
    to capture different aspects of the crystal structure.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Attention-based pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1) # Outputs a single attention score per node
        )
        
        # Pooling fusion network
        self.pool_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # Takes concatenation of mean, max, attention pooled features
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
    def forward(self, x, batch):
        # Standard pooling methods
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        
        # Attention-weighted pooling
        # attention_weights: [N_nodes, 1] - scores for each node
        attention_weights = F.softmax(self.attention_pool(x), dim=0) # Softmax over all nodes in the batch (dim 0)
        
        # Apply weights and then sum-pool per graph
        attention_pool = torch_geometric.utils.scatter(
            x * attention_weights, batch, dim=0, reduce='sum'
        )
        
        # Combine pooling strategies
        combined = torch.cat([mean_pool, max_pool, attention_pool], dim=-1)
        return self.pool_fusion(combined)


class TopologicalCrystalEncoder(nn.Module):
    """
    Enhanced crystal encoder specifically designed for topological classification.
    Incorporates multi-scale interactions, topological feature extraction,
    and adaptive pooling for better TI/TSM/trivial discrimination.
    """
    def __init__(self, 
                 node_feature_dim: int,
                 hidden_dim: int = 256,  # Increased for better representation
                 num_layers: int = 6,    # More layers for complex patterns
                 output_dim: int = 128,  # Main embedding output dimension for the crystal graph
                 radius: float = 4.0,    # Larger radius for long-range interactions
                 num_scales: int = 3,    # Multi-scale interactions
                 use_topological_features: bool = True):
        super().__init__()
        
        self.radius = radius
        self.use_topological_features = use_topological_features
        
        # Enhanced input projection with batch normalization
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-scale EGNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(MultiScaleEGNNConv(hidden_dim, hidden_dim, num_scales=num_scales))
        
        # Topological feature extractor (always instantiated, its output is conditionally used)
        self.topo_extractor = TopologicalFeatureExtractor(hidden_dim)
        self.extracted_topo_feature_dim = 1 + 4 + 1 + 8  # Chern (1) + Z2 (4) + Gap (1) + Symmetry (8) = 14
        
        # Adaptive pooling
        self.adaptive_pool = AdaptivePooling(hidden_dim)
        
        # Final projection: takes pooled_x (hidden_dim) AND extracted_topo_features (if used)
        final_input_dim = hidden_dim + (self.extracted_topo_feature_dim if use_topological_features else 0)

        self.output_proj = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim), # First layer in output projection
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim) # Final output dimension for the main crystal embedding
        )
        
        # Specialized topological classifier head (for auxiliary task logits)
        self.topological_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, 3)  # TI, TSM, Trivial (3 classes)
        )
        
    def forward(self, data, return_topological_logits=False):
        x, pos, batch = data.x, data.pos, data.batch
        
        # Enhanced input projection
        x = self.input_proj(x)
        
        # Build radius graph for inter-atomic interactions
        edge_index = torch_geometric.nn.radius_graph(pos, self.radius, batch)
        
        # Handle empty graphs (no edges) gracefully
        if edge_index.size(1) == 0:
            batch_size = batch.max().item() + 1 if batch is not None else 1
            zero_output_emb = torch.zeros(batch_size, self.output_proj[-1].out_features, device=x.device)
            zero_topo_logits = torch.zeros(batch_size, 3, device=x.device) 
            zero_extracted_topo_features = torch.zeros(batch_size, self.extracted_topo_feature_dim, device=x.device) 

            if return_topological_logits:
                return zero_output_emb, zero_topo_logits, zero_extracted_topo_features
            return zero_output_emb, None, zero_extracted_topo_features # Main output, None for logits, raw topo features
        
        # Compute enhanced edge attributes for EGNN layers
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)
        edge_dir = edge_vec / (edge_dist + 1e-8)
        
        edge_dist_norm = edge_dist / (self.radius + 1e-8)  # Normalized distance
        edge_attr = torch.cat([edge_dir, edge_dist_norm], dim=-1)
        
        # Apply multi-scale EGNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, pos) # Pass 'pos' for internal positional encoding in MultiScaleEGNNConv
        
        # Adaptive pooling to get graph-level features
        pooled_x = self.adaptive_pool(x, batch)
        
        # Extract topological features using the dedicated extractor
        extracted_topo_features = self.topo_extractor(pooled_x) 
        
        # Combine pooled features with extracted topological features for the final output projection
        if self.use_topological_features:
            final_features_for_output_proj = torch.cat([pooled_x, extracted_topo_features], dim=-1)
        else:
            final_features_for_output_proj = pooled_x
        
        # Final projection for the main crystal embedding output
        output_embedding = self.output_proj(final_features_for_output_proj)
        
        # Prepare outputs based on 'return_topological_logits' flag
        if return_topological_logits:
            topo_logits = self.topological_head(output_embedding) # Generate logits for the auxiliary topology task
            # Return main embedding, auxiliary topo logits, and raw extracted topo features
            return output_embedding, topo_logits, extracted_topo_features 
        
        # Default return: main embedding, None for auxiliary logits, and raw extracted topo features
        return output_embedding, None, extracted_topo_features
