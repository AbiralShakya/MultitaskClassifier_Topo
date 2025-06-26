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
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        
        base_scale_dim = out_channels // num_scales
        remainder = out_channels % num_scales

        self.message_networks = nn.ModuleList()
        for i in range(num_scales):
            current_scale_dim = base_scale_dim + (1 if i < remainder else 0)
            self.message_networks.append(nn.Sequential(
                nn.Linear(2 * in_channels + edge_dim + 1, current_scale_dim),
                nn.LayerNorm(current_scale_dim),
                nn.SiLU(),
                nn.Linear(current_scale_dim, current_scale_dim)
            ))

        # Attention mechanism for scale fusion - now expects out_channels (256)
        # Note: You had this defined twice. Keeping the one that seems more consistent.
        self.scale_attention = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4), # This will be nn.Linear(256, 64)
            nn.SiLU(),
            nn.Linear(out_channels // 4, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # Update network with topological bias
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels * 2),
            nn.LayerNorm(out_channels * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels)
        )
        
        # Layer normalization with learnable parameters
        self.norm = nn.LayerNorm(out_channels)
        
        # Position encoding for topological awareness - make it dimension-aware
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, in_channels // 2),
            nn.SiLU(),
            nn.Linear(in_channels // 2, in_channels)
        )
        
    def forward(self, x, edge_index, edge_attr, pos, scale_factors=None):
        if scale_factors is None:
            scale_factors = [1.0, 2.0, 4.0]  # Default multi-scale factors
            
        # Add positional encoding - now dimensions should match
        pos_enc = self.pos_encoder(pos)
        x_enhanced = x + pos_enc
        
        # Multi-scale message passing
        scale_messages = []
        
        for i, scale in enumerate(scale_factors):
            # Scale-specific edge filtering or weighting could be added here
            scale_info = torch.full((edge_attr.size(0), 1), scale, device=edge_attr.device)
            edge_attr_scaled = torch.cat([edge_attr, scale_info], dim=-1)
            
            # Propagate with scale-specific network
            messages = self.propagate(edge_index, x=x_enhanced, edge_attr=edge_attr_scaled, 
                                    message_net=self.message_networks[i])
            scale_messages.append(messages)
        
        # Combine multi-scale messages for attention calculation
        # Concatenate them to form a single tensor of total features, for the attention network input
        # Note: All scale_messages must have the same dimension across samples for concatenation here.
        # This is ensured by base_scale_dim logic in init.
        combined_messages_for_attention = torch.cat(scale_messages, dim=-1) # Use cat here!
        
        # Apply attention weights
        attention_weights = F.softmax(self.scale_attention(combined_messages_for_attention), dim=-1) # [N, num_scales]
        
        # Apply attention weights to individual scale messages and then sum them up
        weighted_sum_messages = torch.zeros_like(scale_messages[0]) # Initialize with shape of first message
        for i, msg in enumerate(scale_messages):
            weighted_sum_messages += msg * attention_weights[:, i].unsqueeze(-1) # [N, current_scale_dim] * [N, 1]
        
        weighted_messages = weighted_sum_messages # Final weighted sum

        # Update with residual connection
        out = self.update_mlp(torch.cat([x_enhanced, weighted_messages], dim=-1))
        
        if x_enhanced.size(-1) == out.size(-1): # Residual connection to x_enhanced
            out = out + x_enhanced
            
        return self.norm(out)
    
    def message(self, x_i, x_j, edge_attr, message_net):
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
            nn.Tanh()  # Bounded output for Chern-like features (-1 to 1 or 0 to 1, depending on desired scale)
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
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Pooling fusion weights
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
        # Ensure attention_pool is applied per node, then scatter-sum
        attention_weights = F.softmax(self.attention_pool(x), dim=0) # Softmax over nodes in the batch (dim 0)
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
                 output_dim: int = 128,  # Main embedding output dimension
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
        
        # Topological feature extractor
        self.topo_extractor = TopologicalFeatureExtractor(hidden_dim) # Always create it
        topo_feature_dim = 1 + 4 + 1 + 8  # Chern (1) + Z2 (4) + Gap (1) + Symmetry (8) = 14
        
        # Adaptive pooling
        self.adaptive_pool = AdaptivePooling(hidden_dim)
        
        # Final projection with topological features
        # Input to output_proj will be (pooled_x (hidden_dim) + topo_features (topo_feature_dim))
        final_input_dim = hidden_dim + (topo_feature_dim if use_topological_features else 0)

        self.output_proj = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim) # This is the main output embedding
        )
        
        # Optional: Add a specialized topological classifier head
        self.topological_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, 3)  # TI, TSM, Trivial - uses output_dim of the main encoder
        )
        
    def forward(self, data, return_topological_logits=False):
        x, pos, batch = data.x, data.pos, data.batch
        
        # Enhanced input projection
        x = self.input_proj(x)
        
        # Build radius graph with larger radius for topological features
        edge_index = torch_geometric.nn.radius_graph(pos, self.radius, batch)
        
        if edge_index.size(1) == 0:
            batch_size = batch.max().item() + 1 if batch is not None else 1
            # Return zeros if no edges, ensuring all expected outputs are present
            zero_output_emb = torch.zeros(batch_size, self.output_proj[-1].out_features, device=x.device)
            zero_topo_logits = torch.zeros(batch_size, 3, device=x.device) # 3 classes for topo
            # The extracted_topo_features has 14 dimensions (1 Chern + 4 Z2 + 1 Gap + 8 Symmetry)
            zero_extracted_topo_features = torch.zeros(batch_size, 14, device=x.device) 

            if return_topological_logits:
                return zero_output_emb, zero_topo_logits, zero_extracted_topo_features
            return zero_output_emb, None, zero_extracted_topo_features # Return None for topo_logits if not requested


        # Compute enhanced edge attributes
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)
        edge_dir = edge_vec / (edge_dist + 1e-8)
        
        edge_dist_norm = edge_dist / (self.radius + 1e-8)  # Normalized distance
        edge_attr = torch.cat([edge_dir, edge_dist_norm], dim=-1)
        
        # Apply multi-scale EGNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, pos) # pos is used by MultiScaleEGNNConv
        
        # Adaptive pooling
        pooled_x = self.adaptive_pool(x, batch)
        
        # Extract topological features
        extracted_topo_features = self.topo_extractor(pooled_x) # Use for regularization
        
        final_features_for_output_proj = torch.cat([pooled_x, extracted_topo_features], dim=-1)
        
        # Final projection for the main crystal embedding output
        output_embedding = self.output_proj(final_features_for_output_proj)
        
        if return_topological_logits:
            # Generate logits for the auxiliary topology task
            topo_logits = self.topological_head(output_embedding) # Use the main output_embedding for this head
            return output_embedding, topo_logits, extracted_topo_features # Main output, topo logits, raw topo features
        
        return output_embedding, None, extracted_topo_features # Main output, None for topo_logits, raw topo features