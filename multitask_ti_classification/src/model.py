import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, TransformerConv
import torch_geometric
from torch_geometric.data import Data as PyGData
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear, Irreps
from e3nn.nn import Gate
from typing import List, Dict, Any, Tuple

import helper.config as config

class EGNNLayer(nn.Module):
    """
    A simplified EGNN-like layer using e3nn components.
    Fixed version with proper Gate initialization and tensor handling.
    """
    def __init__(self, node_irreps_in: Irreps, edge_irreps_in: Irreps, hidden_irreps: Irreps):
        super().__init__()
        self.node_irreps_in = node_irreps_in
        self.edge_irreps_in = edge_irreps_in
        self.hidden_irreps = hidden_irreps

        # TP for messages (from node_j and edge_attr) -> message irreps
        self.tp_messages_ij = FullyConnectedTensorProduct(
            node_irreps_in, edge_irreps_in, hidden_irreps,
        )

        # --- FIXED: Proper Gate initialization for older e3nn versions ---
        # Get the actual output irreps from tensor product
        tp_out_irreps = self.tp_messages_ij.irreps_out
        
        # Separate scalars and non-scalars from the tensor product output
        scalar_parts = []
        gated_parts = []
        
        for mul, irrep in tp_out_irreps:
            if irrep.l == 0:  # scalar
                scalar_parts.append((mul, irrep))
            else:  # non-scalar (will be gated)
                gated_parts.append((mul, irrep))
        
        # Create irreps objects
        irreps_scalars = Irreps(scalar_parts) if scalar_parts else Irreps("0x0e")
        irreps_gated = Irreps(gated_parts) if gated_parts else Irreps("0x1o")
        
        # Create gates - one scalar gate per gated irrep multiplicity
        gate_parts = []
        for mul, irrep in irreps_gated:
            if irrep.l > 0:  # Only create gates for non-scalars
                gate_parts.append((mul, o3.Irrep(0, 1)))  # (mul, 0e) gates
        
        irreps_gates = Irreps(gate_parts) if gate_parts else Irreps("0x0e")
        
        # Create activation lists - must match the number of irrep types, not total multiplicity
        act_scalars = [F.silu] * len(irreps_scalars) if len(irreps_scalars) > 0 else []
        act_gates = [F.sigmoid] * len(irreps_gates) if len(irreps_gates) > 0 else []
        
        try:
            self.gate_messages = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=act_scalars,
                irreps_gates=irreps_gates,
                act_gates=act_gates,
                irreps_gated=irreps_gated
            )
            self.linear_messages_out = Linear(self.gate_messages.irreps_out, hidden_irreps)
        except Exception as e:
            print(f"Warning: Gate initialization failed: {e}")
            print(f"TP output irreps: {tp_out_irreps}")
            print(f"Scalars: {irreps_scalars}, Gates: {irreps_gates}, Gated: {irreps_gated}")
            # # Fallback: use a simple linear layer
            self.gate_messages = Linear(tp_out_irreps, hidden_irreps)
            self.linear_messages_out = nn.Identity()

        # TP for update (from node_i and aggregated_messages) -> new node irreps
        self.tp_update = FullyConnectedTensorProduct(
            node_irreps_in, hidden_irreps, hidden_irreps,
        )
        
        # Gate for update - same approach with older e3nn API
        tp_update_out_irreps = self.tp_update.irreps_out
        
        # Separate scalars and non-scalars for update gate
        update_scalar_parts = []
        update_gated_parts = []
        
        for mul, irrep in tp_update_out_irreps:
            if irrep.l == 0:  # scalar
                update_scalar_parts.append((mul, irrep))
            else:  # non-scalar (will be gated)
                update_gated_parts.append((mul, irrep))
        
        # Create irreps objects for update
        update_irreps_scalars = Irreps(update_scalar_parts) if update_scalar_parts else Irreps("0x0e")
        update_irreps_gated = Irreps(update_gated_parts) if update_gated_parts else Irreps("0x1o")
        
        # Create gates for update
        update_gate_parts = []
        for mul, irrep in update_irreps_gated:
            if irrep.l > 0:  # Only create gates for non-scalars
                update_gate_parts.append((mul, o3.Irrep(0, 1)))  # (mul, 0e) gates
        
        update_irreps_gates = Irreps(update_gate_parts) if update_gate_parts else Irreps("0x0e")
        
        # Create activation lists for update
        update_act_scalars = [F.silu] * len(update_irreps_scalars) if len(update_irreps_scalars) > 0 else []
        update_act_gates = [F.sigmoid] * len(update_irreps_gates) if len(update_irreps_gates) > 0 else []
        
        try:
            self.gate_update = Gate(
                irreps_scalars=update_irreps_scalars,
                act_scalars=update_act_scalars,
                irreps_gates=update_irreps_gates,
                act_gates=update_act_gates,
                irreps_gated=update_irreps_gated
            )
            self.linear_update_out = Linear(self.gate_update.irreps_out, node_irreps_in)
        except Exception as e:
            print(f"Warning: Update gate initialization failed: {e}")
            print(f"Update TP output irreps: {tp_update_out_irreps}")
            # Fallback: use a simple linear layer
            self.gate_update = Linear(tp_update_out_irreps, node_irreps_in)
            self.linear_update_out = nn.Identity()
        
    def forward(self, node_features, edge_index: torch.Tensor, edge_attr_tensor: torch.Tensor, 
                node_attr_scalar_raw: torch.Tensor):
        
        row, col = edge_index

        # Ensure edge_attr_tensor has correct shape and irreps structure
        # edge_attr_tensor should be [num_edges, 4] where first 3 are vector (1o) and last 1 is scalar (0e)
        
        # Debug: Print shapes for troubleshooting
        # print(f"node_features shape: {node_features.shape}")
        # print(f"edge_attr_tensor shape: {edge_attr_tensor.shape}")
        # print(f"node_features irreps: {node_features.irreps if hasattr(node_features, 'irreps') else 'No irreps'}")
        
        # 1. Message passing
        try:
            messages_tp_output = self.tp_messages_ij(node_features[col], edge_attr_tensor)
        except Exception as e:
            print(f"Error in tensor product: {e}")
            print(f"node_features[col] shape: {node_features[col].shape}")
            print(f"edge_attr_tensor shape: {edge_attr_tensor.shape}")
            raise
        
        # Apply gate with error handling
        try:
            messages_gated = self.gate_messages(messages_tp_output)
            messages_from_j = self.linear_messages_out(messages_gated)
        except Exception as e:
            print(f"Error in gate_messages: {e}")
            print(f"messages_tp_output shape: {messages_tp_output.shape}")
            print(f"Expected irreps: {self.tp_messages_ij.irreps_out}")
            # Fallback: use the tensor product output directly
            messages_from_j = messages_tp_output
            if messages_from_j.shape[-1] != self.hidden_irreps.dim:
                # Project to correct dimension
                if not hasattr(self, 'fallback_linear_msg'):
                    self.fallback_linear_msg = nn.Linear(
                        messages_from_j.shape[-1], 
                        self.hidden_irreps.dim
                    ).to(messages_from_j.device)
                messages_from_j = self.fallback_linear_msg(messages_from_j)

        # 2. Aggregation (sum messages for each node)
        aggregated_messages = torch_geometric.utils.scatter(
            messages_from_j, row, dim=0, dim_size=node_features.size(0), reduce="sum"
        )

        # 3. Update (combine current node features with aggregated messages)
        try:
            updated_node_features_tp_output = self.tp_update(node_features, aggregated_messages)
            updated_node_features_gated = self.gate_update(updated_node_features_tp_output)
            updated_node_features_temp = self.linear_update_out(updated_node_features_gated)
        except Exception as e:
            print(f"Error in update: {e}")
            # Fallback: simple linear combination
            if not hasattr(self, 'fallback_linear_update'):
                combined_dim = node_features.shape[-1] + aggregated_messages.shape[-1]
                self.fallback_linear_update = nn.Linear(
                    combined_dim, 
                    node_features.shape[-1]
                ).to(node_features.device)
            
            combined = torch.cat([node_features, aggregated_messages], dim=-1)
            updated_node_features_temp = self.fallback_linear_update(combined)
        
        return node_features + updated_node_features_temp


class RealSpaceEGNNEncoder(nn.Module):
    """
    EGNN encoder for real-space atomic crystal graphs.
    Fixed version with better error handling and tensor management.
    """
    def __init__(self, 
                 node_input_scalar_dim: int,
                 hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
                 n_layers: int = 6,
                 radius: float = 5.0
                ):
        super().__init__()
        
        self.node_input_scalar_dim = node_input_scalar_dim
        
        self.input_node_irreps = Irreps(f"{node_input_scalar_dim}x0e")
        self.edge_irreps = Irreps("1x1o + 1x0e")  # 3D vector + 1D scalar
        self.hidden_irreps = Irreps(hidden_irreps_str)
        
        self.initial_projection = Linear(self.input_node_irreps, self.hidden_irreps)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EGNNLayer(
                node_irreps_in=self.hidden_irreps,
                edge_irreps_in=self.edge_irreps,
                hidden_irreps=self.hidden_irreps
            ))

        # Extract only scalar irreps for final output
        scalar_irreps_list = [(mul, irrep) for mul, irrep in self.hidden_irreps if irrep.l == 0]
        if scalar_irreps_list:
            self.scalar_irreps = Irreps(scalar_irreps_list)
        else:
            # Fallback if no scalars
            self.scalar_irreps = Irreps("1x0e")
            
        output_irreps = Irreps(f"{config.LATENT_DIM_GNN}x0e")
        self.final_linear_0e = Linear(self.scalar_irreps, output_irreps)

        self.radius = radius

    def forward(self, data: PyGData) -> torch.Tensor:
        x_raw_scalars = data.x
        
        # Project to e3nn format
        x_e3nn = self.initial_projection(x_raw_scalars) 

        # Build edge graph
        edge_index = torch_geometric.nn.radius_graph(data.pos, self.radius, data.batch)
        
        if edge_index.size(1) == 0:
            # Handle case with no edges
            print("Warning: No edges found in graph, using zero tensor")
            batch_size = data.batch.max().item() + 1 if data.batch is not None else 1
            return torch.zeros(batch_size, config.LATENT_DIM_GNN, device=x_raw_scalars.device)
        
        row, col = edge_index
        
        # Compute edge attributes
        r_vec = data.pos[row] - data.pos[col]
        dist = r_vec.norm(dim=-1, keepdim=True)
        
        # Avoid division by zero
        normalized_r_vec = r_vec / (dist + 1e-8) 
        
        # Create edge attributes: [normalized_vector (3D), distance (1D)]
        # This should match edge_irreps = "1x1o + 1x0e" (3 + 1 = 4 dimensions)
        edge_attr_tensor = torch.cat([normalized_r_vec, dist], dim=-1)
        
        # Verify edge attribute dimensions
        assert edge_attr_tensor.shape[-1] == 4, f"Expected 4D edge attributes, got {edge_attr_tensor.shape[-1]}D"

        # Pass through EGNN layers
        current_node_features_e3nn = x_e3nn
        for i, layer in enumerate(self.layers):
            try:
                current_node_features_e3nn = layer(
                    current_node_features_e3nn, 
                    edge_index, 
                    edge_attr_tensor,
                    x_raw_scalars
                )
            except Exception as e:
                print(f"Error in layer {i}: {e}")
                raise

        # Extract scalar features for global pooling
        scalar_features = []
        start_idx = 0
        
        # Handle both e3nn tensors and regular tensors
        if hasattr(current_node_features_e3nn, 'irreps'):
            irreps_iter = current_node_features_e3nn.irreps
        else:
            # Fallback: assume all features are scalars
            irreps_iter = self.hidden_irreps
        
        for mul, irrep in irreps_iter:
            end_idx = start_idx + mul * irrep.dim
            if irrep.l == 0:  # Only scalar (l=0) features
                scalar_features.append(current_node_features_e3nn[:, start_idx:end_idx])
            start_idx = end_idx
        
        if scalar_features:
            invariant_features_per_node = torch.cat(scalar_features, dim=1)
        else:
            # Fallback: use first few dimensions as scalars
            scalar_dim = min(current_node_features_e3nn.shape[-1], self.scalar_irreps.dim)
            invariant_features_per_node = current_node_features_e3nn[:, :scalar_dim]
        
        # Global pooling
        graph_embedding_tensor = global_mean_pool(invariant_features_per_node, data.batch)
        
        # Final projection
        try:
            final_embedding = self.final_linear_0e(graph_embedding_tensor)
        except Exception as e:
            print(f"Error in final projection: {e}")
            print(f"graph_embedding_tensor shape: {graph_embedding_tensor.shape}")
            print(f"Expected input dim: {self.scalar_irreps.dim}")
            
            # Fallback projection
            if graph_embedding_tensor.shape[-1] != self.scalar_irreps.dim:
                if not hasattr(self, 'fallback_final_linear'):
                    self.fallback_final_linear = nn.Linear(
                        graph_embedding_tensor.shape[-1], 
                        config.LATENT_DIM_GNN
                    ).to(graph_embedding_tensor.device)
                final_embedding = self.fallback_final_linear(graph_embedding_tensor)
            else:
                final_embedding = self.final_linear_0e(graph_embedding_tensor)
        
        return final_embedding

class KSpaceTransformerGNNEncoder(nn.Module):
    """
    Graph Transformer (TransformerConv) encoder for k-space topology graphs.
    """
    def __init__(self, node_feature_dim: int, hidden_dim: int, out_channels: int, 
                 n_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.initial_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        current_in_channels = hidden_dim 

        for i in range(n_layers):
            out_channels_per_head = hidden_dim 

            self.layers.append(TransformerConv(
                in_channels=current_in_channels, 
                out_channels=out_channels_per_head,
                heads=num_heads,
                dropout=config.DROPOUT_RATE,
                beta=True
            ))
            self.bns.append(nn.BatchNorm1d(out_channels_per_head * num_heads))
            
            current_in_channels = out_channels_per_head * num_heads

        self.final_projection = nn.Linear(current_in_channels, out_channels)


    def forward(self, data: PyGData) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.initial_projection(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index) # Output of TransformerConv is (N_nodes, out_channels_per_head * num_heads)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)

        pooled_x = global_mean_pool(x, batch)
        final_embedding = self.final_projection(pooled_x) # Apply final projection
        
        return final_embedding

# --- 3. ASPH Encoder ---

class PHTokenEncoder(nn.Module):
    """
    Encoder for Atom-Specific Persistent Homology (ASPH) features.
    A simple FFNN to process the feature vector.
    """
    def __init__(self, input_dim: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, out_channels * 2),
            nn.BatchNorm1d(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

# --- 4. Scalar Features Encoder ---

class ScalarFeatureEncoder(nn.Module):
    """
    FFNN encoder for combined scalar features (band_rep_features + metadata features).
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], out_channels: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=config.DROPOUT_RATE))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, out_channels))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

# --- 5. Decomposition Features Encoder ---

class DecompositionFeatureEncoder(nn.Module):
    """
    FFNN encoder for decomposition branches features.
    """
    def __init__(self, input_dim: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, out_channels * 2), 
            nn.BatchNorm1d(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(out_channels * 2, out_channels), 
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

# --- 6. Main Multi-Modal Classifier ---

class MultiModalMaterialClassifier(nn.Module):
    """
    Multi-modal, multi-task classifier for materials.
    Combines Real-space EGNN, K-space Transformer GNN, ASPH, and Scalar features.
    """
    def __init__(
        self,
        crystal_node_feature_dim: int,
        kspace_node_feature_dim: int,
        asph_feature_dim: int,
        scalar_feature_dim: int,
        decomposition_feature_dim: int,
        num_topology_classes: int,
        num_magnetism_classes: int,
        
        # Encoder specific params
        egnn_hidden_irreps_str: str = "64x0e + 32x1o + 16x2e",
        egnn_num_layers: int = 6,
        egnn_radius: float = 5.0,
        
        kspace_gnn_hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers: int = config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads: int = 8,
        
        ffnn_hidden_dims_asph: List[int] = config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar: List[int] = config.FFNN_HIDDEN_DIMS_SCALAR,
        
        # Shared fusion params
        latent_dim_gnn: int = config.LATENT_DIM_GNN,
        # Use specific latent dims for FFNN types
        latent_dim_asph: int = config.LATENT_DIM_ASPH,
        latent_dim_other_ffnn: int = config.LATENT_DIM_OTHER_FFNN,
        
        fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,
    ):
        super().__init__()

        self.crystal_encoder = RealSpaceEGNNEncoder(
            node_input_scalar_dim=crystal_node_feature_dim,
            hidden_irreps_str=egnn_hidden_irreps_str,
            n_layers=egnn_num_layers,
            radius=egnn_radius,
        )
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=kspace_node_feature_dim,
            hidden_dim=kspace_gnn_hidden_channels,
            out_channels=latent_dim_gnn,
            n_layers=kspace_gnn_num_layers,
            num_heads=kspace_gnn_num_heads
        )
        self.asph_encoder = PHTokenEncoder(
            input_dim=asph_feature_dim,
            out_channels=latent_dim_asph
        )
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=scalar_feature_dim,
            hidden_dims=ffnn_hidden_dims_scalar,
            out_channels=latent_dim_other_ffnn
        )
        self.decomposition_encoder = DecompositionFeatureEncoder(
            input_dim=decomposition_feature_dim,
            out_channels=latent_dim_other_ffnn
        )

        total_fused_dim = (latent_dim_gnn * 2) + latent_dim_asph + (latent_dim_other_ffnn * 2) 

        fusion_layers = []
        in_dim_fusion = total_fused_dim
        for h_dim in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(in_dim_fusion, h_dim))
            fusion_layers.append(nn.BatchNorm1d(h_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(p=config.DROPOUT_RATE))
            in_dim_fusion = h_dim 
        self.fusion_network = nn.Sequential(*fusion_layers)

        self.topology_head = nn.Linear(in_dim_fusion, num_topology_classes)
        self.magnetism_head = nn.Linear(in_dim_fusion, num_magnetism_classes)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        crystal_emb = self.crystal_encoder(inputs['crystal_graph'])
        kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
        asph_emb = self.asph_encoder(inputs['asph_features'])
        scalar_emb = self.scalar_encoder(inputs['scalar_features'])
        decomposition_emb = self.decomposition_encoder(inputs['kspace_physics_features']['decomposition_features']) 

        combined_emb = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, decomposition_emb], dim=-1)

        fused_output = self.fusion_network(combined_emb)

        topology_logits = self.topology_head(fused_output)
        magnetism_logits = self.magnetism_head(fused_output)

        return {
            'topology_logits': topology_logits,
            'magnetism_logits': magnetism_logits
        }