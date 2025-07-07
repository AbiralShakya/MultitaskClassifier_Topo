'''
Enhanced MultiModalMaterialClassifier with Topological ML and Spectral Graph Features.
Combines multiple encoders, physics-aware topological ML, and spectral graph embeddings
for 3-way classification (trivial / semimetal / topological insulator) plus auxiliary tasks.
'''  
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from helper.topological_crystal_encoder import TopologicalCrystalEncoder
from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures
from encoders.ph_token_encoder import PHTokenEncoder
from src.model_w_debug import ScalarFeatureEncoder, KSpaceTransformerGNNEncoder

# Topological ML encoder
from topological_ml_encoder import (
    TopologicalMLEncoder,
    TopologicalMLEncoder2D,
    create_hamiltonian_from_features,
    compute_topological_loss
)
# Spectral graph features
from helper.graph_spectral_encoder import GraphSpectralEncoder
import helper.config as config

class EnhancedMultiModalMaterialClassifier(nn.Module):
    def __init__(
        self,
        # Feature dims
        crystal_node_feature_dim: int,
        kspace_node_feature_dim: int,
        asph_feature_dim: int,
        scalar_feature_dim: int,
        decomposition_feature_dim: int,
        # Class counts
        num_topology_classes: int = config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes: int = config.NUM_MAGNETISM_CLASSES,
        num_combined_classes: int = config.NUM_COMBINED_CLASSES,
        # Encoder params
        crystal_encoder_hidden_dim: int = 128,
        crystal_encoder_num_layers: int = 4,
        crystal_encoder_output_dim: int = 128,
        crystal_encoder_radius: float = 5.0,
        crystal_encoder_num_scales: int = 3,
        crystal_encoder_use_topological_features: bool = True,
        kspace_gnn_hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers: int = config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads: int = config.KSPACE_GNN_NUM_HEADS,
        ffnn_hidden_dims_asph: List[int] = config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar: List[int] = config.FFNN_HIDDEN_DIMS_SCALAR,
        latent_dim_gnn: int = config.LATENT_DIM_GNN,
        latent_dim_asph: int = config.LATENT_DIM_ASPH,
        latent_dim_other_ffnn: int = config.LATENT_DIM_OTHER_FFNN,
        fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,
        # Topological ML params
        use_topological_ml: bool = True,
        topological_ml_dim: int = 128,
        topological_ml_k_points: int = 32,
        topological_ml_model_type: str = "1d_a3",
        topological_ml_auxiliary_weight: float = 0.1,
    ):
        super().__init__()
        # Crystal graph encoder
        self.crystal_encoder = TopologicalCrystalEncoder(
            node_feature_dim=crystal_node_feature_dim,
            hidden_dim=crystal_encoder_hidden_dim,
            num_layers=crystal_encoder_num_layers,
            output_dim=crystal_encoder_output_dim,
            radius=crystal_encoder_radius,
            num_scales=crystal_encoder_num_scales,
            use_topological_features=crystal_encoder_use_topological_features
        )
        # k-space GNN encoder
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=kspace_node_feature_dim,
            hidden_dim=kspace_gnn_hidden_channels,
            out_channels=latent_dim_gnn,
            n_layers=kspace_gnn_num_layers,
            num_heads=kspace_gnn_num_heads
        )
        # Persistent homology token encoder
        self.asph_encoder = PHTokenEncoder(
            input_dim=asph_feature_dim,
            output_dim=latent_dim_asph
        )
        # Scalar features encoder
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=scalar_feature_dim,
            hidden_dims=ffnn_hidden_dims_scalar,
            out_channels=latent_dim_other_ffnn
        )
        # Enhanced physics features
        self.enhanced_kspace_physics_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=decomposition_feature_dim,
            gap_features_dim=config.BAND_GAP_SCALAR_DIM,
            dos_features_dim=config.DOS_FEATURE_DIM,
            fermi_features_dim=config.FERMI_FEATURE_DIM,
            output_dim=latent_dim_other_ffnn
        )
        # Spectral graph encoder
        self.spectral_encoder = GraphSpectralEncoder(
            k_eigs=config.K_LAPLACIAN_EIGS,
            hidden=config.SPECTRAL_HID
        )
        # Topological ML setup
        self.use_topological_ml = use_topological_ml
        self.topological_ml_auxiliary_weight = topological_ml_auxiliary_weight
        if use_topological_ml:
            if topological_ml_model_type == "1d_a3":
                self.topological_ml_encoder = TopologicalMLEncoder(
                    input_dim=8,
                    k_points=topological_ml_k_points,
                    hidden_dims=[64, 128, 256],
                    num_classes=num_topology_classes,
                    output_features=topological_ml_dim,
                    extract_local_features=True
                )
            elif topological_ml_model_type == "2d_a":
                k_grid = int(topological_ml_k_points ** 0.5)
                self.topological_ml_encoder = TopologicalMLEncoder2D(
                    input_dim=3,
                    k_grid=k_grid,
                    hidden_dims=[32, 64, 128],
                    num_classes=num_topology_classes,
                    output_features=topological_ml_dim,
                    extract_berry_curvature=True
                )
            else:
                raise ValueError(f"Unknown topological_ml_model_type: {topological_ml_model_type}")
        # Determine fused dimension
        base_fused = (
            crystal_encoder_output_dim + latent_dim_gnn +
            latent_dim_asph + latent_dim_other_ffnn + latent_dim_other_ffnn +
            config.SPECTRAL_HID
        )
        if use_topological_ml:
            fused_dim = base_fused + topological_ml_dim
        else:
            fused_dim = base_fused
        # Fusion MLP
        layers = []
        in_dim = fused_dim
        for h in fusion_hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(config.DROPOUT_RATE)]
            in_dim = h
        self.fusion_network = nn.Sequential(*layers)
        # Output heads
        self.combined_head      = nn.Linear(in_dim, num_combined_classes)
        self.topology_head_aux  = nn.Linear(in_dim, num_topology_classes)
        self.magnetism_head_aux = nn.Linear(in_dim, num_magnetism_classes)
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Encode modalities
        crystal_emb, topo_logits_aux, extracted_topo_features = \
            self.crystal_encoder(inputs['crystal_graph'], return_topological_logits=True)
        kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
        asph_emb   = self.asph_encoder(inputs['asph_features'])
        scalar_emb = self.scalar_encoder(inputs['scalar_features'])
        phys_emb   = self.enhanced_kspace_physics_encoder(
            decomposition_features=inputs['kspace_physics_features']['decomposition_features'],
            gap_features=inputs['kspace_physics_features'].get('gap_features'),
            dos_features=inputs['kspace_physics_features'].get('dos_features'),
            fermi_features=inputs['kspace_physics_features'].get('fermi_features')
        )
        # Spectral graph features
        spec_emb = self.spectral_encoder(
            inputs['crystal_graph'].edge_index,
            inputs['crystal_graph'].num_nodes,
            getattr(inputs['crystal_graph'], 'batch', None)
        )
        # Topological ML features
        top_ml_emb, top_ml_logits = (None, None)
        if self.use_topological_ml:
            # build Hamiltonian from fused raw features
            raw_fuse = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, phys_emb], dim=-1)
            hams = create_hamiltonian_from_features(raw_fuse, k_points=inputs.get('k_points', config.K_LAPLACIAN_EIGS), model_type=getattr(self, 'topological_ml_model_type', '1d_a3'))
            topo_out = self.topological_ml_encoder(hams)
            top_ml_emb    = topo_out.get('topological_features')
            top_ml_logits = topo_out.get('topological_logits') or topo_out.get('chern_logits')
        # Concatenate all
        features = [crystal_emb, kspace_emb, asph_emb, scalar_emb, phys_emb, spec_emb]
        if top_ml_emb is not None:
            features.append(top_ml_emb)
        x = torch.cat(features, dim=-1)
        # Fuse
        fused = self.fusion_network(x)
        # Heads
        combined_logits     = self.combined_head(fused)
        magnetism_logits    = self.magnetism_head_aux(fused)
        # Choose primary topology logits
        if top_ml_logits is not None:
            topology_logits_primary = top_ml_logits
            topology_logits_auxiliary = topo_logits_aux
        else:
            topology_logits_primary = topo_logits_aux
            topology_logits_auxiliary = None
        return {
            'combined_logits': combined_logits,
            'topology_logits_primary': topology_logits_primary,
            'topology_logits_auxiliary': topology_logits_auxiliary,
            'magnetism_logits_aux': magnetism_logits,
            'extracted_topo_features': extracted_topo_features,
            'topological_ml_features': top_ml_emb
        }
    
    def compute_enhanced_loss(self,
            predictions: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        # combined
        if 'combined' in targets:
            losses['combined_loss'] = F.cross_entropy(predictions['combined_logits'], targets['combined'])
        # topology primary
        if 'topology' in targets:
            losses['topology_loss'] = F.cross_entropy(predictions['topology_logits_primary'], targets['topology'])
            if predictions.get('topology_logits_auxiliary') is not None:
                losses['topology_aux_loss'] = F.cross_entropy(
                    predictions['topology_logits_auxiliary'], targets['topology']
                )
        # magnetism
        if 'magnetism' in targets:
            losses['magnetism_loss'] = F.cross_entropy(predictions['magnetism_logits_aux'], targets['magnetism'])
        # topological ML
        if self.use_topological_ml and 'topology' in targets:
            topo_ml_preds = {'topological_logits': predictions['topology_logits_primary']}
            if predictions.get('topological_ml_features') is not None:
                topo_ml_preds['topological_features'] = predictions['topological_ml_features']
            topo_ml_losses = compute_topological_loss(
                topo_ml_preds,
                targets['topology'],
                auxiliary_weight=self.topological_ml_auxiliary_weight
            )
            losses.update({
                'topo_ml_main': topo_ml_losses['main_loss'],
                'topo_ml_feature': topo_ml_losses['feature_loss'],
                'topo_ml_total': topo_ml_losses['total_loss']
            })
        losses['total_loss'] = sum(losses.values())
        return losses



# """
# Enhanced MultiModalMaterialClassifier with Topological ML Encoder.
# Integrates arXiv:1805.10503v2 approach for physics-aware topological classification.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Any, List
# import warnings

# # Import your existing components
# from model import (
#     TopologicalCrystalEncoder, 
#     KSpaceTransformerGNNEncoder,
#     PHTokenEncoder, 
#     ScalarFeatureEncoder,
#     EnhancedKSpacePhysicsFeatures
# )

# # Import the new topological ML encoder
# from topological_ml_encoder import (
#     TopologicalMLEncoder,
#     TopologicalMLEncoder2D,
#     create_hamiltonian_from_features,
#     compute_topological_loss
# )

# from helper.graph_spectral_encoder import GraphSpectralEncoder

# import helper.config as config

# class EnhancedMultiModalMaterialClassifier(nn.Module):
#     """
#     Enhanced multi-modal classifier with topological ML encoder.
#     Combines your existing encoders with physics-aware topological learning.
#     """
    
#     def __init__(
#         self,
#         # Your existing parameters
#         crystal_node_feature_dim: int,
#         kspace_node_feature_dim: int,
#         asph_feature_dim: int,
#         scalar_feature_dim: int,
#         decomposition_feature_dim: int,
        
#         num_topology_classes: int = config.NUM_TOPOLOGY_CLASSES,
#         num_magnetism_classes: int = config.NUM_MAGNETISM_CLASSES,
#         num_combined_classes: int = config.NUM_COMBINED_CLASSES,
        
#         # Crystal encoder params
#         crystal_encoder_hidden_dim: int = 128,
#         crystal_encoder_num_layers: int = 4,
#         crystal_encoder_output_dim: int = 128,
#         crystal_encoder_radius: float = 5.0,
#         crystal_encoder_num_scales: int = 3,
#         crystal_encoder_use_topological_features: bool = True,
        
#         # K-space encoder params
#         kspace_gnn_hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
#         kspace_gnn_num_layers: int = config.GNN_NUM_LAYERS,
#         kspace_gnn_num_heads: int = config.KSPACE_GNN_NUM_HEADS,
        
#         # FFNN params
#         ffnn_hidden_dims_asph: List[int] = config.FFNN_HIDDEN_DIMS_ASPH,
#         ffnn_hidden_dims_scalar: List[int] = config.FFNN_HIDDEN_DIMS_SCALAR,
        
#         # Shared params
#         latent_dim_gnn: int = config.LATENT_DIM_GNN,
#         latent_dim_asph: int = config.LATENT_DIM_ASPH,
#         latent_dim_other_ffnn: int = config.LATENT_DIM_OTHER_FFNN,
#         fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,
        
#         # NEW: Topological ML encoder params
#         use_topological_ml: bool = True,
#         topological_ml_dim: int = 128,
#         topological_ml_k_points: int = 32,
#         topological_ml_model_type: str = "1d_a3",  # "1d_a3" or "2d_a"
#         topological_ml_auxiliary_weight: float = 0.1,
#     ):
#         super().__init__()
        
#         # Your existing encoders
#         self.crystal_encoder = TopologicalCrystalEncoder(
#             node_feature_dim=crystal_node_feature_dim,
#             hidden_dim=crystal_encoder_hidden_dim,
#             num_layers=crystal_encoder_num_layers,
#             output_dim=crystal_encoder_output_dim,
#             radius=crystal_encoder_radius,
#             num_scales=crystal_encoder_num_scales,
#             use_topological_features=crystal_encoder_use_topological_features
#         )
        
#         self.kspace_encoder = KSpaceTransformerGNNEncoder(
#             node_feature_dim=kspace_node_feature_dim,
#             hidden_dim=kspace_gnn_hidden_channels,
#             out_channels=latent_dim_gnn,
#             n_layers=kspace_gnn_num_layers,
#             num_heads=kspace_gnn_num_heads
#         )
        
#         self.asph_encoder = PHTokenEncoder(
#             input_dim=asph_feature_dim,
#             hidden_dims=ffnn_hidden_dims_asph,
#             out_channels=latent_dim_asph
#         )
        
#         self.scalar_encoder = ScalarFeatureEncoder(
#             input_dim=scalar_feature_dim,
#             hidden_dims=ffnn_hidden_dims_scalar,
#             out_channels=latent_dim_other_ffnn
#         )
        
#         self.enhanced_kspace_physics_encoder = EnhancedKSpacePhysicsFeatures(
#             decomposition_dim=decomposition_feature_dim,
#             gap_features_dim=config.BAND_GAP_SCALAR_DIM,
#             dos_features_dim=config.DOS_FEATURE_DIM,
#             fermi_features_dim=config.FERMI_FEATURE_DIM,
#             output_dim=latent_dim_other_ffnn
#         )
        
#         self.spectral_encoder = GraphSpectralEncoder(
#             k_eigs=config.K_LAPLACIAN_EIGS,
#             hidden=config.SPECTRAL_HID
#         )
        
#         # NEW: Topological ML encoder
#         self.use_topological_ml = use_topological_ml
#         self.topological_ml_model_type = topological_ml_model_type
#         self.topological_ml_auxiliary_weight = topological_ml_auxiliary_weight
        
#         if use_topological_ml:
#             if topological_ml_model_type == "1d_a3":
#                 self.topological_ml_encoder = TopologicalMLEncoder(
#                     input_dim=8,  # Real/Im parts of 2x2 matrix D(k)
#                     k_points=topological_ml_k_points,
#                     hidden_dims=[64, 128, 256],
#                     num_classes=num_topology_classes,
#                     output_features=topological_ml_dim,
#                     extract_local_features=True
#                 )
#             elif topological_ml_model_type == "2d_a":
#                 k_grid = int(topological_ml_k_points ** 0.5)
#                 self.topological_ml_encoder = TopologicalMLEncoder2D(
#                     input_dim=3,  # hx, hy, hz components
#                     k_grid=k_grid,
#                     hidden_dims=[32, 64, 128],
#                     num_classes=num_topology_classes,
#                     output_features=topological_ml_dim,
#                     extract_berry_curvature=True
#                 )
#             else:
#                 raise ValueError(f"Unknown topological_ml_model_type: {topological_ml_model_type}")
        
#         # Calculate total fused dimension
#         base_fused_dim = (crystal_encoder_output_dim + latent_dim_gnn + 
#                          latent_dim_asph + latent_dim_other_ffnn + latent_dim_other_ffnn)
        
#         if use_topological_ml:
#             total_fused_dim = base_fused_dim + topological_ml_dim
#         else:
#             total_fused_dim = base_fused_dim
        
#         self.expected_total_fused_dim = total_fused_dim
#         self.fusion_hidden_dims = fusion_hidden_dims
#         self.fusion_network = None
        
#         # Store output dimensions
#         self.num_combined_classes = num_combined_classes
#         self.num_topology_classes = num_topology_classes
#         self.num_magnetism_classes = num_magnetism_classes
        
#         # Initialize output heads as None - will be created dynamically
#         self.combined_head = None
#         self.topology_head_aux = None
#         self.magnetism_head_aux = None
    
#     def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
#         """Enhanced forward pass with topological ML encoder."""
        
#         # Your existing encoders
#         crystal_emb, topo_logits_aux, extracted_topo_features = \
#             self.crystal_encoder(inputs['crystal_graph'], return_topological_logits=True)
        
#         kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
#         asph_emb = self.asph_encoder(inputs['asph_features'])
#         scalar_emb = self.scalar_encoder(inputs['scalar_features'])
        
#         enhanced_kspace_physics_emb = self.enhanced_kspace_physics_encoder(
#             decomposition_features=inputs['kspace_physics_features']['decomposition_features'],
#             gap_features=inputs['kspace_physics_features'].get('gap_features'),
#             dos_features=inputs['kspace_physics_features'].get('dos_features'),
#             fermi_features=inputs['kspace_physics_features'].get('fermi_features')
#         )
        
#         # NEW: Topological ML encoder
#         topological_ml_features = None
#         topological_ml_logits = None
#         local_features = None
        
#         if self.use_topological_ml:
#             # Combine existing features to create Hamiltonians
#             combined_features = torch.cat([
#                 crystal_emb, kspace_emb, asph_emb, scalar_emb, enhanced_kspace_physics_emb
#             ], dim=-1)
            
#             # Create Hamiltonians from features
#             hamiltonians = create_hamiltonian_from_features(
#                 combined_features, 
#                 k_points=32,  # You can make this configurable
#                 model_type=self.topological_ml_model_type
#             )
            
#             # Get topological predictions
#             topological_outputs = self.topological_ml_encoder(hamiltonians)
            
#             topological_ml_features = topological_outputs['topological_features']
#             local_features = topological_outputs.get('local_features')
            
#             # Extract appropriate logits
#             if 'topological_logits' in topological_outputs:
#                 topological_ml_logits = topological_outputs['topological_logits']
#             elif 'chern_logits' in topological_outputs:
#                 topological_ml_logits = topological_outputs['chern_logits']
        
#         # Concatenate all embeddings for fusion
#         embeddings_to_concat = [
#             crystal_emb, kspace_emb, asph_emb, scalar_emb, enhanced_kspace_physics_emb
#         ]
        
#         if topological_ml_features is not None:
#             embeddings_to_concat.append(topological_ml_features)
        
#         combined_emb = torch.cat(embeddings_to_concat, dim=-1)
        
#         # Debug: Print actual dimensions
#         print(f"DEBUG - Enhanced Embedding dimensions:")
#         print(f"  crystal_emb: {crystal_emb.shape}")
#         print(f"  kspace_emb: {kspace_emb.shape}")
#         print(f"  asph_emb: {asph_emb.shape}")
#         print(f"  scalar_emb: {scalar_emb.shape}")
#         print(f"  enhanced_kspace_physics_emb: {enhanced_kspace_physics_emb.shape}")
#         if topological_ml_features is not None:
#             print(f"  topological_ml_features: {topological_ml_features.shape}")
#         print(f"  combined_emb: {combined_emb.shape}")
#         print(f"  Expected total_fused_dim: {self.expected_total_fused_dim}")
        
#         # Create fusion network dynamically if not already created
#         if self.fusion_network is None:
#             actual_fused_dim = combined_emb.shape[-1]
#             print(f"Creating enhanced fusion network with actual dimension: {actual_fused_dim}")
            
#             fusion_layers = []
#             in_dim_fusion = actual_fused_dim
#             for h_dim in self.fusion_hidden_dims:
#                 fusion_layers.append(nn.Linear(in_dim_fusion, h_dim))
#                 fusion_layers.append(nn.BatchNorm1d(h_dim))
#                 fusion_layers.append(nn.ReLU())
#                 fusion_layers.append(nn.Dropout(p=config.DROPOUT_RATE))
#                 in_dim_fusion = h_dim 
#             self.fusion_network = nn.Sequential(*fusion_layers).to(combined_emb.device)
            
#             combined_input_dim = base_feat_dim + config.SPECTRAL_HID

#             # 3) Heads
#             self.combined_head       = nn.Linear(combined_input_dim, config.NUM_COMBINED_CLASSES)
#             self.topology_head_aux   = nn.Linear(combined_input_dim, config.NUM_TOPOLOGY_CLASSES)
#             self.magnetism_head_aux  = nn.Linear(combined_input_dim, config.NUM_MAGNETISM_CLASSES)

#                 # Create output heads
#             # self.combined_head = nn.Linear(in_dim_fusion, self.num_combined_classes).to(combined_emb.device)
#             # self.topology_head_aux = nn.Linear(in_dim_fusion, self.num_topology_classes).to(combined_emb.device)
#             # self.magnetism_head_aux = nn.Linear(in_dim_fusion, self.num_magnetism_classes).to(combined_emb.device)
        
#         fused_output = self.fusion_network(combined_emb)
        
#         # Task-specific logits
#         combined_logits = self.combined_head(fused_output)
#         magnetism_logits_aux = self.magnetism_head_aux(fused_output)
        
#         # Combine topological logits from different sources
#         if topological_ml_logits is not None:
#             # Use topological ML logits as primary, crystal encoder as auxiliary
#             topology_logits_primary = topological_ml_logits
#             topology_logits_auxiliary = topo_logits_aux
#         else:
#             # Fall back to crystal encoder only
#             topology_logits_primary = topo_logits_aux
#             topology_logits_auxiliary = None
        
#         return {
#             'combined_logits': combined_logits,
#             'topology_logits_primary': topology_logits_primary,
#             'topology_logits_auxiliary': topology_logits_auxiliary,
#             'magnetism_logits_aux': magnetism_logits_aux,
#             'extracted_topo_features': extracted_topo_features,
#             'local_features': local_features,  # For interpretability
#             'topological_ml_features': topological_ml_features
#         }
    
#     def compute_enhanced_loss(
#         self, 
#         predictions: Dict[str, torch.Tensor], 
#         targets: Dict[str, torch.Tensor]
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Compute enhanced loss with topological ML components.
        
#         Args:
#             predictions: Output from forward pass
#             targets: Dictionary with 'combined', 'topology', 'magnetism' labels
        
#         Returns:
#             Dictionary containing all loss components
#         """
#         losses = {}
        
#         # Main classification losses
#         if 'combined' in targets:
#             losses['combined_loss'] = F.cross_entropy(
#                 predictions['combined_logits'], targets['combined']
#             )
        
#         if 'topology' in targets:
#             # Use primary topological logits
#             losses['topology_loss'] = F.cross_entropy(
#                 predictions['topology_logits_primary'], targets['topology']
#             )
            
#             # Add auxiliary topology loss if available
#             if predictions['topology_logits_auxiliary'] is not None:
#                 losses['topology_aux_loss'] = F.cross_entropy(
#                     predictions['topology_logits_auxiliary'], targets['topology']
#                 )
        
#         if 'magnetism' in targets:
#             losses['magnetism_loss'] = F.cross_entropy(
#                 predictions['magnetism_logits_aux'], targets['magnetism']
#             )
        
#         # NEW: Topological ML loss
#         if self.use_topological_ml and 'topology' in targets:
#             # Create topological ML predictions dict
#             topo_ml_predictions = {
#                 'topological_logits': predictions['topology_logits_primary']
#             }
#             if predictions['topological_ml_features'] is not None:
#                 topo_ml_predictions['topological_features'] = predictions['topological_ml_features']
            
#             topo_ml_losses = compute_topological_loss(
#                 topo_ml_predictions, 
#                 targets['topology'],
#                 auxiliary_weight=self.topological_ml_auxiliary_weight
#             )
            
#             losses.update({
#                 'topological_ml_total': topo_ml_losses['total_loss'],
#                 'topological_ml_main': topo_ml_losses['main_loss'],
#                 'topological_ml_feature': topo_ml_losses['feature_loss']
#             })
        
#         # Total loss
#         total_loss = sum(losses.values())
#         losses['total_loss'] = total_loss
        
#         return losses 