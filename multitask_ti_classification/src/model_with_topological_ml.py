''' 
src/model_with_topological_ml.py
EnhancedMultiModalMaterialClassifier with Topological ML and Spectral Graph Features.
Combines multiple encoders, physics-aware topological ML, spectral embeddings,
and supports 3-way classification (trivial / semimetal / topological insulator) plus auxiliary tasks.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

# Core encoders
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
        # Crystal encoder params
        crystal_encoder_hidden_dim: int = 128,
        crystal_encoder_num_layers: int = 4,
        crystal_encoder_output_dim: int = 128,
        crystal_encoder_radius: float = 5.0,
        crystal_encoder_num_scales: int = 3,
        crystal_encoder_use_topological_features: bool = True,
        # k-space GNN params
        kspace_gnn_hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers: int = config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads: int = config.KSPACE_GNN_NUM_HEADS,
        latent_dim_gnn: int = config.LATENT_DIM_GNN,
        # ASPH & scalar encoder dims
        latent_dim_asph: int = config.LATENT_DIM_ASPH,
        latent_dim_other_ffnn: int = config.LATENT_DIM_OTHER_FFNN,
        # Fusion MLP
        fusion_hidden_dims: List[int] = config.FUSION_HIDDEN_DIMS,
        dropout_rate: float = config.DROPOUT_RATE,
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
        # ASPH persistent-homology token encoder
        self.asph_encoder = PHTokenEncoder(
            input_dim=asph_feature_dim,
            output_dim=latent_dim_asph
        )
        # Scalar feature encoder
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=scalar_feature_dim,
            hidden_dims=config.FFNN_HIDDEN_DIMS_SCALAR,
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
                # Compute fused dimension using init parameters
        base_dim = (
            crystal_encoder_output_dim +  # size of crystal encoder output
            latent_dim_gnn +               # size of k-space GNN output
            latent_dim_asph +              # size of ASPH token encoder output
            latent_dim_other_ffnn +        # size of scalar feature encoder output
            latent_dim_other_ffnn +        # size of physics encoder output
            config.SPECTRAL_HID            # size of spectral graph embedding
        )
        if use_topological_ml:
            base_dim += topological_ml_dim
        if use_topological_ml:
            base_dim += topological_ml_dim
        # Fusion MLP
        fusion_layers = []
        in_dim = base_dim
        for h in fusion_hidden_dims:
            fusion_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_dim = h
        self.fusion_network = nn.Sequential(*fusion_layers)
        # Output heads
        self.combined_head      = nn.Linear(in_dim, num_combined_classes)
        self.topology_head_aux  = nn.Linear(in_dim, num_topology_classes)
        self.magnetism_head_aux = nn.Linear(in_dim, num_magnetism_classes)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Encode each modality
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
        spec_emb = self.spectral_encoder(
            inputs['crystal_graph'].edge_index,
            inputs['crystal_graph'].num_nodes,
            getattr(inputs['crystal_graph'], 'batch', None)
        )
        # Topological ML features
        top_ml_emb, top_ml_logits = (None, None)
        if self.use_topological_ml:
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
        if top_ml_logits is not None:
            topology_logits_primary   = top_ml_logits
            topology_logits_auxiliary = topo_logits_aux
        else:
            topology_logits_primary   = topo_logits_aux
            topology_logits_auxiliary = None
        return {
            'combined_logits': combined_logits,
            'topology_logits_primary': topology_logits_primary,
            'topology_logits_auxiliary': topology_logits_auxiliary,
            'magnetism_logits_aux': magnetism_logits,
            'extracted_topo_features': extracted_topo_features,
            'topological_ml_features': top_ml_emb
        }

    def compute_enhanced_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        # Main combined loss
        losses['combined_loss'] = F.cross_entropy(
            predictions['combined_logits'], targets['combined']
        )
        # Topology losses
        losses['topology_loss'] = F.cross_entropy(
            predictions['topology_logits_primary'], targets['topology']
        )
        if predictions.get('topology_logits_auxiliary') is not None:
            losses['topology_aux_loss'] = F.cross_entropy(
                predictions['topology_logits_auxiliary'], targets['topology']
            )
        # Magnetism aux
        losses['magnetism_loss'] = F.cross_entropy(
            predictions['magnetism_logits_aux'], targets['magnetism']
        )
        # Topological ML losses
        if self.use_topological_ml:
            topo_ml_preds = {
                'topological_logits': predictions['topology_logits_primary']
            }
            if predictions.get('topological_ml_features') is not None:
                topo_ml_preds['topological_features'] = predictions['topological_ml_features']
            topo_ml_losses = compute_topological_loss(
                topo_ml_preds, targets['topology'], auxiliary_weight=self.topological_ml_auxiliary_weight
            )
            losses.update({
                'topo_ml_main': topo_ml_losses['main_loss'],
                'topo_ml_feature': topo_ml_losses['feature_loss'],
                'topo_ml_total': topo_ml_losses['total_loss']
            })
        losses['total_loss'] = sum(losses.values())
        return losses
