"""
Enhanced Integrated Model combining all improvements from the Nature paper:
1. Enhanced atomic features with Voronoi graph construction
2. Multi-scale attention networks
3. Enhanced topological loss with consistency constraints
4. Persistent homology integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import numpy as np

# Import our enhanced components
from encoders.enhanced_node_features import EnhancedAtomicFeatures, EnhancedCGCNNEncoder
from encoders.enhanced_graph_construction import EnhancedGraphConstructor
from encoders.multi_scale_attention import MultiScaleGraphAttention, TopologicalFeatureExtractor
from helper.enhanced_topological_loss import EnhancedTopologicalLoss, FocalLoss
from encoders.ph_token_encoder import PHTokenEncoder
from src.model_w_debug import KSpaceTransformerGNNEncoder, ScalarFeatureEncoder
from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures
from helper.gpu_spectral_encoder import GPUSpectralEncoder
import helper.config as config


class EnhancedIntegratedMaterialClassifier(nn.Module):
    """
    Enhanced integrated classifier combining all improvements:
    - Rich atomic features (65D)
    - Voronoi graph construction
    - Multi-scale attention
    - Topological consistency
    - Persistent homology
    """
    
    def __init__(
        self,
        # Feature dimensions
        crystal_node_feature_dim: int = 65,  # Enhanced atomic features
        kspace_node_feature_dim: int = 10,
        scalar_feature_dim: int = 4763,
        decomposition_feature_dim: int = 2,
        asph_feature_dim: int = 512,
        # Model parameters
        num_topology_classes: int = 2,  # Binary classification
        hidden_dim: int = 256,
        num_attention_heads: int = 8,
        num_layers: int = 4,
        dropout_rate: float = 0.3,
        # Enhanced features
        use_enhanced_features: bool = True,
        use_voronoi_construction: bool = True,
        use_persistent_homology: bool = True,
        use_topological_consistency: bool = True,
    ):
        super().__init__()
        
        self.use_enhanced_features = use_enhanced_features
        self.use_voronoi_construction = use_voronoi_construction
        self.use_persistent_homology = use_persistent_homology
        self.use_topological_consistency = use_topological_consistency
        self.hidden_dim = hidden_dim
        self.num_topology_classes = num_topology_classes
        
        # Enhanced atomic features
        if use_enhanced_features:
            self.atomic_features = EnhancedAtomicFeatures()
            self.crystal_encoder = EnhancedCGCNNEncoder(
                node_dim=crystal_node_feature_dim,
                edge_dim=15,  # Enhanced edge features
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
        else:
            # Fallback to multi-scale attention
            self.crystal_encoder = MultiScaleGraphAttention(
                node_dim=crystal_node_feature_dim,
                edge_dim=1,
                hidden_dim=hidden_dim,
                num_heads=num_attention_heads,
                num_layers=num_layers
            )
        
        # K-space encoder
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=kspace_node_feature_dim,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
            n_layers=num_layers,
            num_heads=num_attention_heads
        )
        
        # Scalar feature encoder
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=scalar_feature_dim,
            hidden_dims=[hidden_dim * 2, hidden_dim],
            out_channels=hidden_dim
        )
        
        # Enhanced k-space physics encoder
        self.physics_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=decomposition_feature_dim,
            gap_features_dim=config.BAND_GAP_SCALAR_DIM,
            dos_features_dim=config.DOS_FEATURE_DIM,
            fermi_features_dim=config.FERMI_FEATURE_DIM,
            output_dim=hidden_dim
        )
        
        # Spectral encoder
        self.spectral_encoder = GPUSpectralEncoder(
            k_eigs=config.K_LAPLACIAN_EIGS,
            hidden=hidden_dim // 2
        )
        
        # Persistent homology encoder
        if use_persistent_homology:
            self.ph_encoder = PHTokenEncoder(
                input_dim=asph_feature_dim,
                hidden_dims=[hidden_dim, hidden_dim // 2],
                output_dim=hidden_dim // 2
            )
        
        # Topological feature extractor
        if use_topological_consistency:
            self.topo_extractor = TopologicalFeatureExtractor(hidden_dim)
        
        # Calculate fusion input dimension
        fusion_input_dim = hidden_dim * 4  # crystal + kspace + scalar + physics
        if hasattr(self, 'spectral_encoder'):
            fusion_input_dim += hidden_dim // 2  # spectral
        if use_persistent_homology:
            fusion_input_dim += hidden_dim // 2  # persistent homology
        
        # Multi-scale fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification heads
        self.main_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_topology_classes)
        )
        
        # Auxiliary classification head for consistency
        if use_topological_consistency:
            self.aux_classifier = nn.Sequential(
                nn.Linear(6, 32),  # Topological features
                nn.ReLU(),
                nn.Linear(32, num_topology_classes)
            )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1),
            nn.Sigmoid()
        )
        
        # Enhanced loss function
        self.loss_fn = EnhancedTopologicalLoss(
            alpha=1.0,      # Main classification
            beta=0.3,       # Auxiliary classification
            gamma=0.2,      # Topological consistency
            delta=0.1,      # Confidence regularization
            epsilon=0.1     # Feature regularization
        )
        
        # Focal loss for class imbalance
        if hasattr(config, 'USE_FOCAL_LOSS') and config.USE_FOCAL_LOSS:
            focal_alpha = getattr(config, 'FOCAL_LOSS_ALPHA', [1.0, 2.0])
            focal_gamma = getattr(config, 'FOCAL_LOSS_GAMMA', 2.0)
            self.focal_loss = FocalLoss(
                alpha=torch.tensor(focal_alpha),
                gamma=focal_gamma
            )
        else:
            self.focal_loss = None
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with all improvements"""
        
        # Extract crystal graph features with enhanced atomic features
        crystal_graph = inputs['crystal_graph']
        
        if (self.use_enhanced_features and 
            hasattr(crystal_graph, 'atomic_numbers') and 
            hasattr(self, 'atomic_features')):
            try:
                # Use enhanced atomic features
                enhanced_node_features = self.atomic_features.encode_atomic_features(
                    crystal_graph.atomic_numbers
                )
                crystal_features = self.crystal_encoder(
                    enhanced_node_features,
                    crystal_graph.edge_index,
                    crystal_graph.edge_attr,
                    crystal_graph.batch,
                    atomic_numbers=crystal_graph.atomic_numbers
                )
            except Exception as e:
                print(f"Enhanced crystal encoding failed: {e}, using standard features")
                # Fallback to standard features
                crystal_features = self.crystal_encoder(
                    crystal_graph.x,
                    crystal_graph.edge_index,
                    crystal_graph.edge_attr,
                    crystal_graph.batch
                )
        else:
            # Use standard features
            crystal_features = self.crystal_encoder(
                crystal_graph.x,
                crystal_graph.edge_index,
                crystal_graph.edge_attr,
                crystal_graph.batch
            )
        
        # K-space features
        kspace_features = self.kspace_encoder(inputs['kspace_graph'])
        
        # Scalar features
        scalar_features = self.scalar_encoder(inputs['scalar_features'])
        
        # Physics features
        physics_features = self.physics_encoder(
            decomposition_features=inputs['kspace_physics_features']['decomposition_features'],
            gap_features=inputs['kspace_physics_features'].get('gap_features'),
            dos_features=inputs['kspace_physics_features'].get('dos_features'),
            fermi_features=inputs['kspace_physics_features'].get('fermi_features')
        )
        
        # Spectral features
        try:
            spectral_features = self.spectral_encoder(
                crystal_graph.edge_index,
                crystal_graph.num_nodes,
                getattr(crystal_graph, 'batch', None)
            )
        except Exception as e:
            print(f"Warning: Spectral encoding failed: {e}")
            spectral_features = torch.zeros(
                crystal_features.shape[0], 
                self.hidden_dim // 2, 
                device=crystal_features.device
            )
        
        # Collect features for fusion
        features_list = [crystal_features, kspace_features, scalar_features, physics_features]
        
        if hasattr(self, 'spectral_encoder'):
            features_list.append(spectral_features)
        
        # Persistent homology features
        if self.use_persistent_homology and 'asph_features' in inputs:
            ph_features = self.ph_encoder(inputs['asph_features'])
            features_list.append(ph_features)
        
        # Fuse all features
        fused_features = torch.cat(features_list, dim=1)
        fused_features = self.fusion_network(fused_features)
        
        # Main classification
        main_logits = self.main_classifier(fused_features)
        
        # Prepare outputs
        outputs = {
            'logits': main_logits,
            'graph_features': fused_features
        }
        
        # Topological consistency features
        if self.use_topological_consistency:
            topo_features = self.topo_extractor(fused_features)
            outputs['topo_features'] = topo_features
            
            # Auxiliary classification based on topological features
            aux_logits = self.aux_classifier(topo_features['combined'])
            outputs['aux_logits'] = aux_logits
        
        # Confidence estimation
        confidence = self.confidence_head(fused_features)
        outputs['confidence'] = confidence
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Compute enhanced loss with all consistency checks"""
        
        # Use focal loss if available
        if self.focal_loss is not None:
            main_loss = self.focal_loss(outputs['logits'], targets)
            outputs_for_enhanced_loss = outputs.copy()
            outputs_for_enhanced_loss['main_loss'] = main_loss
            return self.loss_fn(outputs_for_enhanced_loss, targets)
        else:
            return self.loss_fn(outputs, targets)
    
    def predict_with_confidence(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Make predictions with confidence scores"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            
            # Get predictions
            probs = F.softmax(outputs['logits'], dim=1)
            predictions = torch.argmax(probs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            
            # Combine model confidence with prediction confidence
            combined_confidence = outputs['confidence'].squeeze() * max_probs
            
            return {
                'predictions': predictions,
                'probabilities': probs,
                'confidence': combined_confidence,
                'model_confidence': outputs['confidence'].squeeze(),
                'prediction_confidence': max_probs
            }


def create_enhanced_model(config_override: Optional[Dict] = None) -> EnhancedIntegratedMaterialClassifier:
    """Factory function to create enhanced model with config"""
    
    # Default configuration
    model_config = {
        'crystal_node_feature_dim': getattr(config, 'CRYSTAL_NODE_FEATURE_DIM', 65),
        'kspace_node_feature_dim': getattr(config, 'KSPACE_GRAPH_NODE_FEATURE_DIM', 10),
        'scalar_feature_dim': getattr(config, 'SCALAR_TOTAL_DIM', 4763),
        'decomposition_feature_dim': getattr(config, 'DECOMPOSITION_FEATURE_DIM', 2),
        'asph_feature_dim': getattr(config, 'ASPH_FEATURE_DIM', 512),
        'num_topology_classes': getattr(config, 'NUM_TOPOLOGY_CLASSES', 2),
        'hidden_dim': 256,
        'num_attention_heads': getattr(config, 'crystal_encoder_num_attention_heads', 8),
        'num_layers': 4,
        'dropout_rate': getattr(config, 'DROPOUT_RATE', 0.3),
        'use_enhanced_features': getattr(config, 'crystal_encoder_use_enhanced_features', True),
        'use_voronoi_construction': getattr(config, 'crystal_encoder_use_voronoi', True),
        'use_persistent_homology': True,
        'use_topological_consistency': True,
    }
    
    # Override with provided config
    if config_override:
        model_config.update(config_override)
    
    return EnhancedIntegratedMaterialClassifier(**model_config)


# Backward compatibility wrapper
class EnhancedMultiModalMaterialClassifier(EnhancedIntegratedMaterialClassifier):
    """Backward compatibility wrapper for existing training code"""
    
    def __init__(self, **kwargs):
        # Map old parameter names to new ones
        param_mapping = {
            'crystal_encoder_hidden_dim': 'hidden_dim',
            'crystal_encoder_num_layers': 'num_layers',
            'crystal_encoder_output_dim': 'hidden_dim',
            'kspace_gnn_num_heads': 'num_attention_heads',
            'fusion_hidden_dims': None,  # Handled internally now
        }
        
        # Filter and map parameters
        new_kwargs = {}
        for key, value in kwargs.items():
            if key in param_mapping:
                new_key = param_mapping[key]
                if new_key is not None:
                    new_kwargs[new_key] = value
            else:
                new_kwargs[key] = value
        
        super().__init__(**new_kwargs)