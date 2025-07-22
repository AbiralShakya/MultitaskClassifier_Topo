#!/usr/bin/env python3
"""
Simple working model for binary topology classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config

class SimpleWorkingModel(nn.Module):
    """
    Simple working model that uses only the essential components
    """
    
    def __init__(
        self,
        crystal_node_feature_dim: int = 3,
        kspace_node_feature_dim: int = 10,
        scalar_feature_dim: int = 4763,
        decomposition_feature_dim: int = 2,
        num_topology_classes: int = 2,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_topology_classes = num_topology_classes
        self.hidden_dim = hidden_dim
        
        # Essential encoders only
        self._init_essential_encoders(
            crystal_node_feature_dim, kspace_node_feature_dim, 
            scalar_feature_dim, decomposition_feature_dim, hidden_dim
        )
        
        # Dynamic fusion network (will be created in forward)
        self.fusion_net = None
    
    def _init_essential_encoders(self, crystal_node_dim, kspace_node_dim, 
                                scalar_dim, decomp_dim, hidden_dim):
        """Initialize only the essential encoders"""
        
        # Crystal encoder
        from helper.topological_crystal_encoder import TopologicalCrystalEncoder
        self.crystal_encoder = TopologicalCrystalEncoder(
            node_feature_dim=crystal_node_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            output_dim=hidden_dim,
            radius=5.0,
            num_scales=3,
            use_topological_features=True
        )
        
        # K-space encoder
        from src.model_w_debug import KSpaceTransformerGNNEncoder
        self.kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=kspace_node_dim,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
            n_layers=4,
            num_heads=8
        )
        
        # Scalar encoder
        from src.model_w_debug import ScalarFeatureEncoder
        self.scalar_encoder = ScalarFeatureEncoder(
            input_dim=scalar_dim,
            hidden_dims=[hidden_dim * 2, hidden_dim],
            out_channels=hidden_dim
        )
        
        # Physics encoder
        from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures
        self.physics_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=decomp_dim,
            gap_features_dim=1,
            dos_features_dim=500,
            fermi_features_dim=1,
            output_dim=hidden_dim
        )
        
        # ASPH encoder
        from encoders.ph_token_encoder import PHTokenEncoder
        self.ph_encoder = PHTokenEncoder(
            input_dim=512,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            output_dim=hidden_dim // 2
        )
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Simple forward pass"""
        
        # Encode each modality
        crystal_emb, _, _ = self.crystal_encoder(
            inputs['crystal_graph'], return_topological_logits=False
        )
        
        kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
        
        scalar_emb = self.scalar_encoder(inputs['scalar_features'])
        
        phys_emb = self.physics_encoder(
            decomposition_features=inputs['kspace_physics_features']['decomposition_features'],
            gap_features=inputs['kspace_physics_features'].get('gap_features'),
            dos_features=inputs['kspace_physics_features'].get('dos_features'),
            fermi_features=inputs['kspace_physics_features'].get('fermi_features')
        )
        
        # ASPH features
        ph_emb = self.ph_encoder(inputs['asph_features'])
        
        # Concatenate features
        features = [crystal_emb, kspace_emb, scalar_emb, phys_emb, ph_emb]
        x = torch.cat(features, dim=-1)
        
        # Dynamic fusion network
        if self.fusion_net is None:
            input_dim = x.shape[1]
            print(f"Creating fusion network with input dimension: {input_dim}")
            self.fusion_net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_topology_classes)
            ).to(x.device)
        
        logits = self.fusion_net(x)
        
        return {'logits': logits}
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Simple cross-entropy loss"""
        return F.cross_entropy(predictions['logits'], targets, label_smoothing=0.1)


def test_simple_model():
    """Test the simple working model"""
    print("🧪 Testing Simple Working Model")
    print("=" * 40)
    
    try:
        # Create model
        model = SimpleWorkingModel()
        print("✅ Model created successfully!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create dummy data
        from torch_geometric.data import Data as PyGData
        
        # Crystal graph
        crystal_graph = PyGData(
            x=torch.randn(50, 3),
            edge_index=torch.randint(0, 50, (2, 100)),
            edge_attr=torch.randn(100, 1),
            batch=torch.zeros(50, dtype=torch.long)
        )
        
        # K-space graph
        kspace_graph = PyGData(
            x=torch.randn(30, 10),
            edge_index=torch.randint(0, 30, (2, 50)),
            edge_attr=torch.randn(50, 4),
            batch=torch.zeros(30, dtype=torch.long)
        )
        
        # Other features
        batch_size = 2
        data = {
            'crystal_graph': crystal_graph,
            'kspace_graph': kspace_graph,
            'scalar_features': torch.randn(batch_size, 4763),
            'asph_features': torch.randn(batch_size, 512),
            'kspace_physics_features': {
                'decomposition_features': torch.randn(batch_size, 2),
                'gap_features': torch.randn(batch_size, 1),
                'dos_features': torch.randn(batch_size, 500),
                'fermi_features': torch.randn(batch_size, 1)
            },
            'topology_label': torch.randint(0, 2, (batch_size,))
        }
        
        # Test forward pass
        print("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            print(f"✅ Forward pass successful! Output shape: {outputs['logits'].shape}")
        
        # Test loss
        print("Testing loss computation...")
        model.train()
        outputs = model(data)
        loss = model.compute_loss(outputs, data['topology_label'])
        print(f"✅ Loss computation successful! Loss: {loss.item():.4f}")
        
        # Test backward pass
        print("Testing backward pass...")
        loss.backward()
        print("✅ Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_model()
    if success:
        print("\n🎉 Simple working model test passed!")
        print("✅ Ready to replace the problematic model")
    else:
        print("\n❌ Simple model test failed")