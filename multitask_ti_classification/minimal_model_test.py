#!/usr/bin/env python3
"""
Minimal model test to isolate the issue
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test individual components first
def test_individual_components():
    """Test each component individually"""
    print("🧪 Testing Individual Components")
    print("=" * 40)
    
    try:
        print("1. Testing TopologicalCrystalEncoder...")
        from helper.topological_crystal_encoder import TopologicalCrystalEncoder
        crystal_encoder = TopologicalCrystalEncoder(
            node_feature_dim=3,
            hidden_dim=256,
            num_layers=4,
            output_dim=256,
            radius=5.0,
            num_scales=3,
            use_topological_features=True
        )
        print("✅ TopologicalCrystalEncoder works")
        
        print("2. Testing KSpaceTransformerGNNEncoder...")
        from src.model_w_debug import KSpaceTransformerGNNEncoder
        kspace_encoder = KSpaceTransformerGNNEncoder(
            node_feature_dim=10,
            hidden_dim=256,
            out_channels=256,
            n_layers=4,
            num_heads=8
        )
        print("✅ KSpaceTransformerGNNEncoder works")
        
        print("3. Testing ScalarFeatureEncoder...")
        from src.model_w_debug import ScalarFeatureEncoder
        scalar_encoder = ScalarFeatureEncoder(
            input_dim=4763,
            hidden_dims=[512, 256],
            out_channels=256
        )
        print("✅ ScalarFeatureEncoder works")
        
        print("4. Testing EnhancedKSpacePhysicsFeatures...")
        from helper.kspace_physics_encoders import EnhancedKSpacePhysicsFeatures
        physics_encoder = EnhancedKSpacePhysicsFeatures(
            decomposition_dim=2,
            gap_features_dim=1,
            dos_features_dim=500,
            fermi_features_dim=1,
            output_dim=256
        )
        print("✅ EnhancedKSpacePhysicsFeatures works")
        
        print("5. Testing GPUSpectralEncoder...")
        from helper.gpu_spectral_encoder import GPUSpectralEncoder
        spectral_encoder = GPUSpectralEncoder(
            k_eigs=10,
            hidden=128
        )
        print("✅ GPUSpectralEncoder works")
        
        print("6. Testing PHTokenEncoder...")
        from encoders.ph_token_encoder import PHTokenEncoder
        ph_encoder = PHTokenEncoder(
            input_dim=512,
            hidden_dims=[256, 128],
            output_dim=128
        )
        print("✅ PHTokenEncoder works")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_model():
    """Test a minimal version of the model"""
    print("\n🔧 Testing Minimal Model")
    print("=" * 40)
    
    try:
        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Just test the crystal encoder
                from helper.topological_crystal_encoder import TopologicalCrystalEncoder
                self.crystal_encoder = TopologicalCrystalEncoder(
                    node_feature_dim=3,
                    hidden_dim=256,
                    num_layers=4,
                    output_dim=256,
                    radius=5.0,
                    num_scales=3,
                    use_topological_features=True
                )
                
                # Simple classifier
                self.classifier = nn.Linear(256, 2)
            
            def forward(self, crystal_graph):
                x, _, _ = self.crystal_encoder(crystal_graph, return_topological_logits=False)
                return self.classifier(x)
        
        model = MinimalModel()
        print("✅ Minimal model created successfully!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Debugging Model Initialization Issues")
    print("=" * 50)
    
    # Test components individually
    comp_success = test_individual_components()
    
    # Test minimal model
    if comp_success:
        model_success = test_minimal_model()
        
        if model_success:
            print("\n🎉 All tests passed!")
            print("✅ Components work individually")
            print("✅ Minimal model works")
            print("The issue might be in the full model initialization")
        else:
            print("\n❌ Minimal model failed")
    else:
        print("\n❌ Component tests failed")