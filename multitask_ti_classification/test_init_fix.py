#!/usr/bin/env python3
"""
Test script to verify the __init__ fix works
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config

def test_model_init():
    """Test that the model initializes correctly"""
    print("🔧 Testing Model Initialization Fix")
    print("=" * 40)
    
    try:
        from training.classifier_training import EnhancedMultiModalMaterialClassifier
        
        print("Creating model...")
        model = EnhancedMultiModalMaterialClassifier(
            crystal_node_feature_dim=3,      # Actual dimension
            kspace_node_feature_dim=10,
            scalar_feature_dim=4763,
            decomposition_feature_dim=2,
            num_topology_classes=2
        )
        
        print("✅ Model initialization successful!")
        print(f"Model type: {type(model)}")
        print(f"Is nn.Module: {isinstance(model, torch.nn.Module)}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Check that key components exist
        components = [
            'legacy_crystal_encoder',
            'legacy_kspace_encoder', 
            'legacy_scalar_encoder',
            'legacy_physics_encoder',
            'legacy_spectral_encoder',
            'legacy_ph_encoder'
        ]
        
        for comp in components:
            if hasattr(model, comp):
                print(f"✅ {comp}: Present")
            else:
                print(f"❌ {comp}: Missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_init()
    if success:
        print("\n🎉 Model initialization fix successful!")
        print("✅ Ready for training")
    else:
        print("\n❌ Model initialization still has issues")