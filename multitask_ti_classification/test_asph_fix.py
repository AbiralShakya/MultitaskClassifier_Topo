#!/usr/bin/env python3
"""
Test script to verify the ASPH encoder fix works
"""

import torch
import numpy as np
from torch_geometric.data import Data as PyGData
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config

def test_asph_encoder_directly():
    """Test the ASPH encoder directly"""
    print("🧪 Testing ASPH Encoder Directly")
    print("=" * 40)
    
    try:
        from encoders.asph_encoder import ASPHEncoder
        
        # Test with actual ASPH feature dimension
        asph_dim = getattr(config, 'ASPH_FEATURE_DIM', 512)
        print(f"ASPH feature dimension: {asph_dim}")
        
        encoder = ASPHEncoder(
            input_dim=asph_dim,
            hidden_dims=256,
            out_dim=128
        )
        
        # Test with batch of ASPH features
        batch_size = 4
        asph_features = torch.randn(batch_size, asph_dim)
        
        print(f"Input shape: {asph_features.shape}")
        
        # Forward pass
        output = encoder(asph_features)
        print(f"Output shape: {output.shape}")
        print(f"Expected: ({batch_size}, 128)")
        
        if output.shape == (batch_size, 128):
            print("✅ ASPH encoder works correctly!")
            return True
        else:
            print(f"❌ Wrong output shape: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ ASPH encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_with_asph_fix():
    """Test the full model with ASPH fix"""
    print("\n🔧 Testing Full Model with ASPH Fix")
    print("=" * 40)
    
    try:
        from training.classifier_training import EnhancedMultiModalMaterialClassifier
        
        print("1. Creating model...")
        model = EnhancedMultiModalMaterialClassifier(
            crystal_node_feature_dim=3,
            kspace_node_feature_dim=10,
            scalar_feature_dim=4763,
            decomposition_feature_dim=2,
            num_topology_classes=2
        )
        print("✅ Model created successfully!")
        
        print("2. Creating test data...")
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
        
        # Test data with correct ASPH features (regular tensor)
        batch_size = 2
        asph_dim = getattr(config, 'ASPH_FEATURE_DIM', 512)
        
        data = {
            'crystal_graph': crystal_graph,
            'kspace_graph': kspace_graph,
            'scalar_features': torch.randn(batch_size, 4763),
            'asph_features': torch.randn(batch_size, asph_dim),  # Regular tensor, not PyG data
            'kspace_physics_features': {
                'decomposition_features': torch.randn(batch_size, 2),
                'gap_features': torch.randn(batch_size, 1),
                'dos_features': torch.randn(batch_size, 500),
                'fermi_features': torch.randn(batch_size, 1)
            },
            'topology_label': torch.randint(0, 2, (batch_size,))
        }
        
        print(f"ASPH features shape: {data['asph_features'].shape}")
        print(f"ASPH features type: {type(data['asph_features'])}")
        print("✅ Test data created successfully!")
        
        print("3. Testing forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            print(f"✅ Forward pass successful!")
            print(f"Output shape: {outputs['logits'].shape}")
            
            if outputs['logits'].shape == (batch_size, 2):
                print("✅ Output shape is correct!")
            else:
                print(f"❌ Wrong output shape: {outputs['logits'].shape}")
                return False
        
        print("4. Testing loss computation...")
        model.train()
        outputs = model(data)
        loss = model.compute_loss(outputs, data['topology_label'])
        print(f"✅ Loss computation successful! Loss: {loss.item():.4f}")
        
        print("5. Testing backward pass...")
        loss.backward()
        print("✅ Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Full model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_flow():
    """Test the data flow to understand the issue better"""
    print("\n🔍 Testing Data Flow")
    print("=" * 40)
    
    try:
        # Simulate how data flows through the system
        print("1. Simulating dataset loading...")
        
        # This is how ASPH features are loaded in the dataset
        asph_dim = getattr(config, 'ASPH_FEATURE_DIM', 512)
        single_asph = torch.randn(asph_dim)  # Single sample
        print(f"Single ASPH sample shape: {single_asph.shape}")
        
        # This is how they're batched in collate_fn
        batch_list = [{'asph_features': single_asph} for _ in range(4)]
        batched_asph = torch.stack([d['asph_features'] for d in batch_list])
        print(f"Batched ASPH shape: {batched_asph.shape}")
        print(f"Batched ASPH type: {type(batched_asph)}")
        
        # This should work with ASPHEncoder
        from encoders.asph_encoder import ASPHEncoder
        encoder = ASPHEncoder(input_dim=asph_dim, out_dim=128)
        
        output = encoder(batched_asph)
        print(f"Encoder output shape: {output.shape}")
        print("✅ Data flow test successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Data flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 ASPH Encoder Fix Test Suite")
    print("=" * 50)
    
    # Test ASPH encoder directly
    test1 = test_asph_encoder_directly()
    
    # Test data flow
    test2 = test_data_flow()
    
    # Test full model
    test3 = test_model_with_asph_fix()
    
    print("\n" + "=" * 50)
    if test1 and test2 and test3:
        print("🎉 ALL ASPH TESTS PASSED!")
        print("✅ ASPH encoder works correctly")
        print("✅ Data flow is correct")
        print("✅ Full model works")
        print("\n🚀 Ready for training!")
    else:
        print("❌ Some tests failed:")
        print(f"  ASPH encoder: {'✅' if test1 else '❌'}")
        print(f"  Data flow: {'✅' if test2 else '❌'}")
        print(f"  Full model: {'✅' if test3 else '❌'}")
    print("=" * 50)