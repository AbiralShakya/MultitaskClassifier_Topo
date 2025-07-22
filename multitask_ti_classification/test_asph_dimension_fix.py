#!/usr/bin/env python3
"""
Test script to verify the ASPH dimension fix works
"""

import torch
import numpy as np
from torch_geometric.data import Data as PyGData
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config

def test_asph_dimensions():
    """Test ASPH encoder with correct dimensions"""
    print("🔧 Testing ASPH Dimension Fix")
    print("=" * 40)
    
    try:
        from encoders.asph_encoder import ASPHEncoder
        
        print(f"Config ASPH_FEATURE_DIM: {config.ASPH_FEATURE_DIM}")
        
        # Test with actual dimension (3115)
        actual_asph_dim = 3115
        print(f"Actual ASPH dimension: {actual_asph_dim}")
        
        # Create encoder with correct dimension
        encoder = ASPHEncoder(
            input_dim=actual_asph_dim,
            hidden_dims=256,
            out_dim=128
        )
        
        # Test with batch of ASPH features
        batch_size = 4
        asph_features = torch.randn(batch_size, actual_asph_dim)
        
        print(f"Input shape: {asph_features.shape}")
        
        # Forward pass
        output = encoder(asph_features)
        print(f"Output shape: {output.shape}")
        print(f"Expected: ({batch_size}, 128)")
        
        if output.shape == (batch_size, 128):
            print("✅ ASPH encoder works with correct dimensions!")
            return True
        else:
            print(f"❌ Wrong output shape: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ ASPH dimension test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_with_correct_asph_dim():
    """Test the full model with correct ASPH dimensions"""
    print("\n🎯 Testing Full Model with Correct ASPH Dimensions")
    print("=" * 50)
    
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
        
        print("2. Creating test data with correct ASPH dimensions...")
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
        
        # Test data with CORRECT ASPH dimensions (3115)
        batch_size = 2
        actual_asph_dim = 3115
        
        data = {
            'crystal_graph': crystal_graph,
            'kspace_graph': kspace_graph,
            'scalar_features': torch.randn(batch_size, 4763),
            'asph_features': torch.randn(batch_size, actual_asph_dim),  # 3115D not 512D
            'kspace_physics_features': {
                'decomposition_features': torch.randn(batch_size, 2),
                'gap_features': torch.randn(batch_size, 1),
                'dos_features': torch.randn(batch_size, 500),
                'fermi_features': torch.randn(batch_size, 1)
            },
            'topology_label': torch.randint(0, 2, (batch_size,))
        }
        
        print(f"ASPH features shape: {data['asph_features'].shape}")
        print("✅ Test data created with correct dimensions!")
        
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

def check_actual_data_dimensions():
    """Check what the actual data dimensions are"""
    print("\n🔍 Checking Actual Data Dimensions")
    print("=" * 40)
    
    try:
        # Try to load a sample from the dataset to check dimensions
        from helper.dataset import MaterialDataset
        
        print("Loading dataset to check actual dimensions...")
        dataset = MaterialDataset(
            master_index_path=config.MASTER_INDEX_PATH,
            kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
            data_root_dir=config.DATA_DIR,
            dos_fermi_dir=config.DOS_FERMI_DIR,
            preload=False  # Don't preload for this test
        )
        
        # Get first sample
        sample = dataset[0]
        
        print("Actual data dimensions:")
        print(f"  Crystal graph x: {sample['crystal_graph'].x.shape}")
        print(f"  K-space graph x: {sample['kspace_graph'].x.shape}")
        print(f"  Scalar features: {sample['scalar_features'].shape}")
        print(f"  ASPH features: {sample['asph_features'].shape}")
        print(f"  Topology label: {sample['topology_label']}")
        
        asph_actual_dim = sample['asph_features'].shape[0] if sample['asph_features'].ndim == 1 else sample['asph_features'].shape[1]
        print(f"\n📊 ASPH actual dimension: {asph_actual_dim}")
        
        if asph_actual_dim == 3115:
            print("✅ ASPH dimension matches expected (3115)")
            return True
        else:
            print(f"❌ ASPH dimension mismatch! Expected 3115, got {asph_actual_dim}")
            return False
        
    except Exception as e:
        print(f"❌ Data dimension check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 ASPH Dimension Fix Test Suite")
    print("=" * 50)
    
    # Check actual data dimensions
    test1 = check_actual_data_dimensions()
    
    # Test ASPH encoder with correct dimensions
    test2 = test_asph_dimensions()
    
    # Test full model
    test3 = test_model_with_correct_asph_dim()
    
    print("\n" + "=" * 50)
    if test1 and test2 and test3:
        print("🎉 ALL ASPH DIMENSION TESTS PASSED!")
        print("✅ Actual data dimensions verified")
        print("✅ ASPH encoder works with 3115D input")
        print("✅ Full model works")
        print("\n🚀 Ready for training!")
    else:
        print("❌ Some tests failed:")
        print(f"  Data dimensions: {'✅' if test1 else '❌'}")
        print(f"  ASPH encoder: {'✅' if test2 else '❌'}")
        print(f"  Full model: {'✅' if test3 else '❌'}")
    print("=" * 50)