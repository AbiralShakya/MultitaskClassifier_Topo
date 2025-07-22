#!/usr/bin/env python3
"""
Final test to verify the simplified model works
"""

import torch
import numpy as np
from torch_geometric.data import Data as PyGData
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config

def test_final_model():
    """Test the final simplified model"""
    print("🎯 Testing Final Simplified Model")
    print("=" * 50)
    
    try:
        from training.classifier_training import EnhancedMultiModalMaterialClassifier
        
        print("1. Creating model...")
        model = EnhancedMultiModalMaterialClassifier(
            crystal_node_feature_dim=3,      # Actual dimension
            kspace_node_feature_dim=10,
            scalar_feature_dim=4763,
            decomposition_feature_dim=2,
            num_topology_classes=2
        )
        print("✅ Model created successfully!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print("2. Creating realistic test data...")
        # Crystal graph with actual dimensions
        crystal_graph = PyGData(
            x=torch.randn(50, 3),  # 3D node features
            edge_index=torch.randint(0, 50, (2, 100)),
            edge_attr=torch.randn(100, 1),  # 1D edge features
            batch=torch.zeros(50, dtype=torch.long)
        )
        
        # K-space graph
        kspace_graph = PyGData(
            x=torch.randn(30, 10),  # 10D node features
            edge_index=torch.randint(0, 30, (2, 50)),
            edge_attr=torch.randn(50, 4),  # 4D edge features
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
        print("✅ Test data created successfully!")
        
        print("3. Testing forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            print(f"✅ Forward pass successful!")
            print(f"Output shape: {outputs['logits'].shape}")
            print(f"Expected: (2, 2) for binary classification")
            
            if outputs['logits'].shape == (2, 2):
                print("✅ Output shape is correct!")
            else:
                print(f"❌ Wrong output shape!")
                return False
        
        print("4. Testing loss computation...")
        model.train()
        outputs = model(data)
        loss = model.compute_loss(outputs, data['topology_label'])
        print(f"✅ Loss computation successful! Loss: {loss.item():.4f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ Loss is NaN or Inf!")
            return False
        
        print("5. Testing backward pass...")
        loss.backward()
        print("✅ Backward pass successful!")
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        print(f"Gradient norm: {grad_norm:.4f}")
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("❌ Gradient norm is NaN or Inf!")
            return False
        
        print("6. Testing predictions...")
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            probs = torch.softmax(outputs['logits'], dim=1)
            predictions = torch.argmax(probs, dim=1)
            print(f"Predictions: {predictions}")
            print(f"Probabilities: {probs}")
            print("✅ Predictions look reasonable!")
        
        return True
        
    except Exception as e:
        print(f"❌ Final model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Final Model Test Suite")
    print("=" * 50)
    
    success = test_final_model()
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Model initialization works")
        print("✅ Forward pass works")
        print("✅ Loss computation works")
        print("✅ Backward pass works")
        print("✅ Predictions work")
        print("\n🚀 Ready for training!")
        print("Run: python training/classifier_training.py")
    else:
        print("\n❌ Tests failed - please check errors above")