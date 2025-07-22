#!/usr/bin/env python3
"""
Test script to verify the dimension fix works
"""

import torch
import numpy as np
from torch_geometric.data import Data as PyGData
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config
from training.classifier_training import EnhancedMultiModalMaterialClassifier

def create_realistic_batch(batch_size=2):
    """Create realistic data matching actual dataset dimensions"""
    
    # Crystal graph data with ACTUAL dimensions (3D node features)
    num_nodes = 50
    crystal_x = torch.randn(num_nodes, 3)  # ACTUAL: 3D features, not 65D
    crystal_edge_index = torch.randint(0, num_nodes, (2, 100))
    crystal_edge_attr = torch.randn(100, 1)  # ACTUAL: 1D edge features
    crystal_pos = torch.randn(num_nodes, 3)
    crystal_batch = torch.zeros(num_nodes, dtype=torch.long)
    
    crystal_graph = PyGData(
        x=crystal_x,
        edge_index=crystal_edge_index,
        edge_attr=crystal_edge_attr,
        pos=crystal_pos,
        batch=crystal_batch
    )
    
    # K-space graph data
    kspace_x = torch.randn(30, 10)  # 10D k-space features
    kspace_edge_index = torch.randint(0, 30, (2, 50))
    kspace_edge_attr = torch.randn(50, 4)  # 4D k-space edge features
    kspace_batch = torch.zeros(30, dtype=torch.long)
    
    kspace_graph = PyGData(
        x=kspace_x,
        edge_index=kspace_edge_index,
        edge_attr=kspace_edge_attr,
        batch=kspace_batch
    )
    
    # Other features with ACTUAL dimensions
    scalar_features = torch.randn(batch_size, 4763)  # Actual scalar dim
    asph_features = torch.randn(batch_size, 512)     # Actual ASPH dim
    
    # K-space physics features
    kspace_physics_features = {
        'decomposition_features': torch.randn(batch_size, 2),
        'gap_features': torch.randn(batch_size, 1),
        'dos_features': torch.randn(batch_size, 500),
        'fermi_features': torch.randn(batch_size, 1)
    }
    
    # Binary topology labels
    topology_labels = torch.randint(0, 2, (batch_size,))
    
    return {
        'crystal_graph': crystal_graph,
        'kspace_graph': kspace_graph,
        'scalar_features': scalar_features,
        'asph_features': asph_features,
        'kspace_physics_features': kspace_physics_features,
        'topology_label': topology_labels
    }

def test_dimension_fix():
    """Test that the dimension fix works"""
    print("🔧 Testing Dimension Fix")
    print("=" * 40)
    
    # Create model with ACTUAL input dimensions
    print("Creating model with actual dimensions...")
    model = EnhancedMultiModalMaterialClassifier(
        crystal_node_feature_dim=3,      # ACTUAL: 3D not 65D
        kspace_node_feature_dim=10,      # ACTUAL: 10D
        scalar_feature_dim=4763,         # ACTUAL: 4763D
        decomposition_feature_dim=2,     # ACTUAL: 2D
        num_topology_classes=2           # Binary classification
    )
    print("✅ Model created successfully")
    
    # Create realistic data
    print("Creating realistic data...")
    data = create_realistic_batch(batch_size=2)
    print(f"Crystal graph x shape: {data['crystal_graph'].x.shape}")
    print(f"Expected: (num_nodes, 3)")
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(data)
            print("✅ Forward pass successful!")
            print(f"Output shape: {outputs['logits'].shape}")
            print(f"Expected: (2, 2) for binary classification")
            
            if outputs['logits'].shape == (2, 2):
                print("✅ Output shape is correct!")
            else:
                print(f"❌ Wrong output shape: {outputs['logits'].shape}")
                return False
                
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss computation
    print("Testing loss computation...")
    try:
        model.train()
        outputs = model(data)
        loss = model.compute_loss(outputs, data['topology_label'])
        print(f"✅ Loss computation successful: {loss.item():.4f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ Loss is NaN or Inf!")
            return False
            
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test backward pass
    print("Testing backward pass...")
    try:
        loss.backward()
        print("✅ Backward pass successful!")
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        print(f"Gradient norm: {grad_norm:.4f}")
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("❌ Gradient norm is NaN or Inf!")
            return False
            
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All dimension fix tests passed!")
    print("✅ Model is ready for training with actual data dimensions")
    return True

if __name__ == "__main__":
    success = test_dimension_fix()
    if success:
        print("\n🚀 Ready to run training with fixed dimensions!")
    else:
        print("\n❌ Please fix the issues above")