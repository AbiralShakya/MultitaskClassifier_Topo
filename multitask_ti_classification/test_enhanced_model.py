#!/usr/bin/env python3
"""
Test script for the enhanced integrated model
"""

import torch
import numpy as np
from torch_geometric.data import Data as PyGData
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.enhanced_integrated_model import create_enhanced_model
import helper.config as config

def create_dummy_data(batch_size=2):
    """Create dummy data for testing"""
    
    # Crystal graph data
    num_nodes = 20
    crystal_x = torch.randn(num_nodes, 3)  # Will be replaced by enhanced features
    crystal_edge_index = torch.randint(0, num_nodes, (2, 40))
    crystal_edge_attr = torch.randn(40, 1)
    crystal_pos = torch.randn(num_nodes, 3)
    crystal_batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Add atomic numbers for enhanced features
    atomic_numbers = torch.randint(1, 119, (num_nodes,)).tolist()
    
    crystal_graph = PyGData(
        x=crystal_x,
        edge_index=crystal_edge_index,
        edge_attr=crystal_edge_attr,
        pos=crystal_pos,
        batch=crystal_batch,
        atomic_numbers=atomic_numbers
    )
    
    # K-space graph data
    kspace_x = torch.randn(15, 10)
    kspace_edge_index = torch.randint(0, 15, (2, 25))
    kspace_edge_attr = torch.randn(25, 4)
    kspace_batch = torch.zeros(15, dtype=torch.long)
    
    kspace_graph = PyGData(
        x=kspace_x,
        edge_index=kspace_edge_index,
        edge_attr=kspace_edge_attr,
        batch=kspace_batch
    )
    
    # Other features
    scalar_features = torch.randn(batch_size, 4763)
    asph_features = torch.randn(batch_size, 512)
    
    # K-space physics features
    kspace_physics_features = {
        'decomposition_features': torch.randn(batch_size, 2),
        'gap_features': torch.randn(batch_size, 1),
        'dos_features': torch.randn(batch_size, 500),
        'fermi_features': torch.randn(batch_size, 1)
    }
    
    # Labels
    topology_labels = torch.randint(0, 2, (batch_size,))
    
    return {
        'crystal_graph': crystal_graph,
        'kspace_graph': kspace_graph,
        'scalar_features': scalar_features,
        'asph_features': asph_features,
        'kspace_physics_features': kspace_physics_features,
        'topology_label': topology_labels
    }

def test_enhanced_model():
    """Test the enhanced model"""
    print("Testing Enhanced Integrated Material Classifier...")
    
    # Create model
    print("Creating enhanced model...")
    model = create_enhanced_model()
    print(f"Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data
    print("Creating dummy data...")
    data = create_dummy_data(batch_size=2)
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(data)
            print("Forward pass successful!")
            print(f"Output keys: {list(outputs.keys())}")
            print(f"Logits shape: {outputs['logits'].shape}")
            if 'confidence' in outputs:
                print(f"Confidence shape: {outputs['confidence'].shape}")
            if 'topo_features' in outputs:
                print(f"Topological features available: {list(outputs['topo_features'].keys())}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test loss computation
    print("Testing loss computation...")
    try:
        model.train()
        outputs = model(data)
        loss = model.compute_loss(outputs, data['topology_label'])
        print(f"Loss computation successful! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test prediction with confidence
    print("Testing prediction with confidence...")
    try:
        predictions = model.predict_with_confidence(data)
        print("Prediction successful!")
        print(f"Predictions: {predictions['predictions']}")
        print(f"Confidence: {predictions['confidence']}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("All tests passed! ✅")
    return True

def test_backward_compatibility():
    """Test backward compatibility with existing training code"""
    print("\nTesting backward compatibility...")
    
    from src.enhanced_integrated_model import EnhancedMultiModalMaterialClassifier
    
    # Test with old parameter names
    try:
        model = EnhancedMultiModalMaterialClassifier(
            crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
            kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
            scalar_feature_dim=config.SCALAR_TOTAL_DIM,
            decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
            num_topology_classes=config.NUM_TOPOLOGY_CLASSES,
            # Old parameter names
            crystal_encoder_hidden_dim=256,
            crystal_encoder_num_layers=4,
            kspace_gnn_num_heads=8,
        )
        print("Backward compatibility test passed! ✅")
        return True
    except Exception as e:
        print(f"Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Integrated Model Test Suite")
    print("=" * 60)
    
    # Test enhanced model
    success1 = test_enhanced_model()
    
    # Test backward compatibility
    success2 = test_backward_compatibility()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! The enhanced model is ready for training.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)