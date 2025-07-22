#!/usr/bin/env python3
"""
Test script to verify binary topology classification is working correctly
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

def create_dummy_batch(batch_size=4):
    """Create dummy data for testing binary classification"""
    
    # Crystal graph data
    num_nodes = 20
    crystal_x = torch.randn(num_nodes, config.CRYSTAL_NODE_FEATURE_DIM)
    crystal_edge_index = torch.randint(0, num_nodes, (2, 40))
    crystal_edge_attr = torch.randn(40, config.CRYSTAL_EDGE_FEATURE_DIM)
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
    kspace_x = torch.randn(15, config.KSPACE_GRAPH_NODE_FEATURE_DIM)
    kspace_edge_index = torch.randint(0, 15, (2, 25))
    kspace_edge_attr = torch.randn(25, config.KSPACE_GRAPH_EDGE_FEATURE_DIM)
    kspace_batch = torch.zeros(15, dtype=torch.long)
    
    kspace_graph = PyGData(
        x=kspace_x,
        edge_index=kspace_edge_index,
        edge_attr=kspace_edge_attr,
        batch=kspace_batch
    )
    
    # Other features
    scalar_features = torch.randn(batch_size, config.SCALAR_TOTAL_DIM)
    asph_features = torch.randn(batch_size, config.ASPH_FEATURE_DIM)
    
    # K-space physics features
    kspace_physics_features = {
        'decomposition_features': torch.randn(batch_size, config.DECOMPOSITION_FEATURE_DIM),
        'gap_features': torch.randn(batch_size, config.BAND_GAP_SCALAR_DIM),
        'dos_features': torch.randn(batch_size, config.DOS_FEATURE_DIM),
        'fermi_features': torch.randn(batch_size, config.FERMI_FEATURE_DIM)
    }
    
    # BINARY topology labels only (0=trivial, 1=topological)
    topology_labels = torch.randint(0, 2, (batch_size,))
    
    return {
        'crystal_graph': crystal_graph,
        'kspace_graph': kspace_graph,
        'scalar_features': scalar_features,
        'asph_features': asph_features,
        'kspace_physics_features': kspace_physics_features,
        'topology_label': topology_labels
    }

def test_binary_classification():
    """Test binary topology classification"""
    print("🧪 Testing Binary Topology Classification")
    print("=" * 50)
    
    # Check config
    print(f"Number of topology classes: {config.NUM_TOPOLOGY_CLASSES}")
    print(f"Topology class mapping: {config.TOPOLOGY_CLASS_MAPPING}")
    
    if config.NUM_TOPOLOGY_CLASSES != 2:
        print("❌ ERROR: Config is not set for binary classification!")
        return False
    
    # Create model
    print("\n📦 Creating enhanced model...")
    try:
        model = EnhancedMultiModalMaterialClassifier(
            crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
            kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
            scalar_feature_dim=config.SCALAR_TOTAL_DIM,
            decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
            num_topology_classes=config.NUM_TOPOLOGY_CLASSES
        )
        print(f"✅ Model created successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create dummy data
    print("\n📊 Creating dummy data...")
    try:
        data = create_dummy_batch(batch_size=4)
        print(f"✅ Dummy data created successfully!")
        print(f"Topology labels: {data['topology_label']}")
        print(f"Label range: {data['topology_label'].min().item()} - {data['topology_label'].max().item()}")
    except Exception as e:
        print(f"❌ Data creation failed: {e}")
        return False
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(data)
            print(f"✅ Forward pass successful!")
            print(f"Output keys: {list(outputs.keys())}")
            print(f"Logits shape: {outputs['logits'].shape}")
            print(f"Expected shape: (4, 2) for binary classification")
            
            if outputs['logits'].shape != (4, 2):
                print(f"❌ ERROR: Wrong output shape! Expected (4, 2), got {outputs['logits'].shape}")
                return False
                
            # Check predictions
            probs = torch.softmax(outputs['logits'], dim=1)
            predictions = torch.argmax(probs, dim=1)
            print(f"Predictions: {predictions}")
            print(f"Probabilities: {probs}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss computation
    print("\n📉 Testing loss computation...")
    try:
        model.train()
        outputs = model(data)
        loss = model.compute_loss(outputs, data['topology_label'])
        print(f"✅ Loss computation successful!")
        print(f"Loss value: {loss.item():.4f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ ERROR: Loss is NaN or Inf!")
            return False
            
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test backward pass
    print("\n⬅️ Testing backward pass...")
    try:
        loss.backward()
        print(f"✅ Backward pass successful!")
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        print(f"Gradient norm: {grad_norm:.4f}")
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"❌ ERROR: Gradient norm is NaN or Inf!")
            return False
            
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All binary classification tests passed!")
    print("✅ Model is ready for binary topology classification training")
    return True

if __name__ == "__main__":
    success = test_binary_classification()
    if success:
        print("\n🚀 Ready to run: python training/classifier_training.py")
    else:
        print("\n❌ Please fix the issues above before training")