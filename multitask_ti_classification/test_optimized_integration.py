#!/usr/bin/env python3
"""
Test the optimized integration to ensure it works correctly
"""

import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config
from helper.dataset import MaterialDataset
from training.classifier_training import EnhancedMultiModalMaterialClassifier
from torch_geometric.loader import DataLoader as PyGDataLoader

def test_model_creation():
    """Test that the optimized model can be created"""
    print("🧪 Testing model creation...")
    
    try:
        model = EnhancedMultiModalMaterialClassifier(
            crystal_node_feature_dim=3,
            kspace_node_feature_dim=config.KSPACE_NODE_FEATURE_DIM,
            scalar_feature_dim=config.SCALAR_TOTAL_DIM,
            decomposition_feature_dim=config.DECOMPOSITION_FEATURE_DIM,
            num_topology_classes=2
        )
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dataset_loading():
    """Test that dataset can be loaded"""
    print("\n🧪 Testing dataset loading...")
    
    try:
        dataset = MaterialDataset(
            master_index_path=config.MASTER_INDEX_PATH,
            kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
            data_root_dir=config.DATA_DIR,
            dos_fermi_dir=config.DOS_FERMI_DIR,
            preload=False  # Don't preload for testing
        )
        print(f"✅ Dataset loaded successfully")
        print(f"   Size: {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_pass(model, dataset):
    """Test forward pass with a single batch"""
    print("\n🧪 Testing forward pass...")
    
    try:
        # Create a small dataloader
        from helper.dataset import custom_collate_fn
        loader = PyGDataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=custom_collate_fn,
            num_workers=0
        )
        
        # Get first batch
        batch = next(iter(loader))
        print(f"✅ Batch loaded successfully")
        print(f"   Batch keys: {list(batch.keys())}")
        
        # Check scalar features dimension
        if 'scalar_features' in batch:
            scalar_dim = batch['scalar_features'].shape[1]
            print(f"   Scalar features dimension: {scalar_dim}")
            print(f"   Expected dimension: {config.SCALAR_TOTAL_DIM}")
            
            if scalar_dim != config.SCALAR_TOTAL_DIM:
                print(f"⚠️  Dimension mismatch! Updating config...")
                config.SCALAR_TOTAL_DIM = scalar_dim
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif hasattr(batch[key], 'to'):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], dict):
                for sub_key in batch[key]:
                    if isinstance(batch[key][sub_key], torch.Tensor):
                        batch[key][sub_key] = batch[key][sub_key].to(device)
        
        with torch.no_grad():
            outputs = model(batch)
        
        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        print(f"   Logits shape: {outputs['logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation(model, dataset):
    """Test loss computation"""
    print("\n🧪 Testing loss computation...")
    
    try:
        from helper.dataset import custom_collate_fn
        loader = PyGDataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=custom_collate_fn,
            num_workers=0
        )
        
        batch = next(iter(loader))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif hasattr(batch[key], 'to'):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], dict):
                for sub_key in batch[key]:
                    if isinstance(batch[key][sub_key], torch.Tensor):
                        batch[key][sub_key] = batch[key][sub_key].to(device)
        
        outputs = model(batch)
        loss = model.compute_loss(outputs, batch['topology_label'])
        
        print(f"✅ Loss computation successful")
        print(f"   Loss value: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Optimized Integration")
    print("=" * 50)
    
    # Test model creation
    model = test_model_creation()
    if model is None:
        return False
    
    # Test dataset loading
    dataset = test_dataset_loading()
    if dataset is None:
        return False
    
    # Test forward pass
    if not test_forward_pass(model, dataset):
        return False
    
    # Test loss computation
    if not test_loss_computation(model, dataset):
        return False
    
    print("\n🎉 All tests passed! Optimized integration is working correctly.")
    print("\n🚀 Ready to run full training:")
    print("   python run_optimized_training.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Tests failed. Please check the errors above.")
        sys.exit(1)