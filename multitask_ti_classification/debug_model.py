import torch
import torch.nn.functional as F
from helper.dataset import MaterialDataset
from training.classifier_training import EnhancedMultiModalMaterialClassifier
import helper.config as config
import numpy as np

def debug_model():
    print("=== MODEL DEBUGGING ===")
    
    # Load dataset
    dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR,
        preload=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    
    # Check labels
    print(f"\n=== LABELS ===")
    print(f"Topology label: {sample['topology_label']}")
    print(f"Magnetism label: {sample['magnetism_label']}")
    print(f"Combined label: {sample['combined_label']}")
    
    # Check feature dimensions
    print(f"\n=== FEATURE DIMENSIONS ===")
    print(f"Crystal graph x: {sample['crystal_graph'].x.shape}")
    print(f"K-space graph x: {sample['kspace_graph'].x.shape}")
    print(f"ASPH features: {sample['asph_features'].shape}")
    print(f"Scalar features: {sample['scalar_features'].shape}")
    
    # Check feature statistics
    print(f"\n=== FEATURE STATISTICS ===")
    print(f"ASPH features - min: {sample['asph_features'].min():.4f}, max: {sample['asph_features'].max():.4f}, mean: {sample['asph_features'].mean():.4f}, std: {sample['asph_features'].std():.4f}")
    print(f"Scalar features - min: {sample['scalar_features'].min():.4f}, max: {sample['scalar_features'].max():.4f}, mean: {sample['scalar_features'].mean():.4f}, std: {sample['scalar_features'].std():.4f}")
    
    # Check for NaN/Inf
    print(f"\n=== NaN/Inf CHECK ===")
    print(f"ASPH has NaN: {torch.isnan(sample['asph_features']).any()}")
    print(f"ASPH has Inf: {torch.isinf(sample['asph_features']).any()}")
    print(f"Scalar has NaN: {torch.isnan(sample['scalar_features']).any()}")
    print(f"Scalar has Inf: {torch.isinf(sample['scalar_features']).any()}")
    
    # Create model
    model = EnhancedMultiModalMaterialClassifier(
        crystal_node_feature_dim=sample['crystal_graph'].x.shape[1],
        kspace_node_feature_dim=sample['kspace_graph'].x.shape[1],
        asph_feature_dim=sample['asph_features'].shape[0],
        scalar_feature_dim=sample['scalar_features'].shape[0],
        decomposition_feature_dim=sample['kspace_physics_features']['decomposition_features'].shape[0]
    )
    
    print(f"\n=== MODEL ARCHITECTURE ===")
    print(f"Model created successfully")
    
    # Test forward pass
    print(f"\n=== FORWARD PASS TEST ===")
    try:
        # Create batch-like input
        batch = {
            'crystal_graph': sample['crystal_graph'],
            'kspace_graph': sample['kspace_graph'],
            'asph_features': sample['asph_features'].unsqueeze(0),
            'scalar_features': sample['scalar_features'].unsqueeze(0),
            'kspace_physics_features': {
                'decomposition_features': sample['kspace_physics_features']['decomposition_features'].unsqueeze(0),
                'gap_features': sample['kspace_physics_features']['gap_features'].unsqueeze(0),
                'dos_features': sample['kspace_physics_features']['dos_features'].unsqueeze(0),
                'fermi_features': sample['kspace_physics_features']['fermi_features'].unsqueeze(0)
            }
        }
        
        outputs = model(batch)
        
        print(f"Forward pass successful!")
        print(f"Combined logits shape: {outputs['combined_logits'].shape}")
        print(f"Combined logits: {outputs['combined_logits']}")
        
        # Check logits statistics
        print(f"\n=== LOGITS STATISTICS ===")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} - min: {value.min():.4f}, max: {value.max():.4f}, mean: {value.mean():.4f}, std: {value.std():.4f}")
        
        # Test loss computation
        print(f"\n=== LOSS COMPUTATION ===")
        targets = {
            'combined': torch.tensor([sample['combined_label']]),
            'topology': torch.tensor([sample['topology_label']]),
            'magnetism': torch.tensor([sample['magnetism_label']])
        }
        
        losses = model.compute_enhanced_loss(outputs, targets)
        print(f"Losses: {losses}")
        
        # Test accuracy
        print(f"\n=== ACCURACY TEST ===")
        combined_pred = torch.argmax(outputs['combined_logits'], dim=1)
        topology_pred = torch.argmax(outputs['topology_logits_primary'], dim=1)
        magnetism_pred = torch.argmax(outputs['magnetism_logits_aux'], dim=1)
        
        combined_acc = (combined_pred == targets['combined']).float().mean()
        topology_acc = (topology_pred == targets['topology']).float().mean()
        magnetism_acc = (magnetism_pred == targets['magnetism']).float().mean()
        
        print(f"Combined accuracy: {combined_acc:.4f}")
        print(f"Topology accuracy: {topology_acc:.4f}")
        print(f"Magnetism accuracy: {magnetism_acc:.4f}")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model() 