import torch
from torch_geometric.loader import DataLoader
import traceback
import sys
from pathlib import Path 
import os
import warnings

# Dynamically add the project root to sys.path
project_root = Path(__file__).parent.parent # Go up two levels from test_components.py
sys.path.insert(0, str(project_root)) # Add to the beginning of sys.path

# adjust these imports to match your code structure
from helper import config
from helper.dataset import MaterialDataset, custom_collate_fn
from helper.data_processing import ImprovedDataPreprocessor, StratifiedDataSplitter
from src.model import RealSpaceEGNNEncoder, MultiModalMaterialClassifier

def main():
    print(f"Using device: {config.DEVICE}")
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1) Build your dataset + dataloader
    print("\n=== Initializing Dataset and Preprocessing ===")
    
    # Load the full dataset (without initial scaler, as preprocessor will handle it)
    full_dataset_raw = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        scaler=None # No scaler at this stage
    )
    
    # Collect all data samples from MaterialDataset to fit and transform
    all_raw_data_dicts = []
    for i in range(len(full_dataset_raw)):
        try:
            all_raw_data_dicts.append(full_dataset_raw[i])
        except Exception as e:
            warnings.warn(f"Skipping material at index {i} during initial data collection due to loading error: {e}")
            continue

    preprocessor = ImprovedDataPreprocessor()
    processed_data_list = preprocessor.fit_transform(all_raw_data_dicts)
    print(f"Data preprocessing complete. Total processed samples: {len(processed_data_list)}")

    # Create dummy splits (StratifiedDataSplitter needs labels, but we just want one batch for testing)
    # For a simple test, we can just use the processed_data_list directly
    if not processed_data_list:
        print("Error: No data processed. Cannot proceed.")
        return

    # Create a DataLoader from the processed data list
    loader = DataLoader(processed_data_list, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    # 2) Grab one batch
    print("\n=== Grabbing one batch ===")
    try:
        batch = next(iter(loader))
        print(f"Successfully grabbed a batch with {len(batch['jid'])} samples.")
        print("Batch keys:", list(batch.keys()))
        if 'crystal_graph' not in batch:
            print("Error: 'crystal_graph' missing from batch. Cannot proceed with crystal encoder test.")
            return
    except StopIteration:
        print("Error: DataLoader is empty. No data to process.")
        return
    except Exception as e:
        print(f"Error grabbing batch: {e}")
        traceback.print_exc()
        return


    # 3) Inspect crystal_graph fields
    cg = batch['crystal_graph']
    print("\n=== Inspecting crystal_graph ===")
    print("  cg.x.shape          ", cg.x.shape)
    print("  cg.pos.shape        ", cg.pos.shape)
    print("  cg.edge_index.shape ", cg.edge_index.shape)
    print("  cg.batch.shape      ", cg.batch.shape)
    print(f"  cg.x irreps: {cg.x.irreps if hasattr(cg.x, 'irreps') else 'No irreps'}")


    # 4) Test just the RealSpaceEGNNEncoder
    print("\n=== Testing RealSpaceEGNNEncoder ===")
    encoder = RealSpaceEGNNEncoder(
        node_input_scalar_dim = config.CRYSTAL_NODE_FEATURE_DIM,
        hidden_irreps_str     = config.EGNN_HIDDEN_IRREPS_STR, # Corrected config key
        n_layers              = config.GNN_NUM_LAYERS,         # Corrected config key
        radius                = config.EGNN_RADIUS
    ).to(config.DEVICE) # Move encoder to device
    
    # Move crystal_graph to device
    cg_on_device = cg.to(config.DEVICE)

    try:
        out = encoder(cg_on_device)
        print("  ✔ encoder output shape:", out.shape)
        print(f"  ✔ encoder output device: {out.device}")
    except Exception as e:
        print("  ✖ encoder failed:")
        traceback.print_exc()

    # 5) Now test full multimodal classifier (optional)
    print("\n=== Testing MultiModalMaterialClassifier ===")
    classifier = MultiModalMaterialClassifier(
        crystal_node_feature_dim    = config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim     = config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim            = config.ASPH_FEATURE_DIM,
        scalar_feature_dim          = config.SCALAR_TOTAL_DIM, # Corrected config key
        decomposition_feature_dim   = config.DECOMPOSITION_FEATURE_DIM,
        num_topology_classes        = config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes       = config.NUM_MAGNETISM_CLASSES,
        # Pass through other hyperparams from config needed for __init__
        egnn_hidden_irreps_str = config.EGNN_HIDDEN_IRREPS_STR,
        egnn_num_layers = config.GNN_NUM_LAYERS,
        egnn_radius = config.EGNN_RADIUS,
        kspace_gnn_hidden_channels = config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers = config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads = config.KSPACE_GNN_NUM_HEADS,
        ffnn_hidden_dims_asph = config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar = config.FFNN_HIDDEN_DIMS_SCALAR,
        latent_dim_gnn = config.LATENT_DIM_GNN,
        latent_dim_asph = config.LATENT_DIM_ASPH,
        latent_dim_other_ffnn = config.LATENT_DIM_OTHER_FFNN,
        fusion_hidden_dims = config.FUSION_HIDDEN_DIMS,
    ).to(config.DEVICE) # Move classifier to device

    # Build the inputs dict exactly as your training loop does:
    inputs = {
        'crystal_graph':           cg_on_device, # Use cg_on_device
        'kspace_graph':            batch['kspace_graph'].to(config.DEVICE),
        'asph_features':           batch['asph_features'].to(config.DEVICE),
        'scalar_features':         batch['scalar_features'].to(config.DEVICE),
        'kspace_physics_features': {k: v.to(config.DEVICE) for k, v in batch['kspace_physics_features'].items()},
    }

    try:
        logits = classifier(inputs)
        for head, logit in logits.items():
            print(f"  ✔ {head} shape: {logit.shape}")
            print(f"  ✔ {head} device: {logit.device}")
    except Exception as e:
        print("  ✖ classifier failed:")
        traceback.print_exc()

if __name__ == '__main__':
    main()