#!/usr/bin/env python3
import torch
from torch_geometric.loader import DataLoader
import traceback
import sys
import warnings
from pathlib import Path # Import Path for robust path handling
import os # Import os for path.join

# Dynamically add the project root to sys.path
# This assumes the script is inside 'src/' and the project root is its parent directory.
project_root = Path(__file__).parent.parent # Go up two levels from test_components.py
sys.path.insert(0, str(project_root)) # Add to the beginning of sys.path

# adjust these imports to match your code structure
from helper import config
from helper.dataset import MaterialDataset, custom_collate_fn
from helper.data_processing import ImprovedDataPreprocessor, StratifiedDataSplitter
from src.model import RealSpaceEGNNEncoder, MultiModalMaterialClassifier

# ... (rest of the test_components.py content) ...
# ——— Adjust this to point at your project root ———
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ——— Your imports — take care to match your structure ———
from helper import config
from helper.dataset import MaterialDataset, custom_collate_fn
from helper.data_processing import ImprovedDataPreprocessor
from src.model import RealSpaceEGNNEncoder, MultiModalMaterialClassifier

def main():
    print(f"Using device: {config.DEVICE}")
    torch.manual_seed(config.SEED)

    # 1) Build & preprocess dataset
    print("\n=== Loading & preprocessing full dataset ===")
    raw = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        scaler=None
    )
    all_dicts = []
    for i in range(len(raw)):
        try:
            all_dicts.append(raw[i])
        except Exception as e:
            warnings.warn(f"skip idx {i}: {e}")
    proc = ImprovedDataPreprocessor()
    data_list = proc.fit_transform(all_dicts)
    print(f"Processed {len(data_list)} samples")

    if not data_list:
        print("❌ no data to test, aborting.")
        return

    # 2) One‐batch DataLoader
    loader = DataLoader(data_list, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    try:
        batch = next(iter(loader))
    except Exception as e:
        print("❌ failed to grab a batch:")
        traceback.print_exc()
        return

    print("\nBatch keys:", batch.keys())
    if 'crystal_graph' not in batch:
        print("❌ missing 'crystal_graph' → cannot proceed.")
        return

    # 3) Inspect crystal_graph
    cg = batch['crystal_graph']
    print("\n=== crystal_graph tensors ===")
    print(" cg.x.shape         ", cg.x.shape)
    print(" cg.pos.shape       ", cg.pos.shape)
    print(" cg.edge_index.shape", cg.edge_index.shape)
    print(" cg.batch.shape     ", cg.batch.shape)

    # 4) Test RealSpaceEGNNEncoder
    print("\n=== Testing RealSpaceEGNNEncoder ===")
    encoder = RealSpaceEGNNEncoder(
        node_input_scalar_dim = config.CRYSTAL_NODE_FEATURE_DIM,
        hidden_irreps_str     = config.EGNN_HIDDEN_IRREPS_STR,
        n_layers              = config.GNN_NUM_LAYERS,
        radius                = config.EGNN_RADIUS
    ).to(config.DEVICE)

    # move data
    cg = cg.to(config.DEVICE)

    # debug‐print & assert right before the TP inside forward
    def debug_forward_check(encoder, cg):
        x_raw = cg.x
        if not torch.jit.is_scripting():
            print(f"[DEBUG] x_raw.shape = {x_raw.shape}, expected dim = {encoder.input_node_irreps.dim}")
        # now call
        return encoder(cg)

    try:
        out = debug_forward_check(encoder, cg)
        print(" ✔ encoder output shape:", out.shape)
    except AssertionError as e:
        print(" ❌ EAGER‐mode assertion:", e)
    except Exception:
        print(" ❌ encoder threw:")
        traceback.print_exc()

    # 5) Test full classifier
    print("\n=== Testing MultiModalMaterialClassifier ===")
    classifier = MultiModalMaterialClassifier(
        crystal_node_feature_dim    = config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim     = config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim            = config.ASPH_FEATURE_DIM,
        scalar_feature_dim          = config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim   = config.DECOMPOSITION_FEATURE_DIM,
        num_topology_classes        = config.NUM_TOPOLOGY_CLASSES,
        num_magnetism_classes       = config.NUM_MAGNETISM_CLASSES,

        # pass through your other hyperparams here...
        egnn_hidden_irreps_str      = config.EGNN_HIDDEN_IRREPS_STR,
        egnn_num_layers             = config.GNN_NUM_LAYERS,
        egnn_radius                 = config.EGNN_RADIUS,
        kspace_gnn_hidden_channels  = config.GNN_HIDDEN_CHANNELS,
        kspace_gnn_num_layers       = config.GNN_NUM_LAYERS,
        kspace_gnn_num_heads        = config.KSPACE_GNN_NUM_HEADS,
        ffnn_hidden_dims_asph       = config.FFNN_HIDDEN_DIMS_ASPH,
        ffnn_hidden_dims_scalar     = config.FFNN_HIDDEN_DIMS_SCALAR,
        latent_dim_gnn              = config.LATENT_DIM_GNN,
        latent_dim_asph             = config.LATENT_DIM_ASPH,
        latent_dim_other_ffnn       = config.LATENT_DIM_OTHER_FFNN,
        fusion_hidden_dims          = config.FUSION_HIDDEN_DIMS,
    ).to(config.DEVICE)

    # prepare inputs
    inputs = {
        'crystal_graph':           cg,
        'kspace_graph':            batch['kspace_graph'].to(config.DEVICE),
        'asph_features':           batch['asph_features'].to(config.DEVICE),
        'scalar_features':         batch['scalar_features'].to(config.DEVICE),
        'kspace_physics_features': {k: v.to(config.DEVICE) for k, v in batch['kspace_physics_features'].items()},
    }

    try:
        logits = classifier(inputs)
        for head, logit in logits.items():
            print(f" ✔ {head} shape:", logit.shape)
    except Exception:
        print(" ❌ classifier threw:")
        traceback.print_exc()


if __name__ == "__main__":
    main()




