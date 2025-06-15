# config.py

import os
from pathlib import Path
import torch

# --- Paths Configuration ---
# Set PROJECT_ROOT to the directory where this config.py resides.
# Adjust DATA_DIR and KSPACE_GRAPHS_DIR to point to your generated data.
PROJECT_ROOT = Path(__file__).resolve().parent

# Base directory for your multimodal material database (output of IntegratedMaterialProcessor)
DATA_DIR = PROJECT_ROOT / "multimodal_materials_db_mp"
# Base directory for pre-generated k-space graphs (output of KSpacePhysicsGraphBuilder)
KSPACE_GRAPHS_DIR = PROJECT_ROOT / "kspace_topology_graphs"
# Path to the master index file
MASTER_INDEX_PATH = DATA_DIR / "master_index.parquet"

# --- Classification Labels Mapping ---
# These mappings must be consistent with how labels are generated in your data pipeline
# (e.g., in IntegratedMaterialProcessor.generate_and_save_material_record)

# Topology Classification
TOPOLOGY_CLASS_MAPPING = {
    "Topological Insulator": 0,
    "Semimetal": 1,         # Includes "Weyl Semimetal", "Dirac Semimetal"
    "Weyl Semimetal": 1,    # Explicitly map to Semimetal
    "Dirac Semimetal": 1,   # Explicitly map to Semimetal
    "Trivial": 2,           # Trivial metal/insulator (materials not classified as TI or TSM)
    "Unknown": 2,           # Treat unknown topological types as trivial for classification purposes, or filter
}
NUM_TOPOLOGY_CLASSES = len(set(TOPOLOGY_CLASS_MAPPING.values())) # Should be 3 (TI, TSM, Trivial)

# Magnetism Classification
# Ensure these match the 'magnetic_type' values from Materials Project (e.g., 'NM', 'FM', 'AFM', 'FiM')
MAGNETISM_CLASS_MAPPING = {
    "NM": 0,    # Non-magnetic
    "FM": 1,    # Ferromagnetic
    "AFM": 2,   # Antiferromagnetic
    "FiM": 3,   # Ferrimagnetic
    "UNKNOWN": 0, # Treat unknown magnetic types as non-magnetic for classification purposes
}
NUM_MAGNETISM_CLASSES = len(set(MAGNETISM_CLASS_MAPPING.values())) # Should be 4

# --- Model Hyperparameters ---
# Shared Encoder / Fusion
LATENT_DIM_GNN = 128      # Output dimension of each GNN encoder
LATENT_DIM_FFNN = 64      # Output dimension of each FFNN encoder
FUSION_HIDDEN_DIMS = [256, 128] # Dimensions for shared fusion layers

# GNN specific (Crystal Graph and K-space Graph)
GNN_NUM_LAYERS = 3
GNN_HIDDEN_CHANNELS = 128 # For node features within GNN layers

# FFNN specific (ASPH and Scalar features)
FFNN_HIDDEN_DIMS_ASPH = [256, 128]
FFNN_HIDDEN_DIMS_SCALAR = [128, 64]

# Training Parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
DROPOUT_RATE = 0.2

# Loss weighting for multi-task learning
# You can tune these weights. Start with equal weights (1.0).
# If one task is performing poorly or has significantly fewer samples, adjust its weight.
LOSS_WEIGHT_TOPOLOGY = 1.0
LOSS_WEIGHT_MAGNETISM = 1.0

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 

# --- Feature Dimensions (will be dynamically determined by dataset.py) ---
# These are placeholders; dataset.py will try to infer them or raise an error if not found.
# If you get dimension mismatch errors, inspect your data and update these or ensure
# your data generation creates consistent sizes.
ASPH_FEATURE_DIM = 63  # Based on your mp-1 example output for asph_features
BAND_REP_FEATURE_DIM = 4756 # Based on your mp-1 example output for band_rep_features
CRYSTAL_NODE_FEATURE_DIM = 3 # Based on _get_elemental_features output (atomic number, period, group)
KSPACE_GRAPH_NODE_FEATURE_DIM = 100 # Placeholder: This is highly dependent on vocab size in KSpacePhysicsGraphBuilder
                                    # It's 3 (k-coords) + irrep_vocab_size + branch_irrep_vocab_size + decomp_vocab_size
                                    # Inspect an actual kspace_graph.x.shape[1] from your data if errors occur.
SCALAR_TOTAL_DIM = None # This will be dynamically set by train.py after scaler fitting.


MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)