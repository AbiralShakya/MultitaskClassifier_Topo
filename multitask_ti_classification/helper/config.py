import os
from pathlib import Path
import torch
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent

SEED = 42

# Base directory for your multimodal material database (output of IntegratedMaterialProcessor)
DATA_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/multimodal_materials_db_mp")
# Base directory for pre-generated k-space graphs (output of KSpacePhysicsGraphBuilder)
KSPACE_GRAPHS_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/nonmagnetic_3d/kspace_topology_graphs")
# Path to the master index directory (containing individual JSON metadata files)
MASTER_INDEX_PATH = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/metadata")

# Topology Classification
TOPOLOGY_CLASS_MAPPING = {
    "Topological Insulator": 0,
    "Semimetal": 1,         # Includes "Weyl Semimetal", "Dirac Semimetal"
    "Weyl Semimetal": 1,    # Explicitly map to Semimetal
    "Dirac Semimetal": 1,   # Explicitly map to Semimetal
    "Trivial": 2,           # Trivial metal/insulator (materials not classified as TI or TSM)
    "Unknown": 2,           # Treat unknown topological types as trivial for classification purposes, or filter
}
NUM_TOPOLOGY_CLASSES = len(set(TOPOLOGY_CLASS_MAPPING.values())) # Should be 3 (TI, SM, Trivial)

NUM_WORKERS = 8

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
PATIENCE = 10 # For early stopping

EGNN_HIDDEN_IRREPS_STR = "64x0e + 32x1o + 16x2e" # As defined in model.py's RealSpaceEGNNEncoder
EGNN_RADIUS = 5.0 # Atomic interaction radius for EGNN

# KSpace GNN specific parameters
KSPACE_GNN_NUM_HEADS = 8 # As defined in model.py's KSpaceTransformerGNNEncoder

# Loss weighting for multi-task learning
LOSS_WEIGHT_TOPOLOGY = 1.0
LOSS_WEIGHT_MAGNETISM = 1.0

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 

# --- Feature Dimensions (YOU MUST SET THESE ACCURATELY) ---
# Inspect your data to get these dimensions.
# For example, if CRYSTAL_NODE_FEATURE_DIM is the output of your featurizer for atomic properties.
CRYSTAL_NODE_FEATURE_DIM = 3 # Original value, if it's atomic number, could be 1. If it's one-hot, much larger.
                           # Based on your previous code, it seems to be 3 (atomic number, period, group).
                           # Please confirm.

ASPH_FEATURE_DIM = 63  # Confirmed from your previous config.

BAND_REP_FEATURE_DIM = 4756 # Confirmed from your previous config.

KSPACE_GRAPH_NODE_FEATURE_DIM = 100 # Original value. This is the `x.shape[1]` for kspace_graph.pt
                                  # You need to confirm this from your kspace_graph.pt files.
                                  # This might be (3 for k-coords) + (size of irrep embedding/one-hot) etc.

# --- K-space Decomposition Features ---
BASE_DECOMPOSITION_FEATURE_DIM = 2

# ALL_POSSIBLE_IRREPS = sorted([
#     "R1", "T1", "U1", "V1", "X1", "Y1", "Z1", "Γ1", "GP1",
#     "R2R2", "T2T2", "U2U2", "V2V2", "X2X2", "Y2Y2", "Z2Z2", "Γ2Γ2", "2GP2",
# ])

with open('/scratch/gpfs/as0714/graph_vector_topological_insulator/multitask_ti_classification/irrep_unique', 'rb') as fp:
    ALL_POSSIBLE_IRREPS = pickle.load(fp)

# MAX_DECOMPOSITION_INDICES_LEN: Maximum expected length of the 'decomposition_indices' list
# in any 'SG_xxx/metadata.json'.
MAX_DECOMPOSITION_INDICES_LEN = 100

# DECOMPOSITION_FEATURE_DIM: Total dimension of the combined decomposition features.
# This value is calculated based on the above three and *must* match the input_dim
# of your DecompositionFeatureEncoder. It will be explicitly set in MaterialDataset.__init__.
DECOMPOSITION_FEATURE_DIM = BASE_DECOMPOSITION_FEATURE_DIM + \
                            len(ALL_POSSIBLE_IRREPS) + \
                            MAX_DECOMPOSITION_INDICES_LEN

# SCALAR_TOTAL_DIM: Total dimension of combined scalar features (band_rep_features + metadata_features)
# This will be BAND_REP_FEATURE_DIM + the number of columns in `scalar_features_columns`
# defined in your MaterialDataset.
SCALAR_TOTAL_DIM = BAND_REP_FEATURE_DIM + len([ # These are the columns in your MaterialDataset's scalar_features_columns
    'band_gap', 'formation_energy', 'density', 'volume', 'nsites',
    'space_group_number', 'total_magnetization', 'energy_above_hull'
])

# --- Model Saving ---
MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)