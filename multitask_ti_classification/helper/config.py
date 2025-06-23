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
MAX_GRAD_NORM = 1.0 # NEW: Max norm for gradient clipping

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
CRYSTAL_NODE_FEATURE_DIM = 3 
ASPH_FEATURE_DIM = 3115 
BAND_REP_FEATURE_DIM = 4756 
KSPACE_GRAPH_NODE_FEATURE_DIM = 10 

# --- K-space Decomposition Features ---
BASE_DECOMPOSITION_FEATURE_DIM = 2

with open('/scratch/gpfs/as0714/graph_vector_topological_insulator/multitask_ti_classification/irrep_unique', 'rb') as fp:
    ALL_POSSIBLE_IRREPS = pickle.load(fp)

MAX_DECOMPOSITION_INDICES_LEN = 100

DECOMPOSITION_FEATURE_DIM = BASE_DECOMPOSITION_FEATURE_DIM + \
                            len(ALL_POSSIBLE_IRREPS) + \
                            MAX_DECOMPOSITION_INDICES_LEN

SCALAR_TOTAL_DIM = BAND_REP_FEATURE_DIM + len([ 
    'band_gap', 'formation_energy', 'density', 'volume', 'nsites',
    'space_group_number', 'total_magnetization', 'energy_above_hull'
])

# --- Model Saving ---
MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
