# Simplified config matching the previous working GitHub version
# This removes the complex encoders that were added later

import os
from pathlib import Path
import torch
import pickle
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent

SEED = 1

# --- Data Paths ---
# Use local paths instead of hardcoded server paths
DATA_DIR = Path(__file__).parent.parent / "multimodal_materials_db_mp"
KSPACE_GRAPHS_DIR = Path(__file__).parent.parent / "kspace_topology_graphs"
MASTER_INDEX_PATH = Path(__file__).parent.parent / "metadata"
DOS_FERMI_DIR = Path(__file__).parent.parent / "dos_fermi_data"

# --- Topology Classification ---
TOPOLOGY_CLASS_MAPPING = {
    "Trivial": 0,
    "Topological Insulator": 1,
    "Semimetal": 2,
    "Weyl Semimetal": 2,
    "Dirac Semimetal": 2,
    "Unknown": 0,
}
NUM_TOPOLOGY_CLASSES = len(set(TOPOLOGY_CLASS_MAPPING.values()))

NUM_WORKERS = 8
PRELOAD_DATASET = True

# --- Magnetism Classification ---
MAGNETISM_CLASS_MAPPING = {
    "NM": 0,
    "FM": 1,
    "AFM": 2,
    "FiM": 3,
    "UNKNOWN": 0,
}
NUM_MAGNETISM_CLASSES = len(set(MAGNETISM_CLASS_MAPPING.values()))

# --- Model Hyperparameters (Simplified) ---
LATENT_DIM_GNN = 128
LATENT_DIM_ASPH = 128
LATENT_DIM_OTHER_FFNN = 128
LATENT_DIM_FFNN = LATENT_DIM_OTHER_FFNN

# Crystal encoder - SIMPLIFIED (matching previous working version)
crystal_encoder_output_dim = 128
crystal_encoder_hidden_dim = 64  # Keep reduced for speed
crystal_encoder_num_layers = 2   # Further reduced
crystal_encoder_radius = 3.0
crystal_encoder_num_scales = 1   # Single scale only
crystal_encoder_use_topological_features = False  # Disabled for speed

# DISABLE complex encoders that were added later
USE_SPECTRAL_ENCODER = False
USE_TOPOLOGICAL_ML = False
USE_ENHANCED_PHYSICS = False

FUSION_HIDDEN_DIMS = [256, 128]

# GNN specific
GNN_NUM_LAYERS = 2  # Reduced
GNN_HIDDEN_CHANNELS = 64  # Reduced
KSPACE_GNN_NUM_HEADS = 4  # Reduced

# FFNN specific
FFNN_HIDDEN_DIMS_ASPH = [128, 64]  # Reduced
FFNN_HIDDEN_DIMS_SCALAR = [128, 64]  # Reduced

# --- Training Parameters ---
LEARNING_RATE = 0.001
BATCH_SIZE = 16  # Increased from 8, but still smaller than original 32
NUM_EPOCHS = 50
DROPOUT_RATE = 0.2
PATIENCE = 10
MAX_GRAD_NORM = 1.0

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# --- Feature Dimensions ---
CRYSTAL_NODE_FEATURE_DIM = 3
CRYSTAL_EDGE_FEATURE_DIM = 1
ASPH_FEATURE_DIM = 512
KSPACE_NODE_FEATURE_DIM = 10
KSPACE_EDGE_FEATURE_DIM = 4
KSPACE_GRAPH_NODE_FEATURE_DIM = 10
KSPACE_GRAPH_EDGE_FEATURE_DIM = 4
BAND_REP_FEATURE_DIM = 4756

BASE_DECOMPOSITION_FEATURE_DIM = 2
ALL_POSSIBLE_IRREPS = []
MAX_DECOMPOSITION_INDICES_LEN = 100

EPSILON_FOR_STD_DIVISION = 1e-8

# Load irrep_unique file
try:
    with open(Path(__file__).parent.parent / 'irrep_unique', 'rb') as fp:
        ALL_POSSIBLE_IRREPS = pickle.load(fp)
except FileNotFoundError:
    warnings.warn("irrep_unique file not found. ALL_POSSIBLE_IRREPS will be empty.")
    ALL_POSSIBLE_IRREPS = []

DECOMPOSITION_FEATURE_DIM = BASE_DECOMPOSITION_FEATURE_DIM + \
                            len(ALL_POSSIBLE_IRREPS) + \
                            MAX_DECOMPOSITION_INDICES_LEN

# --- Simplified Physics Features ---
BAND_GAP_SCALAR_DIM = 1
DOS_FEATURE_DIM = 500
FERMI_FEATURE_DIM = 1

# --- Scalar Total Dim ---
SCALAR_TOTAL_DIM = BAND_REP_FEATURE_DIM + len([
    'formation_energy', 'energy_above_hull', 'density', 'volume', 'nsites',
    'space_group_number', 'total_magnetization'
])

# --- Combined Class Mapping ---
COMBINED_CLASS_MAPPING = {
    ("Trivial", "NM"): 0,
    ("Topological Insulator", "NM"): 1,
    ("Semimetal", "NM"): 2,
    ("Trivial", "Magnetic"): 3,
    ("Topological Insulator", "Magnetic"): 4,
    ("Semimetal", "Magnetic"): 5,
}
NUM_COMBINED_CLASSES = len(set(COMBINED_CLASS_MAPPING.values()))

# --- Model Saving ---
MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) 