# Recommended best-bet config for all modalities, TransformerConv, attention+gated fusion, moderate fusion MLP, 8 GNN layers, 4 crystal layers.

USE_CRYSTAL = True
USE_KSPACE = False
USE_SCALAR = True
USE_DECOMPOSITION = True  # Try False for ablation

KSPACE_GNN_TYPE = 'transformer'  # Options: 'transformer', 'gcn', 'gat', 'sage'
FUSION_HIDDEN_DIMS = [1024, 512, 128]
FUSION_DROPOUT = 0.1
GNN_NUM_LAYERS = 8
crystal_encoder_num_layers = 4
LEARNING_RATE = 5e-4
DROPOUT_RATE = 0.1
BATCH_SIZE = 32
LABEL_SMOOTHING = 0.1
PATIENCE = 20

# Updated for HybridTopoClassifier pipeline (CGCNN + ASPH + k-space GNN/Transformer/Physics)
print("[DEBUG] config.py: Starting import")
import os
print("[DEBUG] config.py: Imported os")
from pathlib import Path
print("[DEBUG] config.py: Imported Path")
import torch
print("[DEBUG] config.py: Imported torch")
import pickle
print("[DEBUG] config.py: Imported pickle")
import warnings
print("[DEBUG] config.py: Imported warnings")

PROJECT_ROOT = Path(__file__).resolve().parent

SEED = 1

# --- Data Paths ---
# Use environment variables or fallback to relative paths
import os

# Check if we're on the server (Della) or local machine
if os.path.exists("/scratch/gpfs/as0714"):
    # Server paths
    DATA_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/multimodal_materials_db_mp")
    KSPACE_GRAPHS_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/nonmagnetic_3d/kspace_topology_graphs")
    MASTER_INDEX_PATH = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/metadata")
    DOS_FERMI_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/pebr_tr_dos_rev5")
    CRYSTAL_GRAPHS_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/crystal_graphs")
    VECTORIZED_FEATURES_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/vectorized_features")
else:
    # Local paths - use relative paths from current directory
    DATA_DIR = PROJECT_ROOT / "multimodal_materials_db_mp"
    KSPACE_GRAPHS_DIR = PROJECT_ROOT / "kspace_topology_graphs"
    MASTER_INDEX_PATH = PROJECT_ROOT / "metadata"
    DOS_FERMI_DIR = PROJECT_ROOT / "dos_fermi_data"
    CRYSTAL_GRAPHS_DIR = PROJECT_ROOT / "crystal_graphs"
    VECTORIZED_FEATURES_DIR = PROJECT_ROOT / "vectorized_features"


# --- Topology Classification ---
TOPOLOGY_CLASS_MAPPING = {
    "Trivial": 0,
    "Topological Insulator": 1,
    "Semimetal": 1,
    "Weyl Semimetal": 1,
    "Dirac Semimetal": 1,
    "Unknown": 0,
}
NUM_TOPOLOGY_CLASSES = 2

NUM_WORKERS = 8

# --- Data Loading Configuration ---
PRELOAD_DATASET = True  # Enable preloading for faster training

# --- Magnetism Classification ---
# Remove or ignore all magnetism-related mappings and class counts

# --- Model Hyperparameters ---
LATENT_DIM_GNN = 256
LATENT_DIM_ASPH = 256  # Keep for compatibility with model_w_debug
LATENT_DIM_OTHER_FFNN = 256
LATENT_DIM_FFNN = LATENT_DIM_OTHER_FFNN

# Crystal encoder specific parameters
crystal_encoder_output_dim = 256

# Topological ML parameters
AUXILIARY_WEIGHT = 0.1 # Weight for auxiliary topological ML loss
crystal_encoder_hidden_dim = 128  # Increased from 64
crystal_encoder_num_layers = 6   # Increased from 3
crystal_encoder_radius = 4.0     # Increased from 3.0
crystal_encoder_num_scales = 3   # Increased from 1 (enable multi-scale)
crystal_encoder_use_topological_features = True  # Enable topological feature extraction

# --- Modality Ablation Flags ---
USE_CRYSTAL = True
USE_KSPACE = True
USE_SCALAR = True
USE_DECOMPOSITION = True

# --- K-space GNN Type ---
KSPACE_GNN_TYPE = 'transformer'  # Options: 'transformer', 'gcn', 'gat', 'sage'

# --- Fusion MLP Hyperparameters ---
FUSION_HIDDEN_DIMS = [1024, 512, 128]  # Default, can be changed
FUSION_DROPOUT = 0.1  # Default, can be changed

# GNN specific
GNN_NUM_LAYERS = 12  # Increased from 8
GNN_HIDDEN_CHANNELS = 1024  # Increased from 512
KSPACE_GNN_NUM_HEADS = 16  # Increased from 8

# FFNN specific
FFNN_HIDDEN_DIMS_SCALAR = [256, 128, 64]  # Reduced from [256, 128] for more stable training
FFNN_HIDDEN_DIMS_ASPH = [256, 128, 64]  # Keep for compatibility with model_w_debug

# --- Training Parameters ---
LEARNING_RATE = 5e-4  # Increased from 1e-4 for faster convergence
BATCH_SIZE = 16 # Increased from 8 for better gradients
NUM_EPOCHS = 150  # Increased from 100
DROPOUT_RATE = 0.1  # Reduced from 0.2 for less regularization
PATIENCE = 30  # Increased from 20 for more training time
MAX_GRAD_NORM = 1.0  # Increased from 0.5 for better gradient flow

EGNN_HIDDEN_IRREPS_STR = "64x0e + 32x1o + 16x2e"
EGNN_RADIUS = 5.0

# --- Loss weighting for multi-task learning ---
LOSS_WEIGHT_PRIMARY_COMBINED = 1.0
LOSS_WEIGHT_TOPOLOGY = 1.0
LOSS_WEIGHT_TOPO_CONSISTENCY = 1.0
LOSS_WEIGHT_REGULARIZATION = 1.0
LOSS_WEIGHT_MAGNETISM = 1.0
LOSS_WEIGHT_AUX_TOPOLOGY = 1.0
LOSS_WEIGHT_AUX_MAGNETISM = 1.0

# --- Device Configuration ---
print("[DEBUG] config.py: About to configure device...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] config.py: Device configured as {DEVICE}")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# --- Feature Dimensions ---
CRYSTAL_NODE_FEATURE_DIM = 3  # actual crystal graph node feature dimension from data
CRYSTAL_EDGE_FEATURE_DIM = 1  # e.g., binned distance
ASPH_FEATURE_DIM = 512  # Keep for compatibility with model_w_debug
KSPACE_NODE_FEATURE_DIM = 10  # updated to match your k-space node feature dim
KSPACE_EDGE_FEATURE_DIM = 4   # adjust to your k-space edge feature dim
KSPACE_GRAPH_NODE_FEATURE_DIM = 10  # k-space graph node feature dimension
KSPACE_GRAPH_EDGE_FEATURE_DIM = 4   # k-space graph edge feature dimension
BAND_REP_FEATURE_DIM = 4756

BASE_DECOMPOSITION_FEATURE_DIM = 2
ALL_POSSIBLE_IRREPS = []
MAX_DECOMPOSITION_INDICES_LEN = 100

EPSILON_FOR_STD_DIVISION = 1e-8

print("[DEBUG] config.py: About to load irrep_unique file...")
# Try to load irrep_unique file from multiple possible locations
irrep_paths = [
    PROJECT_ROOT / "irrep_unique",  # Local project directory
    Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/multitask_ti_classification/irrep_unique"),  # Server path
    PROJECT_ROOT.parent / "irrep_unique"  # Parent directory
]

ALL_POSSIBLE_IRREPS = []
for irrep_path in irrep_paths:
    try:
        if irrep_path.exists():
            with open(irrep_path, 'rb') as fp:
                ALL_POSSIBLE_IRREPS = pickle.load(fp)
            print(f"[DEBUG] config.py: Successfully loaded irrep_unique from {irrep_path} with {len(ALL_POSSIBLE_IRREPS)} items")
            break
    except Exception as e:
        print(f"[DEBUG] config.py: Error loading irrep_unique from {irrep_path}: {e}")
        continue

if not ALL_POSSIBLE_IRREPS:
    warnings.warn("irrep_unique file not found in any location. ALL_POSSIBLE_IRREPS will be empty. Decomposition features might be impacted.")
    print("[DEBUG] config.py: irrep_unique file not found, using empty list")

DECOMPOSITION_FEATURE_DIM = BASE_DECOMPOSITION_FEATURE_DIM + \
                            len(ALL_POSSIBLE_IRREPS) + \
                            MAX_DECOMPOSITION_INDICES_LEN

# --- EnhancedKSpacePhysicsFeatures ---
BAND_GAP_SCALAR_DIM = 1
DOS_FEATURE_DIM = 500  # updated to match your DOS feature dim
FERMI_FEATURE_DIM = 1

# --- Scalar Total Dim (excludes band_gap) ---
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

K_LAPLACIAN_EIGS = 10

# Hidden size of the MLP embedding those eigenvalues
SPECTRAL_HID = 32

def get_combined_label(topology_str: str, magnetism_str: str) -> int:
    topo_str_canonical = TOPOLOGY_INT_TO_CANONICAL_STR.get(
        TOPOLOGY_CLASS_MAPPING.get(topology_str, TOPOLOGY_CLASS_MAPPING["Unknown"]),
        "Trivial"
    )
    magnetism_str_canonical = "Magnetic" if magnetism_str in ["FM", "AFM", "FiM"] else "NM"
    combined_key = (topo_str_canonical, magnetism_str_canonical)
    if combined_key not in COMBINED_CLASS_MAPPING:
        warnings.warn(f"Undefined combined class for {combined_key}. Defaulting to Trivial_NM (0).")
        return 0
    return COMBINED_CLASS_MAPPING[combined_key]

# --- Inverse mappings for easy lookup ---
TOPOLOGY_INT_TO_CANONICAL_STR = {
    0: "Trivial",
    1: "Topological Insulator",
    2: "Semimetal"
}
MAGNETISM_INT_TO_CANONICAL_STR = {
    0: "NM",
    1: "FM",
    2: "AFM",
    3: "FiM"
}

def get_combined_label_from_ints(topology_int: int, magnetism_int: int) -> int:
    topo_str_canonical = TOPOLOGY_INT_TO_CANONICAL_STR.get(topology_int, "Trivial")
    raw_magnetism_str_from_int = MAGNETISM_INT_TO_CANONICAL_STR.get(magnetism_int, "NM")
    magnetism_str_combined_type = "Magnetic" if raw_magnetism_str_from_int in ["FM", "AFM", "FiM"] else "NM"
    combined_key = (topo_str_canonical, magnetism_str_combined_type)
    if combined_key not in COMBINED_CLASS_MAPPING:
        warnings.warn(f"Undefined combined class for {combined_key} from ints ({topology_int}, {magnetism_int}). Defaulting to Trivial_NM (0).")
        return 0
    return COMBINED_CLASS_MAPPING[combined_key]

# --- Model Saving ---
MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)