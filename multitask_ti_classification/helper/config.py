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

SEED = 142 # More robust seed

# --- Data Paths ---
DATA_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/multimodal_materials_db_mp")
KSPACE_GRAPHS_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/nonmagnetic_3d/kspace_topology_graphs")
MASTER_INDEX_PATH = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/metadata")
DOS_FERMI_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/pebr_tr_dos_rev5")

# --- Topology Classification (Binary: Trivial vs Topological) ---
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
LATENT_DIM_GNN = 1024  # Reduced from 4096
LATENT_DIM_ASPH = 256   # Reduced from 512
LATENT_DIM_OTHER_FFNN = 512  # Reduced from 2046
LATENT_DIM_FFNN = LATENT_DIM_OTHER_FFNN

# Enhanced Crystal encoder parameters (based on Nature paper)
crystal_encoder_output_dim = 1024
crystal_encoder_hidden_dim = 512
crystal_encoder_num_layers = 4  # Reduced for stability
crystal_encoder_radius = 15.0   # Increased radius as per paper
crystal_encoder_max_neighbors = 12  # Max neighbors as per paper
crystal_encoder_use_voronoi = True  # Use Voronoi tessellation
crystal_encoder_use_enhanced_features = True  # Rich atomic features
crystal_encoder_num_attention_heads = 8  # Multi-head attention

# Topological ML parameters
AUXILIARY_WEIGHT = 1.0

FUSION_HIDDEN_DIMS = [1024, 512, 256]  # Reduced complexity

# GNN specific
GNN_NUM_LAYERS = 6      # Reduced from 12
GNN_HIDDEN_CHANNELS = 256  # Reduced from 512
KSPACE_GNN_NUM_HEADS = 256 # Reduced from 1024

# FFNN specific
FFNN_HIDDEN_DIMS_ASPH = [1024, 512, 256]
FFNN_HIDDEN_DIMS_SCALAR = [2046, 1024, 512]

# --- Training Parameters ---
# Optimized hyperparameters for 92%+ accuracy
LEARNING_RATE = 2e-4     # Balanced learning rate
WEIGHT_DECAY = 1e-4      # Moderate L2 regularization
BATCH_SIZE = 64          # Standard batch size
NUM_EPOCHS = 50          # Sufficient epochs with early stopping
DROPOUT_RATE = 0.3       # Moderate dropout
PATIENCE = 10            # Patient early stopping
EARLY_STOPPING_METRIC = 'val_loss'  # Stop based on validation loss
MAX_GRAD_NORM = 1.0      # Reduced gradient clipping

EGNN_HIDDEN_IRREPS_STR = "64x0e + 32x1o + 16x2e"
EGNN_RADIUS = 6.0

# --- Enhanced Loss weighting (Nature paper approach) ---
LOSS_WEIGHT_MAIN_CLASSIFICATION = 1.0      # Main topology classification
LOSS_WEIGHT_AUX_CLASSIFICATION = 0.3       # Auxiliary topology head
LOSS_WEIGHT_TOPO_CONSISTENCY = 0.2         # Topological invariant consistency
LOSS_WEIGHT_CONFIDENCE_REG = 0.1           # Confidence calibration
LOSS_WEIGHT_FEATURE_REG = 0.1              # Feature regularization
USE_FOCAL_LOSS = True                       # Handle class imbalance
FOCAL_LOSS_ALPHA = [1.0, 2.0, 1.5]        # Class weights [trivial, TI, semimetal]
FOCAL_LOSS_GAMMA = 2.0                      # Focal loss gamma parameter

# --- Device Configuration ---
print("[DEBUG] config.py: About to configure device...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] config.py: Device configured as {DEVICE}")

TRAIN_RATIO = 0.7      # Reduced training set
VAL_RATIO = 0.15        # Increased validation set
TEST_RATIO = 0.15       # Increased test set
USE_STRATIFIED_SPLIT = True  # Ensure balanced class distribution
LABEL_SMOOTHING = 0.1    # Add label smoothing to prevent overconfidence

# --- Enhanced Feature Dimensions (Nature paper approach) ---
CRYSTAL_NODE_FEATURE_DIM = 65  # Rich atomic features: group(18) + period(7) + 4*binned_props(10)
CRYSTAL_EDGE_FEATURE_DIM = 15  # Enhanced edge features: dist(1) + dist_bins(10) + bond_type(4)
# Enable enhanced features
crystal_encoder_use_enhanced_features = True
crystal_encoder_use_voronoi = True
ASPH_FEATURE_DIM = 3115  # Actual ASPH feature dimension
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
try:
    with open('/scratch/gpfs/as0714/graph_vector_topological_insulator/multitask_ti_classification/irrep_unique', 'rb') as fp:
        ALL_POSSIBLE_IRREPS = pickle.load(fp)
    print(f"[DEBUG] config.py: Successfully loaded irrep_unique with {len(ALL_POSSIBLE_IRREPS)} items")
except FileNotFoundError:
    warnings.warn("irrep_unique file not found. ALL_POSSIBLE_IRREPS will be empty. Decomposition features might be impacted.")
    print("[DEBUG] config.py: irrep_unique file not found, using empty list")
except Exception as e:
    print(f"[DEBUG] config.py: Error loading irrep_unique: {e}")
    ALL_POSSIBLE_IRREPS = []

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

USE_TOPOLOGICAL_ML = True

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