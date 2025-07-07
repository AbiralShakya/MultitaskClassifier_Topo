# Updated for HybridTopoClassifier pipeline (CGCNN + ASPH + k-space GNN/Transformer/Physics)
import os
from pathlib import Path
import torch
import pickle
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent

SEED = 1

# --- Data Paths ---
DATA_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/multimodal_materials_db_mp")
KSPACE_GRAPHS_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/nonmagnetic_3d/kspace_topology_graphs")
MASTER_INDEX_PATH = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/metadata")
DOS_FERMI_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/pebr_tr_dos_rev5")

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

NUM_WORKERS = 4

# --- Magnetism Classification ---
MAGNETISM_CLASS_MAPPING = {
    "NM": 0,
    "FM": 1,
    "AFM": 2,
    "FiM": 3,
    "UNKNOWN": 0,
}
NUM_MAGNETISM_CLASSES = len(set(MAGNETISM_CLASS_MAPPING.values()))

# --- Model Hyperparameters ---
LATENT_DIM_GNN = 128
LATENT_DIM_ASPH = 128
LATENT_DIM_OTHER_FFNN = 128
LATENT_DIM_FFNN = LATENT_DIM_OTHER_FFNN

# Crystal encoder specific parameters
crystal_encoder_output_dim = 128
crystal_encoder_hidden_dim = 128
crystal_encoder_num_layers = 6
crystal_encoder_radius = 4.0
crystal_encoder_num_scales = 3
crystal_encoder_use_topological_features = True

FUSION_HIDDEN_DIMS = [256, 128]

# GNN specific
GNN_NUM_LAYERS = 3
GNN_HIDDEN_CHANNELS = 128
KSPACE_GNN_NUM_HEADS = 8

# FFNN specific
FFNN_HIDDEN_DIMS_ASPH = [256, 128]
FFNN_HIDDEN_DIMS_SCALAR = [256, 128]

# --- Training Parameters ---
LEARNING_RATE = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 50
DROPOUT_RATE = 0.2
PATIENCE = 10
MAX_GRAD_NORM = 1.0

EGNN_HIDDEN_IRREPS_STR = "64x0e + 32x1o + 16x2e"
EGNN_RADIUS = 3.0

# --- Loss weighting for multi-task learning ---
LOSS_WEIGHT_PRIMARY_COMBINED = 1.0
LOSS_WEIGHT_TOPOLOGY = 1.0
LOSS_WEIGHT_TOPO_CONSISTENCY = 1.0
LOSS_WEIGHT_REGULARIZATION = 1.0
LOSS_WEIGHT_MAGNETISM = 1.0
LOSS_WEIGHT_AUX_TOPOLOGY = 1.0
LOSS_WEIGHT_AUX_MAGNETISM = 1.0

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# --- Feature Dimensions ---
CRYSTAL_NODE_FEATURE_DIM = 3  # actual crystal graph node feature dimension from data
CRYSTAL_EDGE_FEATURE_DIM = 1  # e.g., binned distance
ASPH_FEATURE_DIM = 3115
KSPACE_NODE_FEATURE_DIM = 10  # updated to match your k-space node feature dim
KSPACE_EDGE_FEATURE_DIM = 4   # adjust to your k-space edge feature dim
KSPACE_GRAPH_NODE_FEATURE_DIM = 10  # k-space graph node feature dimension
KSPACE_GRAPH_EDGE_FEATURE_DIM = 4   # k-space graph edge feature dimension
BAND_REP_FEATURE_DIM = 4756

BASE_DECOMPOSITION_FEATURE_DIM = 2
ALL_POSSIBLE_IRREPS = []
MAX_DECOMPOSITION_INDICES_LEN = 100

EPSILON_FOR_STD_DIVISION = 1e-8

try:
    with open('/scratch/gpfs/as0714/graph_vector_topological_insulator/multitask_ti_classification/irrep_unique', 'rb') as fp:
        ALL_POSSIBLE_IRREPS = pickle.load(fp)
except FileNotFoundError:
    warnings.warn("irrep_unique file not found. ALL_POSSIBLE_IRREPS will be empty. Decomposition features might be impacted.")

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