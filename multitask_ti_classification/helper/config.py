# import os
# from pathlib import Path
# import torch
# import pickle
# import warnings

# PROJECT_ROOT = Path(__file__).resolve().parent

# SEED = 42

# # Base directory for your multimodal material database (output of IntegratedMaterialProcessor)
# DATA_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/multimodal_materials_db_mp")
# # Base directory for pre-generated k-space graphs (output of KSpacePhysicsGraphBuilder)
# KSPACE_GRAPHS_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/nonmagnetic_3d/kspace_topology_graphs")
# # Path to the master index directory (containing individual JSON metadata files)
# MASTER_INDEX_PATH = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/metadata")

# # Topology Classification - FIXED MAPPING
# TOPOLOGY_CLASS_MAPPING = {
#     "Trivial": 0,           # Trivial metal/insulator (materials not classified as TI or TSM)
#     "Topological Insulator": 1,
#     "Semimetal": 2,         # Includes "Weyl Semimetal", "Dirac Semimetal"
#     "Weyl Semimetal": 2,    # Explicitly map to Semimetal
#     "Dirac Semimetal": 2,   # Explicitly map to Semimetal
#     "Unknown": 0,           # Treat unknown topological types as trivial by default, or filter
# }
# NUM_TOPOLOGY_CLASSES = len(set(TOPOLOGY_CLASS_MAPPING.values())) # Should be 3 (Trivial, TI, SM)

# NUM_WORKERS = 8

# # Magnetism Classification
# # Ensure these match the 'magnetic_type' values from Materials Project (e.g., 'NM', 'FM', 'AFM', 'FiM')
# MAGNETISM_CLASS_MAPPING = {
#     "NM": 0,    # Non-magnetic
#     "FM": 1,    # Ferromagnetic
#     "AFM": 2,   # Antiferromagnetic
#     "FiM": 3,   # Ferrimagnetic
#     "UNKNOWN": 0, # Treat unknown magnetic types as non-magnetic for classification purposes
# }
# NUM_MAGNETISM_CLASSES = len(set(MAGNETISM_CLASS_MAPPING.values())) # Should be 4

# # --- Model Hyperparameters ---
# # Shared Encoder / Fusion
# LATENT_DIM_GNN = 128      # Output dimension of each GNN encoder

# # Specific latent dims for different FFNN types
# LATENT_DIM_ASPH = 3115    # Full dimension for ASPH as requested
# LATENT_DIM_OTHER_FFNN = 64 # Smaller dimension for other FFNNs (Scalar, Decomposition, Enhanced Physics output)

# FUSION_HIDDEN_DIMS = [256, 128] # Dimensions for shared fusion layers

# # GNN specific (Crystal Graph and K-space Graph)
# GNN_NUM_LAYERS = 3
# GNN_HIDDEN_CHANNELS = 128 # For node features within GNN layers

# # FFNN specific (ASPH and Scalar features)
# FFNN_HIDDEN_DIMS_ASPH = [256, 128]
# FFNN_HIDDEN_DIMS_SCALAR = [128, 64]

# # Training Parameters
# LEARNING_RATE = 0.001
# BATCH_SIZE = 256
# NUM_EPOCHS = 50
# DROPOUT_RATE = 0.2
# PATIENCE = 10 # For early stopping
# MAX_GRAD_NORM = 1.0 # Max norm for gradient clipping

# EGNN_HIDDEN_IRREPS_STR = "64x0e + 32x1o + 16x2e" # As defined in model.py's RealSpaceEGNNEncoder
# EGNN_RADIUS = 5.0 # Atomic interaction radius for EGNN

# # KSpace GNN specific parameters
# KSPACE_GNN_NUM_HEADS = 8 # As defined in model.py's KSpaceTransformerGNNEncoder

# # Loss weighting for multi-task learning - Initial weights, these might be overridden by class weights
# LOSS_WEIGHT_TOPOLOGY = 1.0
# LOSS_WEIGHT_MAGNETISM = 1.0

# # --- Device Configuration ---
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAIN_RATIO = 0.8
# VAL_RATIO = 0.1
# TEST_RATIO = 0.1 

# # --- Feature Dimensions (YOU MUST SET THESE ACCURATELY) ---
# CRYSTAL_NODE_FEATURE_DIM = 3 
# ASPH_FEATURE_DIM = 3115 
# BAND_REP_FEATURE_DIM = 4756 
# KSPACE_GRAPH_NODE_FEATURE_DIM = 10 

# # --- K-space Decomposition Features (input to EnhancedKSpacePhysicsFeatures) ---
# BASE_DECOMPOSITION_FEATURE_DIM = 2 # From your existing code
# ALL_POSSIBLE_IRREPS = [] # Will be loaded from file
# MAX_DECOMPOSITION_INDICES_LEN = 100

# try:
#     with open('/scratch/gpfs/as0714/graph_vector_topological_insulator/multitask_ti_classification/irrep_unique', 'rb') as fp:
#         ALL_POSSIBLE_IRREPS = pickle.load(fp)
# except FileNotFoundError:
#     warnings.warn("irrep_unique file not found. ALL_POSSIBLE_IRREPS will be empty. Decomposition features might be impacted.")

# DECOMPOSITION_FEATURE_DIM = BASE_DECOMPOSITION_FEATURE_DIM + \
#                             len(ALL_POSSIBLE_IRREPS) + \
#                             MAX_DECOMPOSITION_INDICES_LEN

# # New feature dimensions for EnhancedKSpacePhysicsFeatures
# BAND_GAP_SCALAR_DIM = 1 # Single scalar band gap value
# DOS_FEATURE_DIM = 100   # Assuming a 100-point DOS vector if available
# FERMI_FEATURE_DIM = 50  # Assuming 50 descriptors for Fermi surface if available

# # Scalar Total Dim (now excludes band_gap as it's separate)
# # It only contains band_rep_features + other scalar metadata features
# SCALAR_TOTAL_DIM = BAND_REP_FEATURE_DIM + len([
#     'formation_energy', 'energy_above_hull', 'density', 'volume', 'nsites',
#     'space_group_number', 'total_magnetization'
# ])

# # --- Model Saving ---
# MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
# os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

import os
from pathlib import Path
import torch
import pickle
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent

SEED = 1

# Base directory for your multimodal material database (output of IntegratedMaterialProcessor)
DATA_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/multimodal_materials_db_mp")
# Base directory for pre-generated k-space graphs (output of KSpacePhysicsGraphBuilder)
KSPACE_GRAPHS_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/nonmagnetic_3d/kspace_topology_graphs")
# Path to the master index directory (containing individual JSON metadata files)
MASTER_INDEX_PATH = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/metadata")

# Topology Classification - FIXED MAPPING
TOPOLOGY_CLASS_MAPPING = {
    "Trivial": 0,           # Trivial metal/insulator (materials not classified as TI or TSM)
    "Topological Insulator": 1,
    "Semimetal": 2,         # Includes "Weyl Semimetal", "Dirac Semimetal"
    "Weyl Semimetal": 2,    # Explicitly map to Semimetal
    "Dirac Semimetal": 2,   # Explicitly map to Semimetal
    "Unknown": 0,           # Treat unknown topological types as trivial by default, or filter
}
NUM_TOPOLOGY_CLASSES = len(set(TOPOLOGY_CLASS_MAPPING.values())) # Should be 3 (Trivial, TI, SM)

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

# Specific latent dims for different FFNN types
LATENT_DIM_ASPH = 3115 # Changed to 3115 based on your empirical testing
LATENT_DIM_OTHER_FFNN = 64 # Smaller dimension for other FFNNs (Scalar, Decomposition, Enhanced Physics output)

FUSION_HIDDEN_DIMS = [1024, 512] # Dimensions for shared fusion layers

# GNN specific (Crystal Graph and K-space Graph)
GNN_NUM_LAYERS = 3
GNN_HIDDEN_CHANNELS = 128 # For node features within GNN layers

# FFNN specific (ASPH and Scalar features)
FFNN_HIDDEN_DIMS_ASPH = [256, 128]
FFNN_HIDDEN_DIMS_SCALAR = [128, 64]

# Training Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128 # Increased batch size for more stable gradients (e.g., 64, 128, or 256)
NUM_EPOCHS = 50
DROPOUT_RATE = 0.2
PATIENCE = 10 # For early stopping
MAX_GRAD_NORM = 1.0 # Max norm for gradient clipping

EGNN_HIDDEN_IRREPS_STR = "64x0e + 32x1o + 16x2e" # As defined in model.py's RealSpaceEGNNEncoder
EGNN_RADIUS = 5.0 # Atomic interaction radius for EGNN

# KSpace GNN specific parameters
KSPACE_GNN_NUM_HEADS = 8 # As defined in model.py's KSpaceTransformerGNNEncoder

# Loss weighting for multi-task learning
# LOSS_WEIGHT_TOPOLOGY = 1.0
# LOSS_WEIGHT_MAGNETISM = 1.0
LOSS_WEIGHT_TOPOLOGY        = 1.0   # α: weight on classification term
LOSS_WEIGHT_TOPO_CONSISTENCY = 0.5   # β: weight on topological‐consistency term
LOSS_WEIGHT_REGULARIZATION   = 0.3   # γ: weight on topo‐feature regularization term
LOSS_WEIGHT_MAGNETISM       = 1.0   # weight on the magnetism classification term

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 

# --- Feature Dimensions 
CRYSTAL_NODE_FEATURE_DIM = 3 
ASPH_FEATURE_DIM = 3115 
BAND_REP_FEATURE_DIM = 4756 
KSPACE_GRAPH_NODE_FEATURE_DIM = 10 

BASE_DECOMPOSITION_FEATURE_DIM = 2
ALL_POSSIBLE_IRREPS = [] 
MAX_DECOMPOSITION_INDICES_LEN = 100

try:
    with open('/scratch/gpfs/as0714/graph_vector_topological_insulator/multitask_ti_classification/irrep_unique', 'rb') as fp:
        ALL_POSSIBLE_IRREPS = pickle.load(fp)
except FileNotFoundError:
    warnings.warn("irrep_unique file not found. ALL_POSSIBLE_IRREPS will be empty. Decomposition features might be impacted.")

DECOMPOSITION_FEATURE_DIM = BASE_DECOMPOSITION_FEATURE_DIM + \
                            len(ALL_POSSIBLE_IRREPS) + \
                            MAX_DECOMPOSITION_INDICES_LEN

# New feature dimensions for EnhancedKSpacePhysicsFeatures
BAND_GAP_SCALAR_DIM = 1 # Single scalar band gap value
DOS_FEATURE_DIM = 100   # Assuming a 100-point DOS vector if available
FERMI_FEATURE_DIM = 50  # Assuming 50 descriptors for Fermi surface if available

# Scalar Total Dim (now excludes band_gap as it's separate)
# It only contains band_rep_features + other scalar metadata features
SCALAR_TOTAL_DIM = BAND_REP_FEATURE_DIM + len([
    'formation_energy', 'energy_above_hull', 'density', 'volume', 'nsites',
    'space_group_number', 'total_magnetization'
])

COMBINED_CLASS_MAPPING = {
    ("Trivial", "NM"): 0,
    ("Topological Insulator", "NM"): 1,
    ("Semimetal", "NM"): 2,
    ("Trivial", "Magnetic"): 3, # This covers FM, AFM, FiM
    ("Topological Insulator", "Magnetic"): 4,
    ("Semimetal", "Magnetic"): 5,
}

NUM_COMBINED_CLASSES = len(set(COMBINED_CLASS_MAPPING.values())) # 6 classes

def get_combined_label(topology_str: str, magnetism_str: str) -> int:
    """
    Converts raw topology and magnetism strings into a single combined integer label.
    Used in data preprocessing.
    """
    topo_str_canonical = TOPOLOGY_INT_TO_CANONICAL_STR.get(
        TOPOLOGY_CLASS_MAPPING.get(topology_str, TOPOLOGY_CLASS_MAPPING["Unknown"]), 
        "Trivial" # Fallback if direct lookup fails
    )

    magnetism_str_canonical = "Magnetic" if magnetism_str in ["FM", "AFM", "FiM"] else "NM"

    combined_key = (topo_str_canonical, magnetism_str_canonical)
    
    if combined_key not in COMBINED_CLASS_MAPPING:
        warnings.warn(f"Undefined combined class for {combined_key}. Defaulting to Trivial_NM (0).")
        return 0 
    
    return COMBINED_CLASS_MAPPING[combined_key]

# NEW: Inverse mappings for easy lookup from integer to string
# Note: "Semimetal" is canonical for both Weyl/Dirac. "Trivial" is canonical for Unknown.
TOPOLOGY_INT_TO_CANONICAL_STR = {
    0: "Trivial",
    1: "Topological Insulator",
    2: "Semimetal"
}

# Note: "Magnetic" is canonical for FM, AFM, FiM. "NM" is canonical for UNKNOWN.
MAGNETISM_INT_TO_CANONICAL_STR = {
    0: "NM",
    1: "FM", # These are the *raw* magnetic types that map to "Magnetic" combined type
    2: "AFM",
    3: "FiM"
}


# NEW: Helper function to get combined label from integer labels
def get_combined_label_from_ints(topology_int: int, magnetism_int: int) -> int:
    """
    Converts integer topology and magnetism labels into a single combined integer label.
    Uses inverse mappings to get canonical string representations first.
    """
    topo_str_canonical = TOPOLOGY_INT_TO_CANONICAL_STR.get(topology_int, "Trivial")

    # Map the magnetism integer back to a *raw* string to then determine "Magnetic" or "NM"
    # This is slightly tricky: MAGNETISM_INT_TO_CANONICAL_STR maps 0->NM, 1->FM, etc.
    # We then use the logic from get_combined_label to group FM, AFM, FiM into "Magnetic"
    raw_magnetism_str_from_int = MAGNETISM_INT_TO_CANONICAL_STR.get(magnetism_int, "NM")
    
    # Now use the existing logic for "Magnetic" vs "NM"
    magnetism_str_combined_type = "Magnetic" if raw_magnetism_str_from_int in ["FM", "AFM", "FiM"] else "NM"

    combined_key = (topo_str_canonical, magnetism_str_combined_type)
    
    if combined_key not in COMBINED_CLASS_MAPPING:
        warnings.warn(f"Undefined combined class for {combined_key} from ints ({topology_int}, {magnetism_int}). Defaulting to Trivial_NM (0).")
        return 0 
    
    return COMBINED_CLASS_MAPPING[combined_key]

# --- Model Saving ---
MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)




# import os
# from pathlib import Path
# import torch
# import pickle
# import warnings

# PROJECT_ROOT = Path(__file__).resolve().parent

# SEED = 1

# # Base directory for your multimodal material database (output of IntegratedMaterialProcessor)
# DATA_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/multimodal_materials_db_mp")
# # Base directory for pre-generated k-space graphs (output of KSpacePhysicsGraphBuilder)
# KSPACE_GRAPHS_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/nonmagnetic_3d/kspace_topology_graphs")
# # Path to the master index directory (containing individual JSON metadata files)
# MASTER_INDEX_PATH = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/metadata")

# # Topology Classification - FIXED MAPPING
# TOPOLOGY_CLASS_MAPPING = {
#     "Trivial": 0,           # Trivial metal/insulator (materials not classified as TI or TSM)
#     "Topological Insulator": 1,
#     "Semimetal": 2,         # Includes "Weyl Semimetal", "Dirac Semimetal"
#     "Weyl Semimetal": 2,    # Explicitly map to Semimetal
#     "Dirac Semimetal": 2,   # Explicitly map to Semimetal
#     "Unknown": 0,           # Treat unknown topological types as trivial by default, or filter
# }
# NUM_TOPOLOGY_CLASSES = len(set(TOPOLOGY_CLASS_MAPPING.values())) # Should be 3 (Trivial, TI, SM)

# NUM_WORKERS = 8

# # Magnetism Classification
# # Ensure these match the 'magnetic_type' values from Materials Project (e.g., 'NM', 'FM', 'AFM', 'FiM')
# MAGNETISM_CLASS_MAPPING = {
#     "NM": 0,    # Non-magnetic
#     "FM": 1,    # Ferromagnetic
#     "AFM": 2,   # Antiferromagnetic
#     "FiM": 3,   # Ferrimagnetic
#     "UNKNOWN": 0, # Treat unknown magnetic types as non-magnetic for classification purposes
# }
# NUM_MAGNETISM_CLASSES = len(set(MAGNETISM_CLASS_MAPPING.values())) # Should be 4

# # --- Model Hyperparameters ---
# # Shared Encoder / Fusion
# LATENT_DIM_GNN = 512      # Output dimension of each GNN encoder

# # Specific latent dims for different FFNN types
# LATENT_DIM_ASPH = 3115
# LATENT_DIM_OTHER_FFNN = 1024 # Smaller dimension for other FFNNs (Scalar, Decomposition, Enhanced Physics output)

# FUSION_HIDDEN_DIMS = [256, 128] # Dimensions for shared fusion layers

# # GNN specific (Crystal Graph and K-space Graph)
# GNN_NUM_LAYERS = 6
# GNN_HIDDEN_CHANNELS = 512 # For node features within GNN layers

# # FFNN specific (ASPH and Scalar features)
# FFNN_HIDDEN_DIMS_ASPH = [128,64]
# FFNN_HIDDEN_DIMS_SCALAR = [64, 32]

# # Training Parameters
# LEARNING_RATE = 0.001
# BATCH_SIZE = 32 # Increased batch size for more stable gradients (e.g., 64, 128, or 256)
# NUM_EPOCHS = 50
# DROPOUT_RATE = 0.2
# PATIENCE = 10 # For early stopping
# MAX_GRAD_NORM = 3.0 # Max norm for gradient clipping

# EGNN_HIDDEN_IRREPS_STR = "64x0e + 32x1o + 16x2e" # As defined in model.py's RealSpaceEGNNEncoder
# EGNN_RADIUS = 3.0 # Atomic interaction radius for EGNN

# # KSpace GNN specific parameters
# KSPACE_GNN_NUM_HEADS = 8 # As defined in model.py's KSpaceTransformerGNNEncoder

# # Loss weighting for multi-task learning
# #LOSS_WEIGHT_TOPOLOGY = 1.0
# #LOSS_WEIGHT_MAGNETISM = 1.0
# LOSS_WEIGHT_TOPOLOGY        =  1.0 # α: weight on classification term
# LOSS_WEIGHT_TOPO_CONSISTENCY = 1.0   # β: weight on topological‐consistency term
# LOSS_WEIGHT_REGULARIZATION   = 0.1   # γ: weight on topo‐feature regularization term
# LOSS_WEIGHT_MAGNETISM       = 0.4   # weight on the magnetism classification term

# # --- Device Configuration ---
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAIN_RATIO = 0.8
# VAL_RATIO = 0.1
# TEST_RATIO = 0.1 

# # --- Feature Dimensions 
# CRYSTAL_NODE_FEATURE_DIM = 3 
# ASPH_FEATURE_DIM = 3115 
# BAND_REP_FEATURE_DIM = 4756 
# KSPACE_GRAPH_NODE_FEATURE_DIM = 10 

# BASE_DECOMPOSITION_FEATURE_DIM = 2
# ALL_POSSIBLE_IRREPS = [] 
# MAX_DECOMPOSITION_INDICES_LEN = 100

# try:
#     with open('/scratch/gpfs/as0714/graph_vector_topological_insulator/multitask_ti_classification/irrep_unique', 'rb') as fp:
#         ALL_POSSIBLE_IRREPS = pickle.load(fp)
# except FileNotFoundError:
#     warnings.warn("irrep_unique file not found. ALL_POSSIBLE_IRREPS will be empty. Decomposition features might be impacted.")

# DECOMPOSITION_FEATURE_DIM = BASE_DECOMPOSITION_FEATURE_DIM + \
#                             len(ALL_POSSIBLE_IRREPS) + \
#                             MAX_DECOMPOSITION_INDICES_LEN

# # New feature dimensions for EnhancedKSpacePhysicsFeatures
# BAND_GAP_SCALAR_DIM = 1 # Single scalar band gap value
# DOS_FEATURE_DIM = 100   # Assuming a 100-point DOS vector if available
# FERMI_FEATURE_DIM = 50  # Assuming 50 descriptors for Fermi surface if available

# # Scalar Total Dim (now excludes band_gap as it's separate)
# # It only contains band_rep_features + other scalar metadata features
# SCALAR_TOTAL_DIM = BAND_REP_FEATURE_DIM + len([
#     'formation_energy', 'energy_above_hull', 'density', 'volume', 'nsites',
#     'space_group_number', 'total_magnetization'
# ])

# # --- Model Saving ---
# MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
# os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


#INTERESTING RESULT

# #with following config: THERE WAS 0.1 WEIGHTAGE ON MAGNETISM
# import os
# from pathlib import Path
# import torch
# import pickle
# import warnings

# PROJECT_ROOT = Path(__file__).resolve().parent

# SEED = 1

# # Base directory for your multimodal material database (output of IntegratedMaterialProcessor)
# DATA_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/multimodal_materials_db_mp")
# # Base directory for pre-generated k-space graphs (output of KSpacePhysicsGraphBuilder)
# KSPACE_GRAPHS_DIR = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/nonmagnetic_3d/kspace_topology_graphs")
# # Path to the master index directory (containing individual JSON metadata files)
# MASTER_INDEX_PATH = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/metadata")

# # Topology Classification - FIXED MAPPING
# TOPOLOGY_CLASS_MAPPING = {
#     "Trivial": 0,           # Trivial metal/insulator (materials not classified as TI or TSM)
#     "Topological Insulator": 1,
#     "Semimetal": 2,         # Includes "Weyl Semimetal", "Dirac Semimetal"
#     "Weyl Semimetal": 2,    # Explicitly map to Semimetal
#     "Dirac Semimetal": 2,   # Explicitly map to Semimetal
#     "Unknown": 0,           # Treat unknown topological types as trivial by default, or filter
# }
# NUM_TOPOLOGY_CLASSES = len(set(TOPOLOGY_CLASS_MAPPING.values())) # Should be 3 (Trivial, TI, SM)

# NUM_WORKERS = 8

# # Magnetism Classification
# # Ensure these match the 'magnetic_type' values from Materials Project (e.g., 'NM', 'FM', 'AFM', 'FiM')
# MAGNETISM_CLASS_MAPPING = {
#     "NM": 0,    # Non-magnetic
#     "FM": 1,    # Ferromagnetic
#     "AFM": 2,   # Antiferromagnetic
#     "FiM": 3,   # Ferrimagnetic
#     "UNKNOWN": 0, # Treat unknown magnetic types as non-magnetic for classification purposes
# }
# NUM_MAGNETISM_CLASSES = len(set(MAGNETISM_CLASS_MAPPING.values())) # Should be 4

# # --- Model Hyperparameters ---
# # Shared Encoder / Fusion
# LATENT_DIM_GNN = 512      # Output dimension of each GNN encoder

# # Specific latent dims for different FFNN types
# LATENT_DIM_ASPH = 3115
# LATENT_DIM_OTHER_FFNN = 1024 # Smaller dimension for other FFNNs (Scalar, Decomposition, Enhanced Physics output)

# FUSION_HIDDEN_DIMS = [256, 128] # Dimensions for shared fusion layers

# # GNN specific (Crystal Graph and K-space Graph)
# GNN_NUM_LAYERS = 6
# GNN_HIDDEN_CHANNELS = 512 # For node features within GNN layers

# # FFNN specific (ASPH and Scalar features)
# FFNN_HIDDEN_DIMS_ASPH = [128,64]
# FFNN_HIDDEN_DIMS_SCALAR = [64, 32]

# # Training Parameters
# LEARNING_RATE = 0.001
# BATCH_SIZE = 8 # Increased batch size for more stable gradients (e.g., 64, 128, or 256)
# NUM_EPOCHS = 50
# DROPOUT_RATE = 0.2
# PATIENCE = 10 # For early stopping
# MAX_GRAD_NORM = 3.0 # Max norm for gradient clipping

# EGNN_HIDDEN_IRREPS_STR = "64x0e + 32x1o + 16x2e" # As defined in model.py's RealSpaceEGNNEncoder
# EGNN_RADIUS = 3.0 # Atomic interaction radius for EGNN

# # KSpace GNN specific parameters
# KSPACE_GNN_NUM_HEADS = 8 # As defined in model.py's KSpaceTransformerGNNEncoder

# # Loss weighting for multi-task learning
# #LOSS_WEIGHT_TOPOLOGY = 1.0
# #LOSS_WEIGHT_MAGNETISM = 1.0
# LOSS_WEIGHT_TOPOLOGY        =  1.0 # α: weight on classification term
# LOSS_WEIGHT_TOPO_CONSISTENCY = 1.0   # β: weight on topological‐consistency term
# LOSS_WEIGHT_REGULARIZATION   = 1.0   # γ: weight on topo‐feature regularization term
# LOSS_WEIGHT_MAGNETISM       = 0.1   # weight on the magnetism classification term

# # --- Device Configuration ---
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAIN_RATIO = 0.8
# VAL_RATIO = 0.1
# TEST_RATIO = 0.1 

# # --- Feature Dimensions 
# CRYSTAL_NODE_FEATURE_DIM = 3 
# ASPH_FEATURE_DIM = 3115 
# BAND_REP_FEATURE_DIM = 4756 
# KSPACE_GRAPH_NODE_FEATURE_DIM = 10 

# BASE_DECOMPOSITION_FEATURE_DIM = 2
# ALL_POSSIBLE_IRREPS = [] 
# MAX_DECOMPOSITION_INDICES_LEN = 100

# try:
#     with open('/scratch/gpfs/as0714/graph_vector_topological_insulator/multitask_ti_classification/irrep_unique', 'rb') as fp:
#         ALL_POSSIBLE_IRREPS = pickle.load(fp)
# except FileNotFoundError:
#     warnings.warn("irrep_unique file not found. ALL_POSSIBLE_IRREPS will be empty. Decomposition features might be impacted.")

# DECOMPOSITION_FEATURE_DIM = BASE_DECOMPOSITION_FEATURE_DIM + \
#                             len(ALL_POSSIBLE_IRREPS) + \
#                             MAX_DECOMPOSITION_INDICES_LEN

# # New feature dimensions for EnhancedKSpacePhysicsFeatures
# BAND_GAP_SCALAR_DIM = 1 # Single scalar band gap value
# DOS_FEATURE_DIM = 100   # Assuming a 100-point DOS vector if available
# FERMI_FEATURE_DIM = 50  # Assuming 50 descriptors for Fermi surface if available

# # Scalar Total Dim (now excludes band_gap as it's separate)
# # It only contains band_rep_features + other scalar metadata features
# SCALAR_TOTAL_DIM = BAND_REP_FEATURE_DIM + len([
#     'formation_energy', 'energy_above_hull', 'density', 'volume', 'nsites',
#     'space_group_number', 'total_magnetization'
# ])

# # --- Model Saving ---
# MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
# os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

#but 89 percent, output is 65386280