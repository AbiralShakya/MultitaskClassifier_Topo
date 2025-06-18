import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as PyGDataLoader # For handling PyGData objects in batches
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
import json
import glob
import torch_geometric
import os # Added for os.path.join

# Import from local modules
import helper.config as config
# Assuming these exist and are correctly implemented:
from helper.topo_utils import SpaceGroupManager, load_material_graph_from_dict, load_pickle_data 

# --- GLOBAL SETTING FOR TORCH.LOAD SAFETY ---
# This line should be executed ONCE at the start of your program
# (e.g., in your main training script, or at the top of this module if it's imported early)
# This addresses the 'WeightsUnpickler error: Unsupported global: GLOBAL torch_geometric.data.data.DataEdgeAttr'
# It makes torch.load safer by explicitly allowing this type.
torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
# --- END GLOBAL SETTING ---


# Suppress specific warnings from pandas when mapping values that might not exist
warnings.filterwarnings("ignore", ".*is not in the top-level domain.*", UserWarning)


class MaterialDataset(Dataset):
    def __init__(self, master_index_path: Union[str, Path], kspace_graphs_base_dir: Union[str, Path],
                 data_root_dir: Union[str, Path], scaler: Optional[StandardScaler] = None):
        """
        Args:
            master_index_path: Path to the directory containing individual JSON metadata files.
            kspace_graphs_base_dir: Base directory where k-space graphs are stored by space group.
            data_root_dir: The root directory for the entire multimodal database (e.g., 'multimodal_materials_db_mp').
                           Used to resolve relative paths in master_index.
            scaler: A pre-fitted StandardScaler for numerical features. If None, the dataset can be initialized
                    without a scaler, and one can be fitted and assigned later.
        """
        self.metadata_json_dir = Path(master_index_path)
        self.kspace_graphs_base_dir = Path(kspace_graphs_base_dir)
        self.data_root_dir = Path(data_root_dir)

        if not self.metadata_json_dir.is_dir():
            raise NotADirectoryError(f"master_index_path must be a directory containing JSON files: {self.metadata_json_dir}")
        
        all_json_files = list(self.metadata_json_dir.glob("*.json"))
        if not all_json_files:
            raise FileNotFoundError(f"No JSON files found in the directory: {self.metadata_json_dir}")

        data_records = []
        for json_file_path in all_json_files:
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                record = {
                    'jid': json_data.get('jid'),
                    'formula': json_data.get('formula'),
                    'space_group': json_data.get('space_group'),
                    'space_group_number': json_data.get('space_group_number'),
                    'topological_class': json_data.get('topological_class'),
                    'topological_binary': json_data.get('topological_binary'),
                    'band_gap': json_data.get('band_gap'),
                    'formation_energy': json_data.get('formation_energy'),
                    'energy_above_hull': json_data.get('energy_above_hull'),
                    'density': json_data.get('density'),
                    'volume': json_data.get('volume'),
                    'nsites': json_data.get('nsites'),
                    'total_magnetization': json_data.get('total_magnetization'),
                    'magnetic_type': json_data.get('magnetic_type'),
                    'theoretical': json_data.get('theoretical'),
                    
                    'file_locations.structure_hdf5': json_data.get('file_locations', {}).get('structure_hdf5'),
                    'file_locations.point_cloud': json_data.get('file_locations', {}).get('point_cloud'),
                    'file_locations.crystal_graph': json_data.get('file_locations', {}).get('crystal_graph'),
                    'file_locations.kspace_graph_shared': json_data.get('file_locations', {}).get('kspace_graph_shared'),
                    'file_locations.vectorized_features_dir': json_data.get('file_locations', {}).get('vectorized_features_dir'),
                }
                data_records.append(record)
        
        self.metadata_df = pd.DataFrame(data_records)
        
        print("\n--- Debugging metadata_df in MaterialDataset init ---")
        print(f"Path to metadata directory: {self.metadata_json_dir}")
        print("Columns found in metadata_df (after loading JSONs):")
        print(self.metadata_df.columns.tolist())
        print("First few rows of metadata_df:")
        print(self.metadata_df.head())
        print("----------------------------------------------------\n")

        initial_count = len(self.metadata_df)
        self.metadata_df = self.metadata_df[
            self.metadata_df['topological_class'].isin(config.TOPOLOGY_CLASS_MAPPING.keys()) &
            self.metadata_df['magnetic_type'].isin(config.MAGNETISM_CLASS_MAPPING.keys())
        ].reset_index(drop=True)

        if len(self.metadata_df) < initial_count:
            print(f"Filtered out {initial_count - len(self.metadata_df)} materials due to undefined topological or magnetic types.")

        self.space_group_manager = SpaceGroupManager(self.kspace_graphs_base_dir)
        self.scaler = scaler

        self.topology_class_map = config.TOPOLOGY_CLASS_MAPPING
        self.magnetism_class_map = config.MAGNETISM_CLASS_MAPPING

        self.scalar_features_columns = [
            'band_gap', 'formation_energy', 'density', 'volume', 'nsites',
            'space_group_number', 'total_magnetization', 'energy_above_hull'
        ]
        
        # --- NEW: Define all possible irreps and max decomposition indices length
        # These should be determined by scanning your dataset or from domain knowledge.
        # Ensure 'config' holds these values or derive them robustly here.
        self.all_possible_irreps = getattr(config, 'ALL_POSSIBLE_IRREPS', [])
        if not self.all_possible_irreps:
            # Fallback: if not in config, you might need to build it dynamically
            # or raise an error for the user to define it.
            # For robust training, this list should be fixed and comprehensive.
            warnings.warn("config.ALL_POSSIBLE_IRREPS is not set. Using a limited default. This may cause issues.")
            self.all_possible_irreps = sorted([
                "R1", "T1", "U1", "V1", "X1", "Y1", "Z1", "Γ1", "GP1",
                "R2R2", "T2T2", "U2U2", "V2V2", "X2X2", "Y2Y2", "Z2Z2", "Γ2Γ2", "2GP2"
            ])

        self.max_decomposition_indices_len = getattr(config, 'MAX_DECOMPOSITION_INDICES_LEN', 5)
        
        # Calculate expected final decomposition feature dim (for dummy data and model init)
        # This relies on BASE_DECOMPOSITION_FEATURE_DIM being defined in config
        self._expected_decomposition_feature_dim = getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 0) + \
                                                   len(self.all_possible_irreps) + \
                                                   self.max_decomposition_indices_len
        
        # Override config's DECOMPOSITION_FEATURE_DIM with the calculated one if it's not set
        if not hasattr(config, 'DECOMPOSITION_FEATURE_DIM') or config.DECOMPOSITION_FEATURE_DIM is None:
             config.DECOMPOSITION_FEATURE_DIM = self._expected_decomposition_feature_dim
        elif config.DECOMPOSITION_FEATURE_DIM != self._expected_decomposition_feature_dim:
             warnings.warn(f"Config DECOMPOSITION_FEATURE_DIM ({config.DECOMPOSITION_FEATURE_DIM}) does not match calculated ({self._expected_decomposition_feature_dim}). Using calculated.")
             config.DECOMPOSITION_FEATURE_DIM = self._expected_decomposition_feature_dim

        # Initial feature dimensions (will be used by model for init, no longer dynamically set in __getitem__)
        self._crystal_node_feature_dim = getattr(config, 'CRYSTAL_NODE_FEATURE_DIM', 0)
        self._kspace_graph_node_feature_dim = getattr(config, 'KSPACE_GRAPH_NODE_FEATURE_DIM', 0)
        self._asph_feature_dim = getattr(config, 'ASPH_FEATURE_DIM', 0)
        self._scalar_total_dim = getattr(config, 'SCALAR_TOTAL_DIM', len(self.scalar_features_columns) + getattr(config, 'BAND_REP_FEATURE_DIM', 0)) # Initial estimate
        # Make sure these are set correctly in your config based on real data
        
    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads and processes data for a single material at the given index.
        """
        row = self.metadata_df.iloc[idx]
        
        # --- 1. Load Crystal Graph ---
        crystal_graph_path = Path('/scratch/gpfs/as0714/graph_vector_topological_insulator/crystal_graphs') / row['jid'] / 'crystal_graph.pkl'
        crystal_graph_dict = load_pickle_data(crystal_graph_path)
        crystal_graph = load_material_graph_from_dict(crystal_graph_dict)
        
        # --- 2. Load ASPH Features ---
        asph_features_path = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/vectorized_features") / row['jid'] / "asph_features.npy"
        asph_features = torch.tensor(np.load(asph_features_path), dtype=torch.float)


        # --- 3. Load K-space Graph and related Physics Features ---
        sg_number = row['space_group_number'] 
        kspace_sg_folder = self.kspace_graphs_base_dir / f"SG_{str(int(sg_number)).zfill(3)}"

        # Load kspace_graph.pt
        kspace_graph_path = kspace_sg_folder / 'kspace_graph.pt'
        kspace_graph = None
        try:
            kspace_graph = torch.load(kspace_graph_path) # No weights_only=False needed due to global setting
        except Exception as e:
            print(f"Warning: Could not load k-space graph for SG {sg_number} (JID: {row['jid']}) from {kspace_graph_path}: {e}")
            kspace_graph = self._generate_dummy_kspace_graph()

        # Load base physics_features.pt
        base_physics_features_path = kspace_sg_folder / 'physics_features.pt'
        base_decomposition_features_tensor = None
        try:
            # Assuming physics_features.pt contains a single tensor, or can be converted
            loaded_data = torch.load(base_physics_features_path)
            if isinstance(loaded_data, dict) and 'decomposition_features' in loaded_data:
                base_decomposition_features_tensor = loaded_data['decomposition_features']
            elif isinstance(loaded_data, torch.Tensor):
                base_decomposition_features_tensor = loaded_data
            else:
                warnings.warn(f"Unexpected data type in {base_physics_features_path}. Expected dict with 'decomposition_features' or a tensor.")
                base_decomposition_features_tensor = torch.zeros(getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 1), dtype=torch.float32)

            # Ensure it's a 1D tensor of correct size
            if base_decomposition_features_tensor.ndim == 0:
                base_decomposition_features_tensor = torch.tensor([base_decomposition_features_tensor.item()])
            elif base_decomposition_features_tensor.ndim > 1:
                base_decomposition_features_tensor = base_decomposition_features_tensor.squeeze()
            
            if base_decomposition_features_tensor.numel() != getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', base_decomposition_features_tensor.numel()):
                warnings.warn(f"Loaded base decomposition features for {row['jid']} (SG {sg_number}) have wrong dim {base_decomposition_features_tensor.numel()}, expected {getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 0)}. Using dummy.")
                base_decomposition_features_tensor = torch.zeros(getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 1), dtype=torch.float32)

        except Exception as e:
            print(f"Warning: Could not load base physics features for SG {sg_number} (JID: {row['jid']}) from {base_physics_features_path}: {e}")
            base_decomposition_features_tensor = torch.zeros(getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 1), dtype=torch.float32)


        # Load SG-specific metadata.json for EBR and Decomposition Branches
        sg_metadata_json_path = kspace_sg_folder / 'metadata.json'
        sg_metadata = {}
        try:
            with open(sg_metadata_json_path, 'r') as f:
                sg_metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load SG metadata for SG {sg_number} (JID: {row['jid']}) from {sg_metadata_json_path}: {e}. Using empty metadata.")

        # Process EBR data
        ebr_features_vec = torch.zeros(len(self.all_possible_irreps), dtype=torch.float32)
        if 'ebr_data' in sg_metadata and 'irrep_multiplicities' in sg_metadata['ebr_data']:
            multiplicities = sg_metadata['ebr_data']['irrep_multiplicities']
            processed_multiplicities = {k.replace('\u0393', 'Γ'): v for k, v in multiplicities.items()}
            for i, irrep_name in enumerate(self.all_possible_irreps):
                if irrep_name in processed_multiplicities:
                    ebr_features_vec[i] = processed_multiplicities[irrep_name]
        
        # Process Decomposition Branches
        sg_decomposition_indices_tensor = torch.zeros(self.max_decomposition_indices_len, dtype=torch.float32)
        if 'decomposition_branches' in sg_metadata and 'decomposition_indices' in sg_metadata['decomposition_branches']:
            indices_list = sg_metadata['decomposition_branches']['decomposition_indices']
            temp_tensor = torch.tensor(indices_list, dtype=torch.float32)
            num_elements_to_copy = min(temp_tensor.numel(), self.max_decomposition_indices_len)
            sg_decomposition_indices_tensor[:num_elements_to_copy] = temp_tensor[:num_elements_to_copy]

        # Combine all decomposition-related features
        # Ensure all components have at least 1 dimension for concatenation
        full_decomposition_features = torch.cat([
            base_decomposition_features_tensor,
            ebr_features_vec,
            sg_decomposition_indices_tensor
        ])
        
        # Ensure final concatenated feature matches expected dimension
        if full_decomposition_features.numel() != self._expected_decomposition_feature_dim:
            warnings.warn(f"Final decomposition feature dim mismatch for {row['jid']} (SG {sg_number}). Expected {self._expected_decomposition_feature_dim}, got {full_decomposition_features.numel()}. Adjusting.")
            if full_decomposition_features.numel() < self._expected_decomposition_feature_dim:
                padding = torch.zeros(self._expected_decomposition_feature_dim - full_decomposition_features.numel(), dtype=torch.float32)
                full_decomposition_features = torch.cat([full_decomposition_features, padding])
            else:
                full_decomposition_features = full_decomposition_features[:self._expected_decomposition_feature_dim]

        # Consolidated kspace_physics_features dictionary for the model
        kspace_physics_features_dict = {'decomposition_features': full_decomposition_features}


        # --- 4. Extract Scalar Features (Band Reps + Metadata) ---
        band_rep_features_path = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/vectorized_features") / row['jid'] / 'band_rep_features.npy'
        band_rep_features = torch.tensor(np.load(band_rep_features_path), dtype=torch.float)
        
        scalar_metadata_features = [row[col] for col in self.scalar_features_columns]
        scalar_metadata_features = [0.0 if pd.isna(val) else val for val in scalar_metadata_features]
        scalar_metadata_features = torch.tensor(scalar_metadata_features, dtype=torch.float)

        combined_scalar_features = torch.cat([band_rep_features, scalar_metadata_features])
        
        if self.scaler:
            # Ensure scaler handles 1D features correctly (unsqueeze for transform, squeeze back)
            if combined_scalar_features.ndim == 1:
                combined_scalar_features = torch.tensor(self.scaler.transform(combined_scalar_features.unsqueeze(0)).squeeze(0), dtype=torch.float)
            else: # If it's somehow already 2D (batch dim), transform directly
                 combined_scalar_features = torch.tensor(self.scaler.transform(combined_scalar_features), dtype=torch.float)

        # --- 5. Prepare Labels ---
        topology_label_str = row['topological_class']
        topology_label = torch.tensor(self.topology_class_map.get(topology_label_str, self.topology_class_map["Unknown"]), dtype=torch.long)
        
        magnetism_label_str = row['magnetic_type']
        magnetism_label = torch.tensor(self.magnetism_class_map.get(magnetism_label_str, self.magnetism_class_map["UNKNOWN"]), dtype=torch.long)

        # --- Set Feature Dimensions in Config for Model Initialization (if not already set) ---
        # This is for the *first* item load. Subsequent loads won't re-run this.
        # This ensures the model's __init__ gets correct dimensions from config.
        if config.CRYSTAL_NODE_FEATURE_DIM is None or config.CRYSTAL_NODE_FEATURE_DIM == 0:
            config.CRYSTAL_NODE_FEATURE_DIM = crystal_graph.x.shape[1]
        if config.KSPACE_GRAPH_NODE_FEATURE_DIM is None or config.KSPACE_GRAPH_NODE_FEATURE_DIM == 0:
            config.KSPACE_GRAPH_NODE_FEATURE_DIM = kspace_graph.x.shape[1]
        if config.ASPH_FEATURE_DIM is None or config.ASPH_FEATURE_DIM == 0:
            config.ASPH_FEATURE_DIM = asph_features.shape[0]
        if config.SCALAR_TOTAL_DIM is None or config.SCALAR_TOTAL_DIM == 0:
            config.SCALAR_TOTAL_DIM = combined_scalar_features.shape[0]
        # DECOMPOSITION_FEATURE_DIM is set in __init__ for consistency.


        return {
            'crystal_graph': crystal_graph,
            'kspace_graph': kspace_graph,
            'asph_features': asph_features,
            'scalar_features': combined_scalar_features,
            'kspace_physics_features': kspace_physics_features_dict, # Use the consolidated dict
            'topology_label': topology_label,
            'magnetism_label': magnetism_label,
            'jid': row['jid']
        }

    # --- Dummy Data Generation Methods ---
    # Ensure these reflect the dimensions set in config.py
    def _generate_dummy_crystal_graph(self):
        num_nodes_dummy = 10
        # Use config value, fall back to a sensible default if not set for safety
        crystal_node_feature_dim = getattr(config, 'CRYSTAL_NODE_FEATURE_DIM', 92) 
        x_dummy = torch.randn(num_nodes_dummy, crystal_node_feature_dim)
        pos_dummy = torch.randn(num_nodes_dummy, 3)
        edge_index_dummy = torch.randint(0, num_nodes_dummy, (2, 20))
        batch_dummy = torch.zeros(num_nodes_dummy, dtype=torch.long)
        return PyGData(x=x_dummy, pos=pos_dummy, edge_index=edge_index_dummy, batch=batch_dummy)

    def _generate_dummy_kspace_graph(self):
        num_nodes_dummy = 5
        kspace_node_feature_dim = getattr(config, 'KSPACE_GRAPH_NODE_FEATURE_DIM', 10)
        x_dummy = torch.randn(num_nodes_dummy, kspace_node_feature_dim)
        edge_index_dummy = torch.randint(0, num_nodes_dummy, (2, 8))
        batch_dummy = torch.zeros(num_nodes_dummy, dtype=torch.long)
        return PyGData(x=x_dummy, edge_index=edge_index_dummy, batch=batch_dummy)

    def _generate_dummy_asph_features(self):
        asph_feature_dim = getattr(config, 'ASPH_FEATURE_DIM', 128)
        return torch.randn(asph_feature_dim)

    # Note: _generate_dummy_physics_features is no longer explicitly called for the *combined*
    # features. Instead, individual dummy parts are generated for BASE, EBR, SG_DECOMP_INDICES.
    # This method is for the *base* part from physics_features.pt
    def _generate_dummy_base_decomposition_features(self):
        base_decomposition_feature_dim = getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 100)
        return torch.randn(base_decomposition_feature_dim)


# Custom collate function for PyGDataLoader to handle dictionary of Data objects and other tensors
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for PyGDataLoader to handle a dictionary of inputs.
    It will batch PyGData objects separately and stack other tensors.
    """
    if not batch:
        return {}

    # Batch crystal graphs
    # DataLoader(dataset).dataset will give the original items. We need to create a new DataLoader
    # for each PyG list to properly batch them using PyG's internal batching logic.
    crystal_graphs_batch = torch_geometric.data.Batch.from_data_list([d['crystal_graph'] for d in batch])
    kspace_graphs_batch = torch_geometric.data.Batch.from_data_list([d['kspace_graph'] for d in batch])

    collated_batch = {
        'crystal_graph': crystal_graphs_batch,
        'kspace_graph': kspace_graphs_batch,
        'asph_features': torch.stack([d['asph_features'] for d in batch]),
        'scalar_features': torch.stack([d['scalar_features'] for d in batch]),
        'topology_label': torch.stack([d['topology_label'] for d in batch]),
        'magnetism_label': torch.stack([d['magnetism_label'] for d in batch]),
        'jid': [d['jid'] for d in batch]
    }

    # Handle kspace_physics_features which is a dict of tensors
    kspace_physics_features_collated = defaultdict(list)
    for d in batch:
        for key, tensor in d['kspace_physics_features'].items():
            kspace_physics_features_collated[key].append(tensor)
    
    # Ensure all collected tensors for a key are stacked
    for key in kspace_physics_features_collated:
        kspace_physics_features_collated[key] = torch.stack(kspace_physics_features_collated[key])
    collated_batch['kspace_physics_features'] = kspace_physics_features_collated

    return collated_batch