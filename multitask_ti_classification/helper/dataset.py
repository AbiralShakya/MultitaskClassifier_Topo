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
import json # Added import for JSON handling
import glob # Added import for finding files

# Import from local modules
import helper.config as config
from helper.topo_utils import SpaceGroupManager, load_material_graph_from_dict, load_pickle_data

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
        self.metadata_json_dir = Path(master_index_path) # Renamed for clarity: this is now a directory
        self.kspace_graphs_base_dir = Path(kspace_graphs_base_dir)
        self.data_root_dir = Path(data_root_dir)

        if not self.metadata_json_dir.is_dir():
            raise NotADirectoryError(f"master_index_path must be a directory containing JSON files: {self.metadata_json_dir}")
        
        # --- NEW CODE TO READ MULTIPLE JSON FILES FROM DIRECTORY ---
        all_json_files = list(self.metadata_json_dir.glob("*.json")) # Finds all .json files in the directory
        if not all_json_files:
            raise FileNotFoundError(f"No JSON files found in the directory: {self.metadata_json_dir}")

        data_records = []
        for json_file_path in all_json_files:
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                
                # Flatten the relevant parts of the JSON into a single dictionary record.
                # This is crucial because Pandas DataFrames expect flat columns.
                # The keys here ('jid', 'topological_class', 'file_locations.crystal_graph', etc.)
                # will become the column names in your metadata_df.
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
                    
                    # Flatten 'file_locations' nested dictionary
                    'file_locations.structure_hdf5': json_data.get('file_locations', {}).get('structure_hdf5'),
                    'file_locations.point_cloud': json_data.get('file_locations', {}).get('point_cloud'),
                    'file_locations.crystal_graph': json_data.get('file_locations', {}).get('crystal_graph'),
                    'file_locations.kspace_graph_shared': json_data.get('file_locations', {}).get('kspace_graph_shared'),
                    'file_locations.vectorized_features_dir': json_data.get('file_locations', {}).get('vectorized_features_dir'),

                    # Add other top-level keys or flattened nested keys if needed for scalar_features_columns
                    # e.g., if you need anything from 'electronic_structure' or 'mechanical_properties'
                }
                data_records.append(record)
        
        self.metadata_df = pd.DataFrame(data_records)
        # --- END NEW CODE ---
        
        # --- DEBUG PRINTS (KEEP THESE IN FOR NOW TO VERIFY) ---
        print("\n--- Debugging metadata_df in MaterialDataset init ---")
        print(f"Path to metadata directory: {self.metadata_json_dir}")
        print("Columns found in metadata_df (after loading JSONs):")
        print(self.metadata_df.columns.tolist()) # Convert to list for cleaner print
        print("First few rows of metadata_df:")
        print(self.metadata_df.head())
        print("----------------------------------------------------\n")
        # --- END DEBUG PRINTS ---

        # Filter out materials that don't have valid labels in our defined mappings
        initial_count = len(self.metadata_df)
        
        # These column accesses are now correct because we flattened the JSONs
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

        # Define which scalar features to extract from the metadata.df
        # These columns should now exist directly in your metadata_df after flattening
        self.scalar_features_columns = [
            'band_gap', 'formation_energy', 'density', 'volume', 'nsites',
            'space_group_number', 'total_magnetization', 'energy_above_hull'
        ]
        
        # Initialize feature dimensions (will be updated dynamically by train.py after first item load)
        self._crystal_node_feature_dim = None
        self._kspace_graph_node_feature_dim = None
        self._asph_feature_dim = None
        self._scalar_total_dim = None
        self._decomposition_feature_dim = None 

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads and processes data for a single material at the given index.
        """
        row = self.metadata_df.iloc[idx]
        
        # --- 1. Load Crystal Graph ---
        # These nested keys are now treated as flat column names due to JSON flattening
        #crystal_graph_path = self.data_root_dir / row['file_locations.crystal_graph']
        crystal_graph_path = Path('/scratch/gpfs/as0714/graph_vector_topological_insulator/crystal_graphs') / row['jid'] / 'crystal_graph.pkl'
        crystal_graph_dict = load_pickle_data(crystal_graph_path)
        crystal_graph = load_material_graph_from_dict(crystal_graph_dict)
        
        # --- 2. Load ASPH Features ---
        asph_features_path =  Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/vectorized_features") /row['jid'] / "asph_features.npy"
        asph_features = torch.tensor(np.load(asph_features_path), dtype=torch.float)


        # --- 3. Load K-space Graph and Physics Features (from shared files) ---
        sg_number = row['space_group_number'] 
        kspace_data_tuple = self.space_group_manager.load_kspace_data(sg_number)

        kspace_graph = None
        kspace_physics_features = None

        if kspace_data_tuple:
            kspace_graph, kspace_physics_features = kspace_data_tuple
        else:
            print(f"Warning: Missing shared k-space graph/features for SG {sg_number} (JID: {row['jid']}). Using dummy data.")
            dummy_node_dim = self._kspace_graph_node_feature_dim if self._kspace_graph_node_feature_dim else config.KSPACE_GRAPH_NODE_FEATURE_DIM
            dummy_global_dim = config.BAND_REP_FEATURE_DIM + 3 + 1
            
            kspace_graph = PyGData(x=torch.zeros(1, dummy_node_dim),
                                   edge_index=torch.empty(2, 0, dtype=torch.long),
                                   u=torch.zeros(1, dummy_global_dim))
            kspace_physics_features = {
                'ebr_features': torch.zeros(5, dtype=torch.float),
                'topological_indices': torch.zeros(5, dtype=torch.float),
                'decomposition_features': torch.zeros(config.DECOMPOSITION_FEATURE_DIM if config.DECOMPOSITION_FEATURE_DIM else 5, dtype=torch.float)
            }


        # --- 4. Extract Scalar Features (Band Reps + Metadata) ---
        band_rep_features_path =  Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/vectorized_features") / row['jid'] / 'band_rep_features.npy'
        band_rep_features = torch.tensor(np.load(band_rep_features_path), dtype=torch.float)
        
        scalar_metadata_features = [row[col] for col in self.scalar_features_columns]
        scalar_metadata_features = [0.0 if pd.isna(val) else val for val in scalar_metadata_features]
        scalar_metadata_features = torch.tensor(scalar_metadata_features, dtype=torch.float)

        combined_scalar_features = torch.cat([band_rep_features, scalar_metadata_features])
        
        if self.scaler:
            combined_scalar_features = torch.tensor(self.scaler.transform(combined_scalar_features.unsqueeze(0)).squeeze(0), dtype=torch.float)
        
        # Dynamically set feature dimensions after the first item is loaded
        if self._crystal_node_feature_dim is None:
            self._crystal_node_feature_dim = crystal_graph.x.shape[1]
            config.CRYSTAL_NODE_FEATURE_DIM = self._crystal_node_feature_dim
        if self._kspace_graph_node_feature_dim is None:
            self._kspace_graph_node_feature_dim = kspace_graph.x.shape[1]
            config.KSPACE_GRAPH_NODE_FEATURE_DIM = self._kspace_graph_node_feature_dim
        if self._asph_feature_dim is None:
            self._asph_feature_dim = asph_features.shape[0]
            config.ASPH_FEATURE_DIM = self._asph_feature_dim
        if self._scalar_total_dim is None:
            self._scalar_total_dim = combined_scalar_features.shape[0]
            config.SCALAR_TOTAL_DIM = self._scalar_total_dim
        if self._decomposition_feature_dim is None:
            self._decomposition_feature_dim = kspace_physics_features['decomposition_features'].shape[0]
            config.DECOMPOSITION_FEATURE_DIM = self._decomposition_feature_dim

        # --- 5. Prepare Labels ---
        topology_label_str = row['topological_class']
        topology_label = torch.tensor(self.topology_class_map.get(topology_label_str, self.topology_class_map["Unknown"]), dtype=torch.long)
        
        magnetism_label_str = row['magnetic_type']
        magnetism_label = torch.tensor(self.magnetism_class_map.get(magnetism_label_str, self.magnetism_class_map["UNKNOWN"]), dtype=torch.long)

        return {
            'crystal_graph': crystal_graph,
            'kspace_graph': kspace_graph,
            'asph_features': asph_features,
            'scalar_features': combined_scalar_features,
            'kspace_physics_features': kspace_physics_features,
            'topology_label': topology_label,
            'magnetism_label': magnetism_label,
            'jid': row['jid']
        }

# Custom collate function for PyGDataLoader to handle dictionary of Data objects and other tensors
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for PyGDataLoader to handle a dictionary of inputs.
    It will batch PyGData objects separately and stack other tensors.
    """
    if not batch:
        return {}

    # Batch crystal graphs
    crystal_graphs_batch = PyGDataLoader(
        [d['crystal_graph'] for d in batch],
        batch_size=len(batch) 
    ).dataset 
    # Batch kspace graphs
    kspace_graphs_batch = PyGDataLoader(
        [d['kspace_graph'] for d in batch],
        batch_size=len(batch)
    ).dataset

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
    
    for key in kspace_physics_features_collated:
        kspace_physics_features_collated[key] = torch.stack(kspace_physics_features_collated[key])
    collated_batch['kspace_physics_features'] = kspace_physics_features_collated

    return collated_batch