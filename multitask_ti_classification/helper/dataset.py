import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as PyGDataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter # Import Counter for debugging
import warnings
import json
import glob
import torch_geometric
import os 

# Import from local modules
import helper.config as config
from helper.topo_utils import SpaceGroupManager, load_material_graph_from_dict, load_pickle_data 

# --- GLOBAL SETTING FOR TORCH.LOAD SAFETY ---
torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage 
])

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
        self.data_root_dir = Path(data_root_dir) # Corrected back to data_root_dir, assuming that was the original variable name.

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
        # Scaler is now a dictionary to potentially hold separate scalers
        self.scaler = scaler 

        self.topology_class_map = config.TOPOLOGY_CLASS_MAPPING
        self.magnetism_class_map = config.MAGNETISM_CLASS_MAPPING

        # Updated scalar features columns, now excluding 'band_gap' if it's extracted separately
        self.scalar_features_columns = [
            'formation_energy', 'energy_above_hull', 'density', 'volume', 'nsites',
            'space_group_number', 'total_magnetization'
        ]
        
        self.all_possible_irreps = getattr(config, 'ALL_POSSIBLE_IRREPS', [])
        if not self.all_possible_irreps:
            warnings.warn("config.ALL_POSSIBLE_IRREPS is not set. Using a limited default. This may cause issues.")
            # Default to some common irreps if the file is missing
            self.all_possible_irreps = sorted([
                "R1", "T1", "U1", "V1", "X1", "Y1", "Z1", "Γ1", "GP1",
                "R2R2", "T2T2", "U2U2", "V2V2", "X2X2", "Y2Y2", "Z2Z2", "Γ2Γ2", "2GP2"
            ])

        self.max_decomposition_indices_len = getattr(config, 'MAX_DECOMPOSITION_INDICES_LEN', 5)
        
        self._expected_decomposition_feature_dim = getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 0) + \
                                                   len(self.all_possible_irreps) + \
                                                   self.max_decomposition_indices_len
        
        if not hasattr(config, 'DECOMPOSITION_FEATURE_DIM') or config.DECOMPOSITION_FEATURE_DIM is None:
             config.DECOMPOSITION_FEATURE_DIM = self._expected_decomposition_feature_dim
        elif config.DECOMPOSITION_FEATURE_DIM != self._expected_decomposition_feature_dim:
             warnings.warn(f"Config DECOMPOSITION_FEATURE_DIM ({config.DECOMPOSITION_FEATURE_DIM}) does not match calculated ({self._expected_decomposition_feature_dim}). Using calculated.")
             config.DECOMPOSITION_FEATURE_DIM = self._expected_decomposition_feature_dim

        # Ensure these are initialized only if they aren't already explicitly set in config
        # and if a sample is available to infer from.
        self._crystal_node_feature_dim = getattr(config, 'CRYSTAL_NODE_FEATURE_DIM', 0)
        self._kspace_graph_node_feature_dim = getattr(config, 'KSPACE_GRAPH_NODE_FEATURE_DIM', 0)
        self._asph_feature_dim = getattr(config, 'ASPH_FEATURE_DIM', 0)
        self._scalar_total_dim = getattr(config, 'SCALAR_TOTAL_DIM', len(self.scalar_features_columns) + getattr(config, 'BAND_REP_FEATURE_DIM', 0)) # Excludes band_gap here.
        
    def __len__(self) -> int:
        return len(self.metadata_df)

    def _check_and_handle_nan_inf(self, tensor: torch.Tensor, feature_name: str, jid: str) -> torch.Tensor:
        """Helper to check for NaN/Inf and replace them with zeros, warning if found."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            warnings.warn(f"NaN/Inf detected in {feature_name} for JID {jid}. Replacing with zeros.")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata_df.iloc[idx]
        jid = row['jid'] 

        # --- 1. Load Crystal Graph ---
        crystal_graph_path = Path('/scratch/gpfs/as0714/graph_vector_topological_insulator/crystal_graphs') / jid / 'crystal_graph.pkl'
        crystal_graph_dict = load_pickle_data(crystal_graph_path)
        crystal_graph = load_material_graph_from_dict(crystal_graph_dict)
        crystal_graph.x = self._check_and_handle_nan_inf(crystal_graph.x, f"crystal_graph.x", jid)
        crystal_graph.pos = self._check_and_handle_nan_inf(crystal_graph.pos, f"crystal_graph.pos", jid)
        crystal_graph.edge_attr = self._check_and_handle_nan_inf(crystal_graph.edge_attr, f"crystal_graph.edge_attr", jid)
        
        # --- 2. Load ASPH Features ---
        asph_features_path = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/vectorized_features") / jid / "asph_features_rev2.npy"
        try:
            asph_features_np = np.load(asph_features_path)
            asph_features = torch.tensor(asph_features_np, dtype=torch.float)
            asph_features = self._check_and_handle_nan_inf(asph_features, f"asph_features", jid)
        except Exception as e:
            warnings.warn(f"Could not load ASPH features for JID {jid} from {asph_features_path}: {e}. Returning zeros.")
            asph_features = torch.zeros(getattr(config, 'ASPH_FEATURE_DIM', 3115), dtype=torch.float)


        # --- 3. Load K-space Graph and related Physics Features ---
        sg_number = row['space_group_number'] 
        kspace_sg_folder = self.kspace_graphs_base_dir / f"SG_{str(int(sg_number)).zfill(3)}"

        # Load kspace_graph.pt
        kspace_graph_path = kspace_sg_folder / 'kspace_graph.pt'
        kspace_graph = None
        try:
            # Assuming kspace_graph.pt might contain pos and symmetry_labels
            kspace_graph = torch.load(kspace_graph_path)
            kspace_graph.x = self._check_and_handle_nan_inf(kspace_graph.x, f"kspace_graph.x", jid)
            if hasattr(kspace_graph, 'pos') and kspace_graph.pos is not None:
                kspace_graph.pos = self._check_and_handle_nan_inf(kspace_graph.pos, f"kspace_graph.pos", jid)
            if hasattr(kspace_graph, 'edge_attr') and kspace_graph.edge_attr is not None:
                 kspace_graph.edge_attr = self._check_and_handle_nan_inf(kspace_graph.edge_attr, f"kspace_graph.edge_attr", jid)
            if hasattr(kspace_graph, 'u') and kspace_graph.u is not None: # global features
                 kspace_graph.u = self._check_and_handle_nan_inf(kspace_graph.u, f"kspace_graph.u", jid)
            # Add symmetry_labels if present in the kspace_graph object itself or another source
            # For now, assuming symmetry_labels might be an attribute of PyGData if provided by your graph builder
            if not hasattr(kspace_graph, 'symmetry_labels'):
                kspace_graph.symmetry_labels = None # Explicitly set to None if not loaded

        except Exception as e:
            warnings.warn(f"Could not load k-space graph for SG {sg_number} (JID: {jid}) from {kspace_graph_path}: {e}. Returning dummy graph.")
            kspace_graph = self._generate_dummy_kspace_graph()
            kspace_graph.pos = torch.zeros((kspace_graph.num_nodes if hasattr(kspace_graph, 'num_nodes') else 5), 3, dtype=torch.float) # Ensure dummy has pos
            kspace_graph.symmetry_labels = None


        # Load base physics_features.pt (decomposition features)
        base_physics_features_path = kspace_sg_folder / 'physics_features.pt'
        base_decomposition_features_tensor = None
        try:
            loaded_data = torch.load(base_physics_features_path)
            if isinstance(loaded_data, dict) and 'decomposition_features' in loaded_data:
                base_decomposition_features_tensor = loaded_data['decomposition_features']
            elif isinstance(loaded_data, torch.Tensor):
                base_decomposition_features_tensor = loaded_data
            else:
                warnings.warn(f"Unexpected data type in {base_physics_features_path} for JID {jid}. Expected dict or tensor.")
                base_decomposition_features_tensor = torch.zeros(getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 2), dtype=torch.float32)

            base_decomposition_features_tensor = self._check_and_handle_nan_inf(base_decomposition_features_tensor, f"base_decomposition_features", jid)
            
            if base_decomposition_features_tensor.ndim == 0:
                base_decomposition_features_tensor = torch.tensor([base_decomposition_features_tensor.item()])
            elif base_decomposition_features_tensor.ndim > 1:
                base_decomposition_features_tensor = base_decomposition_features_tensor.squeeze()
            
            if base_decomposition_features_tensor.numel() != getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', base_decomposition_features_tensor.numel()):
                warnings.warn(f"Loaded base decomposition features for {jid} (SG {sg_number}) have wrong dim {base_decomposition_features_tensor.numel()}, expected {getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 0)}. Using dummy.")
                base_decomposition_features_tensor = torch.zeros(getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 2), dtype=torch.float32)

        except Exception as e:
            warnings.warn(f"Could not load base physics features for SG {sg_number} (JID: {jid}) from {base_physics_features_path}: {e}. Returning zeros.")
            base_decomposition_features_tensor = torch.zeros(getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 2), dtype=torch.float32)


        # Load SG-specific metadata.json for EBR and Decomposition Branches
        sg_metadata_json_path = kspace_sg_folder / 'metadata.json'
        sg_metadata = {}
        try:
            with open(sg_metadata_json_path, 'r') as f:
                sg_metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            warnings.warn(f"Could not load SG metadata for SG {sg_number} (JID: {jid}) from {sg_metadata_json_path}: {e}. Using empty metadata.")

        # Process EBR data
        ebr_features_vec = torch.zeros(len(self.all_possible_irreps), dtype=torch.float32)
        if 'ebr_data' in sg_metadata and 'irrep_multiplicities' in sg_metadata['ebr_data']:
            multiplicities = sg_metadata['ebr_data']['irrep_multiplicities']
            processed_multiplicities = {k.replace('\u0393', 'Γ'): v for k, v in multiplicities.items()}
            for i, irrep_name in enumerate(self.all_possible_irreps):
                if irrep_name in processed_multiplicities:
                    ebr_features_vec[i] = processed_multiplicities[irrep_name]
        ebr_features_vec = self._check_and_handle_nan_inf(ebr_features_vec, f"ebr_features_vec", jid)
        
        # Process Decomposition Branches
        sg_decomposition_indices_tensor = torch.zeros(self.max_decomposition_indices_len, dtype=torch.float32)
        if 'decomposition_branches' in sg_metadata and 'decomposition_indices' in sg_metadata['decomposition_branches']:
            indices_list = sg_metadata['decomposition_branches']['decomposition_indices']
            temp_tensor = torch.tensor(indices_list, dtype=torch.float32)
            num_elements_to_copy = min(temp_tensor.numel(), self.max_decomposition_indices_len)
            sg_decomposition_indices_tensor[:num_elements_to_copy] = temp_tensor[:num_elements_to_copy]
        sg_decomposition_indices_tensor = self._check_and_handle_nan_inf(sg_decomposition_indices_tensor, f"sg_decomposition_indices_tensor", jid)


        # Combine all decomposition-related features for the `decomposition_features` input
        full_decomposition_features_tensor = torch.cat([
            base_decomposition_features_tensor,
            ebr_features_vec,
            sg_decomposition_indices_tensor
        ])
        
        if full_decomposition_features_tensor.numel() != self._expected_decomposition_feature_dim:
            warnings.warn(f"Final decomposition feature dim mismatch for {jid} (SG {sg_number}). Expected {self._expected_decomposition_feature_dim}, got {full_decomposition_features_tensor.numel()}. Adjusting.")
            if full_decomposition_features_tensor.numel() < self._expected_decomposition_feature_dim:
                padding = torch.zeros(self._expected_decomposition_feature_dim - full_decomposition_features_tensor.numel(), dtype=torch.float32)
                full_decomposition_features_tensor = torch.cat([full_decomposition_features_tensor, padding])
            else:
                full_decomposition_features_tensor = full_decomposition_features_tensor[:self._expected_decomposition_feature_dim]
        full_decomposition_features_tensor = self._check_and_handle_nan_inf(full_decomposition_features_tensor, f"full_decomposition_features", jid)


        # --- NEW: Extract specific gap, DOS, Fermi features ---
        # gap_features (Band Gap)
        band_gap_val = row['band_gap']
        gap_features_tensor = torch.tensor([0.0 if pd.isna(band_gap_val) else band_gap_val], dtype=torch.float)
        gap_features_tensor = self._check_and_handle_nan_inf(gap_features_tensor, "band_gap", jid)
        # Ensure it has the correct dimension as defined in config
        if gap_features_tensor.numel() != config.BAND_GAP_SCALAR_DIM:
            warnings.warn(f"Band gap tensor for {jid} has dim {gap_features_tensor.numel()}, expected {config.BAND_GAP_SCALAR_DIM}. Adjusting.")
            gap_features_tensor = torch.zeros(config.BAND_GAP_SCALAR_DIM, dtype=torch.float)

        # DOS Features (placeholder if not directly loaded)
        # You would replace this with actual loading if you have DOS data files
        dos_features_tensor = torch.zeros(config.DOS_FEATURE_DIM, dtype=torch.float) # Placeholder
        dos_features_tensor = self._check_and_handle_nan_inf(dos_features_tensor, "dos_features", jid)

        # Fermi Surface Features (placeholder if not directly loaded)
        # You would replace this with actual loading if you have Fermi surface data
        fermi_features_tensor = torch.zeros(config.FERMI_FEATURE_DIM, dtype=torch.float) # Placeholder
        fermi_features_tensor = self._check_and_handle_nan_inf(fermi_features_tensor, "fermi_features", jid)

        # Consolidate kspace_physics_features dictionary for the model
        kspace_physics_features_dict = {
            'decomposition_features': full_decomposition_features_tensor,
            'gap_features': gap_features_tensor,
            'dos_features': dos_features_tensor,
            'fermi_features': fermi_features_tensor
        }


        # --- 4. Extract Scalar Features (Band Reps + Metadata, NOW EXCLUDING BAND GAP) ---
        band_rep_features_path = Path("/scratch/gpfs/as0714/graph_vector_topological_insulator/vectorized_features") / jid / 'band_rep_features.npy'
        try:
            band_rep_features = torch.tensor(np.load(band_rep_features_path), dtype=torch.float)
            band_rep_features = self._check_and_handle_nan_inf(band_rep_features, f"band_rep_features", jid)
        except Exception as e:
            warnings.warn(f"Could not load Band Rep features for JID {jid} from {band_rep_features_path}: {e}. Returning zeros.")
            band_rep_features = torch.zeros(getattr(config, 'BAND_REP_FEATURE_DIM', 4756), dtype=torch.float)


        scalar_metadata_features = [row[col] for col in self.scalar_features_columns]
        scalar_metadata_features = [0.0 if pd.isna(val) else val for val in scalar_metadata_features]
        scalar_metadata_features = torch.tensor(scalar_metadata_features, dtype=torch.float)
        scalar_metadata_features = self._check_and_handle_nan_inf(scalar_metadata_features, f"scalar_metadata_features", jid)


        combined_scalar_features = torch.cat([band_rep_features, scalar_metadata_features])
        combined_scalar_features = self._check_and_handle_nan_inf(combined_scalar_features, f"combined_scalar_features_before_scaler", jid)
        
        # Apply scaling to combined_scalar_features.
        if self.scaler:
            if combined_scalar_features.ndim == 1:
                combined_scalar_features_np = combined_scalar_features.unsqueeze(0).cpu().numpy()
                scaled_features_np = self.scaler.transform(combined_scalar_features_np)
                combined_scalar_features = torch.tensor(scaled_features_np.squeeze(0), dtype=torch.float)
            else: 
                scaled_features_np = self.scaler.transform(combined_scalar_features.cpu().numpy())
                combined_scalar_features = torch.tensor(scaled_features_np, dtype=torch.float)
            combined_scalar_features = self._check_and_handle_nan_inf(combined_scalar_features, f"combined_scalar_features_after_scaler", jid)

        # --- 5. Prepare Labels ---
        topology_label_str = row['topological_class']
        topology_label = torch.tensor(self.topology_class_map.get(topology_label_str, self.topology_class_map["Unknown"]), dtype=torch.long)
        
        magnetism_label_str = row['magnetic_type']
        magnetism_label = torch.tensor(self.magnetism_class_map.get(magnetism_label_str, self.magnetism_class_map["UNKNOWN"]), dtype=torch.long)

        # --- Set Feature Dimensions in Config for Model Initialization (if not already set) ---
        if config.CRYSTAL_NODE_FEATURE_DIM is None or config.CRYSTAL_NODE_FEATURE_DIM == 0:
            config.CRYSTAL_NODE_FEATURE_DIM = crystal_graph.x.shape[1]
        if config.KSPACE_GRAPH_NODE_FEATURE_DIM is None or config.KSPACE_GRAPH_NODE_FEATURE_DIM == 0:
            config.KSPACE_GRAPH_NODE_FEATURE_DIM = kspace_graph.x.shape[1]
        if config.ASPH_FEATURE_DIM is None or config.ASPH_FEATURE_DIM == 0:
            config.ASPH_FEATURE_DIM = asph_features.shape[0]
        if config.SCALAR_TOTAL_DIM is None or config.SCALAR_TOTAL_DIM == 0:
            config.SCALAR_TOTAL_DIM = combined_scalar_features.shape[0]

        return {
            'crystal_graph': crystal_graph,
            'kspace_graph': kspace_graph, # Now includes .pos and .symmetry_labels (if available)
            'asph_features': asph_features,
            'scalar_features': combined_scalar_features, # Excludes band_gap
            'kspace_physics_features': kspace_physics_features_dict, # Now includes decomp, gap, dos, fermi
            'topology_label': topology_label,
            'magnetism_label': magnetism_label,
            'jid': jid 
        }

    # --- Dummy Data Generation Methods ---
    def _generate_dummy_crystal_graph(self):
        num_nodes_dummy = 10
        crystal_node_feature_dim = getattr(config, 'CRYSTAL_NODE_FEATURE_DIM', 3) 
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
        pos_dummy = torch.randn(num_nodes_dummy, 3) # Add pos to dummy
        symmetry_labels_dummy = torch.randint(0, 10, (num_nodes_dummy,), dtype=torch.long) # Add dummy symmetry labels
        return PyGData(x=x_dummy, edge_index=edge_index_dummy, batch=batch_dummy, pos=pos_dummy, symmetry_labels=symmetry_labels_dummy)

    def _generate_dummy_asph_features(self):
        asph_feature_dim = getattr(config, 'ASPH_FEATURE_DIM', 3115) 
        return torch.zeros(asph_feature_dim)

    def _generate_dummy_base_decomposition_features(self):
        base_decomposition_feature_dim = getattr(config, 'BASE_DECOMPOSITION_FEATURE_DIM', 2) 
        return torch.zeros(base_decomposition_feature_dim)

# Custom collate function remains the same, as it stacks the dicts as before.
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for PyGDataLoader to handle a dictionary of inputs.
    It will batch PyGData objects separately and stack other tensors.
    """
    if not batch:
        return {}

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

    kspace_physics_features_collated = defaultdict(list)
    for d in batch:
        for key, tensor in d['kspace_physics_features'].items():
            kspace_physics_features_collated[key].append(tensor)
    
    for key in kspace_physics_features_collated:
        for i, tensor in enumerate(kspace_physics_features_collated[key]):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                warnings.warn(f"NaN/Inf detected in kspace_physics_features[{key}] for batch element {i} during collate. Replacing with zeros.")
                kspace_physics_features_collated[key][i] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        kspace_physics_features_collated[key] = torch.stack(kspace_physics_features_collated[key])
    collated_batch['kspace_physics_features'] = kspace_physics_features_collated

    return collated_batch