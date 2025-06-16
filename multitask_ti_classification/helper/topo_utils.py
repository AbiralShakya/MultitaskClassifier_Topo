# utils.py

import torch
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from torch_geometric.data import Data as PyGData # Alias to avoid conflict with `Data` from dataclasses if imported

class SpaceGroupManager:
    """
    Manages shared space group k-space graphs and physics features.
    Adapted from your KSpacePhysicsGraphBuilder's saving logic.
    """
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self._cached_data = {} # Cache for (graph, physics_features) tuples

    def get_kspace_graph_path(self, space_group_number: int) -> Path:
        """Returns the expected path for a k-space graph .pt file."""
        return self.base_path / f'SG_{space_group_number:03d}' / 'kspace_graph.pt'

    def get_physics_features_path(self, space_group_number: int) -> Path:
        """Returns the expected path for physics features .pt file."""
        return self.base_path / f'SG_{space_group_number:03d}' / 'physics_features.pt'

    def load_kspace_data(self, space_group_number: int) -> Optional[Tuple[PyGData, Dict[str, torch.Tensor]]]:
        """
        Loads both the PyG k-space graph and its associated physics features.
        Caches the loaded data.
        """
        if space_group_number in self._cached_data:
            return self._cached_data[space_group_number]

        graph_path = self.get_kspace_graph_path(space_group_number)
        features_path = self.get_physics_features_path(space_group_number)

        graph = None
        features = None

        if graph_path.exists():
            try:
                graph = torch.load(graph_path, map_location="cpu")
            except Exception as e:
                print(f"Warning: Could not load k-space graph for SG {space_group_number} from {graph_path}: {e}")

        if features_path.exists():
            try:
                features = torch.load(features_path, map_location="cpu")
            except Exception as e:
                print(f"Warning: Could not load physics features for SG {space_group_number} from {features_path}: {e}")
        
        # Only cache and return if both are successfully loaded
        if graph is not None and features is not None:
            self._cached_data[space_group_number] = (graph, features)
            return graph, features
        else:
            return None # Indicate failure to load both


def load_material_graph_from_dict(graph_dict: Dict[str, Any]) -> PyGData:
    """
    Reconstructs a PyTorch Geometric Data object from a dictionary.
    This is necessary because MaterialRecord serializes PyGData to dict.
    """
    # Ensure tensors are on CPU first if loaded from numpy arrays, then can move to device later
    x = torch.tensor(graph_dict['x'], dtype=torch.float) if isinstance(graph_dict['x'], np.ndarray) else graph_dict['x']
    pos = torch.tensor(graph_dict['pos'], dtype=torch.float) if isinstance(graph_dict['pos'], np.ndarray) else graph_dict['pos']
    edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long) if isinstance(graph_dict['edge_index'], np.ndarray) else graph_dict['edge_index']
    edge_attr = torch.tensor(graph_dict['edge_attr'], dtype=torch.float) if isinstance(graph_dict['edge_attr'], np.ndarray) else graph_dict['edge_attr']

    # For 'y' label, ensure it's a tensor and has appropriate dtype (e.g., long for classification)
    y_label = torch.tensor(graph_dict['y'], dtype=torch.long) if isinstance(graph_dict['y'], np.ndarray) else graph_dict['y']

    # Handle optional 'num_nodes' if present in the dict
    num_nodes = graph_dict.get('num_nodes', None) # PyGData will infer if None

    return PyGData(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_label,
        num_nodes=num_nodes
    )

def load_pickle_data(path: Union[str, Path]) -> Any:
    """Helper to load data from pickle files."""
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_json_data(path: Union[str, Path]) -> Any:
    """Helper to load data from JSON files."""
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    return data