"""
Real Data Loaders for Topological Material Classification
========================================================

This module provides proper PyTorch Geometric data loaders for all modalities:
- Crystal graphs (atomic structure)
- K-space graphs (electronic structure) 
- Scalar features (bandgap, symmetry, etc.)
- Decomposition features (irreducible representations)
- Spectral features (Laplacian eigenvalues)

All loaders support proper batching, collation, and data augmentation.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.transforms import BaseTransform
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import random
from sklearn.preprocessing import StandardScaler
import warnings

# Optional imports for advanced features
# Optuna removed - hyperparameter optimization disabled
OPTUNA_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("Wandb not available. Logging will be disabled.")


class TopologicalMaterialDataset(Dataset):
    """
    Main dataset class for topological materials with all modalities.
    """
    
    def __init__(self, 
                 data_dir: Path,
                 split: str = 'train',
                 modalities: List[str] = None,
                 transform: Optional[BaseTransform] = None,
                 max_crystal_nodes: int = 1000,
                 max_kspace_nodes: int = 500):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing processed data
            split: 'train', 'val', or 'test'
            modalities: List of modalities to load ['crystal', 'kspace', 'scalar', 'decomposition', 'spectral']
            transform: Optional PyG transform to apply
            max_crystal_nodes: Maximum number of nodes in crystal graphs
            max_kspace_nodes: Maximum number of nodes in k-space graphs
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities or ['crystal', 'kspace', 'scalar', 'decomposition', 'spectral']
        self.transform = transform
        self.max_crystal_nodes = max_crystal_nodes
        self.max_kspace_nodes = max_kspace_nodes
        
        # Load data index
        self.data_index = self._load_data_index()
        
        # Load feature scalers
        self.scalers = self._load_scalers()
        
        # Load labels
        self.labels = self._load_labels()
        
        print(f"Loaded {len(self.data_index)} samples for {split} split")
        print(f"Modalities: {self.modalities}")
        
    def _load_data_index(self) -> List[Dict]:
        """Load the data index file."""
        index_path = self.data_dir / f"{self.split}_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Data index not found: {index_path}")
        
        with open(index_path, 'r') as f:
            return json.load(f)
    
    def _load_scalers(self) -> Dict[str, StandardScaler]:
        """Load pre-computed feature scalers."""
        scalers = {}
        scaler_path = self.data_dir / "scalers.pkl"
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
        
        return scalers
    
    def _load_labels(self) -> Dict[str, int]:
        """Load material labels."""
        labels_path = self.data_dir / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        with open(labels_path, 'r') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with all modalities."""
        sample_info = self.data_index[idx]
        material_id = sample_info['material_id']
        
        sample_data = {}
        
        # Load crystal graph
        if 'crystal' in self.modalities:
            crystal_data = self._load_crystal_graph(sample_info)
            if crystal_data is not None:
                sample_data.update({
                    'crystal_x': crystal_data.x,
                    'crystal_edge_index': crystal_data.edge_index,
                    'crystal_edge_attr': crystal_data.edge_attr if hasattr(crystal_data, 'edge_attr') else None
                })
        
        # Load k-space graph
        if 'kspace' in self.modalities:
            kspace_data = self._load_kspace_graph(sample_info)
            if kspace_data is not None:
                sample_data.update({
                    'kspace_x': kspace_data.x,
                    'kspace_edge_index': kspace_data.edge_index,
                    'kspace_edge_attr': kspace_data.edge_attr if hasattr(kspace_data, 'edge_attr') else None
                })
        
        # Load scalar features
        if 'scalar' in self.modalities:
            scalar_features = self._load_scalar_features(sample_info)
            if scalar_features is not None:
                sample_data['scalar_features'] = scalar_features
        
        # Load decomposition features
        if 'decomposition' in self.modalities:
            decomp_features = self._load_decomposition_features(sample_info)
            if decomp_features is not None:
                sample_data['decomposition_features'] = decomp_features
        
        # Load spectral features
        if 'spectral' in self.modalities:
            spectral_data = self._load_spectral_features(sample_info)
            if spectral_data is not None:
                sample_data.update({
                    'spectral_edge_index': spectral_data['edge_index'],
                    'spectral_num_nodes': spectral_data['num_nodes']
                })
        
        # Add label
        sample_data['label'] = torch.tensor(self.labels[material_id], dtype=torch.long)
        sample_data['material_id'] = material_id
        
        return sample_data
    
    def _load_crystal_graph(self, sample_info: Dict) -> Optional[Data]:
        """Load crystal graph data."""
        try:
            graph_path = self.data_dir / "crystal_graphs" / f"{sample_info['material_id']}.pt"
            if not graph_path.exists():
                return None
            
            data = torch.load(graph_path)
            
            # Apply transforms if specified
            if self.transform is not None:
                data = self.transform(data)
            
            # Limit number of nodes if needed
            if data.x.size(0) > self.max_crystal_nodes:
                # Simple random sampling of nodes
                indices = torch.randperm(data.x.size(0))[:self.max_crystal_nodes]
                data.x = data.x[indices]
                # Update edge_index to only include selected nodes
                mask = torch.isin(data.edge_index[0], indices) & torch.isin(data.edge_index[1], indices)
                data.edge_index = data.edge_index[:, mask]
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[mask]
            
            return data
        except Exception as e:
            print(f"Error loading crystal graph for {sample_info['material_id']}: {e}")
            return None
    
    def _load_kspace_graph(self, sample_info: Dict) -> Optional[Data]:
        """Load k-space graph data."""
        try:
            graph_path = self.data_dir / "kspace_graphs" / f"{sample_info['material_id']}.pt"
            if not graph_path.exists():
                return None
            
            data = torch.load(graph_path)
            
            # Apply transforms if specified
            if self.transform is not None:
                data = self.transform(data)
            
            # Limit number of nodes if needed
            if data.x.size(0) > self.max_kspace_nodes:
                indices = torch.randperm(data.x.size(0))[:self.max_kspace_nodes]
                data.x = data.x[indices]
                mask = torch.isin(data.edge_index[0], indices) & torch.isin(data.edge_index[1], indices)
                data.edge_index = data.edge_index[:, mask]
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[mask]
            
            return data
        except Exception as e:
            print(f"Error loading k-space graph for {sample_info['material_id']}: {e}")
            return None
    
    def _load_scalar_features(self, sample_info: Dict) -> Optional[torch.Tensor]:
        """Load scalar features."""
        try:
            features_path = self.data_dir / "scalar_features" / f"{sample_info['material_id']}.npy"
            if not features_path.exists():
                return None
            
            features = np.load(features_path)
            
            # Apply scaling if scaler is available
            if 'scalar' in self.scalers:
                features = self.scalers['scalar'].transform(features.reshape(1, -1)).flatten()
            
            return torch.tensor(features, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading scalar features for {sample_info['material_id']}: {e}")
            return None
    
    def _load_decomposition_features(self, sample_info: Dict) -> Optional[torch.Tensor]:
        """Load decomposition features."""
        try:
            features_path = self.data_dir / "decomposition_features" / f"{sample_info['material_id']}.npy"
            if not features_path.exists():
                return None
            
            features = np.load(features_path)
            
            # Apply scaling if scaler is available
            if 'decomposition' in self.scalers:
                features = self.scalers['decomposition'].transform(features.reshape(1, -1)).flatten()
            
            return torch.tensor(features, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading decomposition features for {sample_info['material_id']}: {e}")
            return None
    
    def _load_spectral_features(self, sample_info: Dict) -> Optional[Dict]:
        """Load spectral features."""
        try:
            spectral_path = self.data_dir / "spectral_features" / f"{sample_info['material_id']}.pt"
            if not spectral_path.exists():
                return None
            
            spectral_data = torch.load(spectral_path)
            return spectral_data
        except Exception as e:
            print(f"Error loading spectral features for {sample_info['material_id']}: {e}")
            return None


class MultiModalCollate:
    """
    Custom collate function for multi-modal data with proper batching.
    """
    
    def __init__(self, modalities: List[str]):
        self.modalities = modalities
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of multi-modal samples."""
        collated = {}
        
        # Handle labels
        if 'label' in batch[0]:
            collated['labels'] = torch.stack([item['label'] for item in batch])
        
        # Handle material IDs
        if 'material_id' in batch[0]:
            collated['material_ids'] = [item['material_id'] for item in batch]
        
        # Handle crystal graphs
        if 'crystal_x' in batch[0]:
            crystal_batch = []
            for item in batch:
                if 'crystal_x' in item:
                    data = Data(
                        x=item['crystal_x'],
                        edge_index=item['crystal_edge_index'],
                        edge_attr=item.get('crystal_edge_attr')
                    )
                    crystal_batch.append(data)
            
            if crystal_batch:
                collated['crystal_batch'] = Batch.from_data_list(crystal_batch)
        
        # Handle k-space graphs
        if 'kspace_x' in batch[0]:
            kspace_batch = []
            for item in batch:
                if 'kspace_x' in item:
                    data = Data(
                        x=item['kspace_x'],
                        edge_index=item['kspace_edge_index'],
                        edge_attr=item.get('kspace_edge_attr')
                    )
                    kspace_batch.append(data)
            
            if kspace_batch:
                collated['kspace_batch'] = Batch.from_data_list(kspace_batch)
        
        # Handle scalar features
        if 'scalar_features' in batch[0]:
            scalar_features = [item['scalar_features'] for item in batch if 'scalar_features' in item]
            if scalar_features:
                collated['scalar_features'] = torch.stack(scalar_features)
        
        # Handle decomposition features
        if 'decomposition_features' in batch[0]:
            decomp_features = [item['decomposition_features'] for item in batch if 'decomposition_features' in item]
            if decomp_features:
                collated['decomposition_features'] = torch.stack(decomp_features)
        
        # Handle spectral features
        if 'spectral_edge_index' in batch[0]:
            spectral_batch = []
            for item in batch:
                if 'spectral_edge_index' in item:
                    spectral_batch.append({
                        'edge_index': item['spectral_edge_index'],
                        'num_nodes': item['spectral_num_nodes']
                    })
            
            if spectral_batch:
                collated['spectral_batch'] = spectral_batch
        
        return collated


class DataAugmentation:
    """
    Data augmentation for multi-modal topological material data.
    """
    
    def __init__(self, 
                 mixup_alpha: float = 0.2,
                 cutmix_prob: float = 0.5,
                 feature_mask_prob: float = 0.1,
                 edge_dropout: float = 0.1,
                 node_feature_noise: float = 0.05):
        """
        Initialize data augmentation.
        
        Args:
            mixup_alpha: Alpha parameter for Mixup
            cutmix_prob: Probability of applying CutMix
            feature_mask_prob: Probability of masking features
            edge_dropout: Probability of dropping edges
            node_feature_noise: Standard deviation of noise to add to node features
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
        self.feature_mask_prob = feature_mask_prob
        self.edge_dropout = edge_dropout
        self.node_feature_noise = node_feature_noise
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentations to a batch."""
        augmented_batch = batch.copy()
        
        # Apply Mixup
        if self.mixup_alpha > 0 and random.random() < 0.5:
            augmented_batch = self._apply_mixup(augmented_batch)
        
        # Apply CutMix
        if self.cutmix_prob > 0 and random.random() < self.cutmix_prob:
            augmented_batch = self._apply_cutmix(augmented_batch)
        
        # Apply feature masking
        if self.feature_mask_prob > 0:
            augmented_batch = self._apply_feature_masking(augmented_batch)
        
        # Apply edge dropout
        if self.edge_dropout > 0:
            augmented_batch = self._apply_edge_dropout(augmented_batch)
        
        # Apply node feature noise
        if self.node_feature_noise > 0:
            augmented_batch = self._apply_node_feature_noise(augmented_batch)
        
        return augmented_batch
    
    def _apply_mixup(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply Mixup augmentation."""
        if 'labels' not in batch:
            return batch
        
        batch_size = batch['labels'].size(0)
        if batch_size < 2:
            return batch
        
        # Generate mixing weights
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Randomly shuffle indices
        indices = torch.randperm(batch_size)
        
        # Mix features
        for key in ['scalar_features', 'decomposition_features']:
            if key in batch:
                batch[key] = lam * batch[key] + (1 - lam) * batch[key][indices]
        
        # Mix labels (soft labels)
        if 'labels' in batch:
            batch['labels'] = lam * F.one_hot(batch['labels'], num_classes=3).float() + \
                            (1 - lam) * F.one_hot(batch['labels'][indices], num_classes=3).float()
        
        return batch
    
    def _apply_cutmix(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply CutMix augmentation."""
        if 'labels' not in batch:
            return batch
        
        batch_size = batch['labels'].size(0)
        if batch_size < 2:
            return batch
        
        # Generate cutmix parameters
        lam = np.random.beta(1.0, 1.0)
        indices = torch.randperm(batch_size)
        
        # Apply CutMix to scalar features
        if 'scalar_features' in batch:
            features = batch['scalar_features']
            cut_len = int(features.size(1) * (1 - lam))
            cut_start = random.randint(0, features.size(1) - cut_len)
            
            mixed_features = features.clone()
            mixed_features[:, cut_start:cut_start + cut_len] = features[indices, cut_start:cut_start + cut_len]
            batch['scalar_features'] = mixed_features
        
        # Mix labels
        if 'labels' in batch:
            batch['labels'] = lam * F.one_hot(batch['labels'], num_classes=3).float() + \
                            (1 - lam) * F.one_hot(batch['labels'][indices], num_classes=3).float()
        
        return batch
    
    def _apply_feature_masking(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply feature masking."""
        for key in ['scalar_features', 'decomposition_features']:
            if key in batch:
                mask = torch.rand(batch[key].size()) < self.feature_mask_prob
                batch[key][mask] = 0.0
        
        return batch
    
    def _apply_edge_dropout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply edge dropout to graphs."""
        for graph_key in ['crystal_batch', 'kspace_batch']:
            if graph_key in batch:
                graph = batch[graph_key]
                edge_mask = torch.rand(graph.edge_index.size(1)) > self.edge_dropout
                graph.edge_index = graph.edge_index[:, edge_mask]
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    graph.edge_attr = graph.edge_attr[edge_mask]
        
        return batch
    
    def _apply_node_feature_noise(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add noise to node features."""
        for graph_key in ['crystal_batch', 'kspace_batch']:
            if graph_key in batch:
                graph = batch[graph_key]
                noise = torch.randn_like(graph.x) * self.node_feature_noise
                graph.x = graph.x + noise
        
        return batch


def create_data_loaders(data_dir: Path,
                       batch_size: int = 32,
                       modalities: List[str] = None,
                       num_workers: int = 4,
                       augment: bool = True,
                       **augmentation_kwargs) -> Dict[str, DataLoader]:
    """
    Create data loaders for all splits.
    
    Args:
        data_dir: Directory containing processed data
        batch_size: Batch size for training
        modalities: List of modalities to load
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation
        **augmentation_kwargs: Arguments for DataAugmentation
    
    Returns:
        Dictionary containing train, validation, and test data loaders
    """
    modalities = modalities or ['crystal', 'kspace', 'scalar', 'decomposition', 'spectral']
    
    # Create datasets
    train_dataset = TopologicalMaterialDataset(
        data_dir=data_dir,
        split='train',
        modalities=modalities
    )
    
    val_dataset = TopologicalMaterialDataset(
        data_dir=data_dir,
        split='val',
        modalities=modalities
    )
    
    test_dataset = TopologicalMaterialDataset(
        data_dir=data_dir,
        split='test',
        modalities=modalities
    )
    
    # Create collate function
    collate_fn = MultiModalCollate(modalities)
    
    # Create augmentation
    augmentation = DataAugmentation(**augmentation_kwargs) if augment else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Apply augmentation to training data
    if augmentation is not None:
        train_loader.dataset.augmentation = augmentation
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def get_class_weights(data_dir: Path) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        data_dir: Directory containing processed data
    
    Returns:
        Class weights tensor
    """
    labels_path = data_dir / "labels.json"
    if not labels_path.exists():
        return torch.ones(3)  # Default equal weights
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    # Count class frequencies
    class_counts = [0, 0, 0]  # trivial, semimetal, insulator
    for label in labels.values():
        class_counts[label] += 1
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    
    return torch.tensor(class_weights, dtype=torch.float32)


if __name__ == "__main__":
    # Test the data loaders
    data_dir = Path("data/processed")
    
    if data_dir.exists():
        loaders = create_data_loaders(
            data_dir=data_dir,
            batch_size=4,
            modalities=['crystal', 'scalar'],
            augment=True
        )
        
        print("Data loaders created successfully!")
        print(f"Train batches: {len(loaders['train'])}")
        print(f"Val batches: {len(loaders['val'])}")
        print(f"Test batches: {len(loaders['test'])}")
        
        # Test a batch
        for batch in loaders['train']:
            print("Sample batch keys:", batch.keys())
            if 'labels' in batch:
                print("Labels shape:", batch['labels'].shape)
            break
    else:
        print(f"Data directory {data_dir} not found. Please run data preprocessing first.") 