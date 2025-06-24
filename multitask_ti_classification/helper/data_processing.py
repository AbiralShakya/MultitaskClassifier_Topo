import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit

class ImprovedDataPreprocessor:
    """Improved data preprocessing for better training stability"""
    
    def __init__(self):
        self.feature_scalers = {}
        self.fitted = False
        
    def fit_transform(self, dataset):
        """Fit scalers and transform data"""
        print("Fitting data preprocessors...")
        
        # Collect all features for fitting scalers
        all_asph_features = []
        all_scalar_features = []
        all_decomp_features = []
        
        for data in dataset:
            if 'asph_features' in data:
                all_asph_features.append(data['asph_features'].numpy())
            if 'scalar_features' in data:
                all_scalar_features.append(data['scalar_features'].numpy())
            if 'kspace_physics_features' in data and 'decomposition_features' in data['kspace_physics_features']:
                all_decomp_features.append(data['kspace_physics_features']['decomposition_features'].numpy())
        
        if all_asph_features:
            all_asph = np.vstack(all_asph_features)
            self.feature_scalers['asph'] = RobustScaler()
            self.feature_scalers['asph'].fit(all_asph)
            
        if all_scalar_features:
            all_scalar = np.vstack(all_scalar_features)
            self.feature_scalers['scalar'] = RobustScaler()
            self.feature_scalers['scalar'].fit(all_scalar)
            
        if all_decomp_features:
            all_decomp = np.vstack(all_decomp_features)
            self.feature_scalers['decomp'] = RobustScaler()
            self.feature_scalers['decomp'].fit(all_decomp)
        
        self.fitted = True
        
        return self.transform(dataset)
    
    def transform(self, dataset):
        """Transform dataset using fitted scalers"""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted yet!")
        
        transformed_dataset = []
        
        for data in dataset:
            transformed_data = data.copy()
            
            # Transform ASPH features
            if 'asph_features' in data and 'asph' in self.feature_scalers:
                features = data['asph_features'].numpy()
                scaled_features = self.feature_scalers['asph'].transform(features.reshape(1, -1))
                transformed_data['asph_features'] = torch.FloatTensor(scaled_features.flatten())
            
            # Transform scalar features
            if 'scalar_features' in data and 'scalar' in self.feature_scalers:
                features = data['scalar_features'].numpy()
                scaled_features = self.feature_scalers['scalar'].transform(features.reshape(1, -1))
                transformed_data['scalar_features'] = torch.FloatTensor(scaled_features.flatten())
            
            # Transform decomposition features
            if ('kspace_physics_features' in data and 
                'decomposition_features' in data['kspace_physics_features'] and 
                'decomp' in self.feature_scalers):
                features = data['kspace_physics_features']['decomposition_features'].numpy()
                scaled_features = self.feature_scalers['decomp'].transform(features.reshape(1, -1))
                transformed_data['kspace_physics_features']['decomposition_features'] = torch.FloatTensor(scaled_features.flatten())
            
            # Normalize crystal node features (if they're not already normalized)
            if 'crystal_graph' in data and hasattr(data['crystal_graph'], 'x'):
                x = data['crystal_graph'].x
                if x.std() > 10:  # If features seem unnormalized
                    x_normalized = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
                    transformed_data['crystal_graph'].x = x_normalized
            
            # Normalize k-space node features
            if 'kspace_graph' in data and hasattr(data['kspace_graph'], 'x'):
                x = data['kspace_graph'].x
                if x.std() > 10:  # If features seem unnormalized
                    x_normalized = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
                    transformed_data['kspace_graph'].x = x_normalized
            
            transformed_dataset.append(transformed_data)
        
        return transformed_dataset

class StratifiedDataSplitter:
    """Create stratified splits for multi-task learning"""
    
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split(self, dataset):
        """Create stratified train/val/test splits"""
        
        # Extract labels for stratification
        topology_labels = []
        magnetism_labels = []
        
        for data in dataset:
            topology_labels.append(data['topology_targets'].item())
            magnetism_labels.append(data['magnetism_targets'].item())
        
        # Create combined labels for stratification
        combined_labels = [f"{t}_{m}" for t, m in zip(topology_labels, magnetism_labels)]
        
        # First split: train+val vs test
        splitter1 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        train_val_idx, test_idx = next(splitter1.split(range(len(dataset)), combined_labels))
        
        # Second split: train vs val
        train_val_labels = [combined_labels[i] for i in train_val_idx]
        val_size_adjusted = self.val_size / (1 - self.test_size)
        
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=val_size_adjusted, 
            random_state=self.random_state
        )
        
        train_idx_local, val_idx_local = next(splitter2.split(range(len(train_val_idx)), train_val_labels))
        
        # Convert back to global indices
        train_idx = train_val_idx[train_idx_local]
        val_idx = train_val_idx[val_idx_local]
        
        # Create split datasets
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]
        
        print(f"Dataset split: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset