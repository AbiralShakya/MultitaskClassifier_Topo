import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
import warnings

# Import config for access to label mappings and helper function
import helper.config as config 

class ImprovedDataPreprocessor:
    """Improved data preprocessing for better training stability"""
    
    def __init__(self):
        self.feature_scalers = {}
        self.fitted = False
        
    def fit_transform(self, dataset):
        """
        Fit scalers on the provided dataset and then transform it.
        Also calculates and adds the 'combined_label' to each data item.
        """
        print("Fitting data preprocessors...")
        
        # Collect all features for fitting scalers
        all_asph_features = []
        all_scalar_features = []
        all_decomp_features = []
        
        # Data items from MaterialDataset are dicts with tensors (potentially on CPU)
        for data in dataset:
            if 'asph_features' in data and data['asph_features'] is not None and data['asph_features'].numel() > 0:
                all_asph_features.append(data['asph_features'].numpy())
            if 'scalar_features' in data and data['scalar_features'] is not None and data['scalar_features'].numel() > 0:
                all_scalar_features.append(data['scalar_features'].numpy())
            if 'kspace_physics_features' in data and 'decomposition_features' in data['kspace_physics_features'] and \
               data['kspace_physics_features']['decomposition_features'] is not None and \
               data['kspace_physics_features']['decomposition_features'].numel() > 0:
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
        """
        Transform dataset using fitted scalers and add 'combined_label'.
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted yet! Call fit_transform first.")
        
        transformed_dataset = []
        
        for data in dataset:
            transformed_data = data.copy() # Make a shallow copy to modify
            
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
            
            # # Normalize crystal node features (if they're not already normalized)
            # if 'crystal_graph' in data and hasattr(data['crystal_graph'], 'x') and data['crystal_graph'].x is not None:
            #     x = data['crystal_graph'].x
            #     # Check for standard deviation to avoid normalizing already scaled features or all-zero features
            #     if x.numel() > 0 and x.std() > 1e-6: # Only normalize if not all zeros and std is significant
            #         x_normalized = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
            #         transformed_data['crystal_graph'].x = x_normalized

            if x.numel() > 0: # Ensure tensor is not empty
                x_mean = x.mean(dim=0)
                x_std = x.std(dim=0, unbiased=False) # Use unbiased=False for n instead of n-1 for std, if it suits your needs
                x_std_safe = torch.where(x_std == 0, torch.tensor(1.0, device=x_std.device), x_std) # Replace 0 std with 1.0
                x_normalized = (x - x_mean) / (x_std_safe + 1e-8)
                transformed_data['crystal_graph'].x = x_normalized
                 
            # Normalize k-space node features
            if 'kspace_graph' in data and hasattr(data['kspace_graph'], 'x') and data['kspace_graph'].x is not None:
                x = data['kspace_graph'].x
                if x.numel() > 0 and x.std() > 1e-6:
                    x_normalized = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
                    transformed_data['kspace_graph'].x = x_normalized
            
            # --- NEW: Calculate and add combined_label ---
            if 'topology_label' in data and 'magnetism_label' in data:
                topo_int = data['topology_label'].item()
                mag_int = data['magnetism_label'].item()
                combined_label_val = config.get_combined_label_from_ints(topo_int, mag_int)
                transformed_data['combined_label'] = torch.tensor(combined_label_val, dtype=torch.long)
            else:
                warnings.warn(f"Missing 'topology_label' or 'magnetism_label' in data item {data.get('jid', 'unknown_jid')}. Cannot create 'combined_label'. Setting to default 0.")
                transformed_data['combined_label'] = torch.tensor(0, dtype=torch.long) # Default to 0

            transformed_dataset.append(transformed_data)
        
        return transformed_dataset

class StratifiedDataSplitter:
    """Create stratified splits for multi-task learning"""
    
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split(self, dataset):
        """
        Create stratified train/val/test splits.
        Expects 'combined_label' to be present in each item of the dataset.
        """
        # Ensure 'combined_label' is present for stratification
        if not all('combined_label' in d for d in dataset):
            raise ValueError("All data items must contain 'combined_label' for stratification. "
                             "Ensure ImprovedDataPreprocessor is configured to add it.")

        # Extract 'combined_label' for stratification
        combined_labels = [data['combined_label'].item() for data in dataset]
        
        # First split: train+val vs test
        splitter1 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        train_val_idx, test_idx = next(splitter1.split(range(len(dataset)), combined_labels))
        
        # Second split: train vs val
        train_val_labels = [combined_labels[i] for i in train_val_idx]
        val_size_adjusted = self.val_size / (1 - self.test_size) # Adjust val size relative to remaining data
        
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

