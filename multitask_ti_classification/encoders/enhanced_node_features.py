import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

class EnhancedAtomicFeatures:
    """
    Enhanced atomic feature encoding based on the Nature paper approach.
    Creates rich node features for better topological classification.
    """
    
    def __init__(self):
        # Atomic properties for elements 1-118
        self.atomic_properties = self._initialize_atomic_properties()
        
    def _initialize_atomic_properties(self) -> Dict:
        """Initialize comprehensive atomic properties for elements 1-118"""
        return {
            # Electronegativity (Pauling scale)
            'electronegativity': {
                1: 2.20, 2: 4.16, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
                9: 3.98, 10: 4.79, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19,
                16: 2.58, 17: 3.16, 18: 3.24, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54,
                23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90,
                30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 3.00,
                37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.6, 42: 2.16, 43: 1.9,
                44: 2.2, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
                51: 2.05, 52: 2.1, 53: 2.66, 54: 2.60, 55: 0.79, 56: 0.89, 57: 1.10,
                58: 1.12, 59: 1.13, 60: 1.14, 61: 1.13, 62: 1.17, 63: 1.2, 64: 1.20,
                65: 1.1, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.1, 71: 1.27,
                72: 1.3, 73: 1.5, 74: 2.36, 75: 1.9, 76: 2.2, 77: 2.20, 78: 2.28,
                79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.0, 85: 2.2,
                86: 2.2, 87: 0.7, 88: 0.9, 89: 1.1, 90: 1.3, 91: 1.5, 92: 1.38,
                93: 1.36, 94: 1.28, 95: 1.13, 96: 1.28, 97: 1.3, 98: 1.3, 99: 1.3,
                100: 1.3, 101: 1.3, 102: 1.3, 103: 1.3, 104: 1.3, 105: 1.3, 106: 1.3,
                107: 1.3, 108: 1.3, 109: 1.3, 110: 1.3, 111: 1.3, 112: 1.3, 113: 1.3,
                114: 1.3, 115: 1.3, 116: 1.3, 117: 1.3, 118: 1.3
            },
            
            # Covalent radii (Å) - Cordero et al. 2008
            'covalent_radius': {
                1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66,
                9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
                16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60,
                23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32,
                30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16,
                37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64, 42: 1.54, 43: 1.47,
                44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39,
                51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44, 56: 2.15, 57: 2.07,
                58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96,
                65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.90, 70: 1.87, 71: 1.87,
                72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36,
                79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48, 84: 1.40, 85: 1.50,
                86: 1.50, 87: 2.60, 88: 2.21, 89: 2.15, 90: 2.06, 91: 2.00, 92: 1.96,
                93: 1.90, 94: 1.87, 95: 1.80, 96: 1.69, 97: 1.60, 98: 1.60, 99: 1.60,
                100: 1.60, 101: 1.60, 102: 1.60, 103: 1.60, 104: 1.60, 105: 1.60, 106: 1.60,
                107: 1.60, 108: 1.60, 109: 1.60, 110: 1.60, 111: 1.60, 112: 1.60, 113: 1.60,
                114: 1.60, 115: 1.60, 116: 1.60, 117: 1.60, 118: 1.60
            },
            
            # First ionization energy (eV)
            'ionization_energy': {
                1: 13.60, 2: 24.59, 3: 5.39, 4: 9.32, 5: 8.30, 6: 11.26, 7: 14.53, 8: 13.62,
                9: 17.42, 10: 21.56, 11: 5.14, 12: 7.65, 13: 5.99, 14: 8.15, 15: 10.49,
                16: 10.36, 17: 12.97, 18: 15.76, 19: 4.34, 20: 6.11, 21: 6.56, 22: 6.83,
                23: 6.75, 24: 6.77, 25: 7.43, 26: 7.90, 27: 7.88, 28: 7.64, 29: 7.73,
                30: 9.39, 31: 5.99, 32: 7.90, 33: 9.79, 34: 9.75, 35: 11.81, 36: 14.00,
                37: 4.18, 38: 5.69, 39: 6.22, 40: 6.63, 41: 6.76, 42: 7.09, 43: 7.28,
                44: 7.36, 45: 7.46, 46: 8.34, 47: 7.58, 48: 8.99, 49: 5.79, 50: 7.34,
                51: 8.61, 52: 9.01, 53: 10.45, 54: 12.13, 55: 3.89, 56: 5.21, 57: 5.58,
                58: 5.54, 59: 5.47, 60: 5.53, 61: 5.58, 62: 5.64, 63: 5.67, 64: 6.15,
                65: 5.86, 66: 5.94, 67: 6.02, 68: 6.11, 69: 6.18, 70: 6.25, 71: 5.43,
                72: 6.83, 73: 7.55, 74: 7.86, 75: 7.83, 76: 8.44, 77: 8.97, 78: 8.96,
                79: 9.23, 80: 10.44, 81: 6.11, 82: 7.42, 83: 7.29, 84: 8.41, 85: 9.32,
                86: 10.75, 87: 4.07, 88: 5.28, 89: 5.17, 90: 6.31, 91: 5.89, 92: 6.19,
                93: 6.27, 94: 6.03, 95: 5.97, 96: 5.99, 97: 6.20, 98: 6.28, 99: 6.42,
                100: 6.50, 101: 6.58, 102: 6.65, 103: 4.90, 104: 6.00, 105: 6.00, 106: 6.00,
                107: 6.00, 108: 6.00, 109: 6.00, 110: 6.00, 111: 6.00, 112: 6.00, 113: 6.00,
                114: 6.00, 115: 6.00, 116: 6.00, 117: 6.00, 118: 6.00
            },
            
            # Atomic volume (Å³)
            'atomic_volume': {
                1: 14.1, 2: 32.0, 3: 13.1, 4: 5.0, 5: 4.6, 6: 5.3, 7: 17.3, 8: 14.0,
                9: 17.1, 10: 16.7, 11: 23.7, 12: 14.0, 13: 10.0, 14: 12.1, 15: 17.0,
                16: 15.5, 17: 18.7, 18: 24.2, 19: 45.3, 20: 26.0, 21: 15.0, 22: 10.6,
                23: 8.35, 24: 7.23, 25: 7.35, 26: 7.1, 27: 6.7, 28: 6.6, 29: 7.1,
                30: 9.2, 31: 11.8, 32: 13.6, 33: 13.1, 34: 16.5, 35: 23.5, 36: 27.9,
                37: 55.9, 38: 33.7, 39: 19.9, 40: 14.1, 41: 10.8, 42: 9.4, 43: 8.5,
                44: 8.3, 45: 8.3, 46: 8.9, 47: 10.3, 48: 13.1, 49: 15.7, 50: 16.3,
                51: 18.4, 52: 20.5, 53: 25.7, 54: 35.9, 55: 70.0, 56: 38.2, 57: 22.5,
                58: 20.7, 59: 20.8, 60: 20.6, 61: 19.9, 62: 19.9, 63: 28.9, 64: 19.9,
                65: 19.3, 66: 19.0, 67: 18.7, 68: 18.4, 69: 18.1, 70: 24.8, 71: 17.8,
                72: 13.6, 73: 10.9, 74: 9.5, 75: 8.9, 76: 8.4, 77: 8.5, 78: 9.1,
                79: 10.2, 80: 14.8, 81: 17.2, 82: 18.3, 83: 21.3, 84: 22.7, 85: 32.0,
                86: 50.0, 87: 71.0, 88: 45.0, 89: 22.5, 90: 19.9, 91: 15.0, 92: 12.5,
                93: 11.6, 94: 12.3, 95: 17.1, 96: 18.1, 97: 16.8, 98: 16.5, 99: 16.2,
                100: 16.0, 101: 15.8, 102: 15.6, 103: 15.4, 104: 15.2, 105: 15.0, 106: 14.8,
                107: 14.6, 108: 14.4, 109: 14.2, 110: 14.0, 111: 13.8, 112: 13.6, 113: 13.4,
                114: 13.2, 115: 13.0, 116: 12.8, 117: 12.6, 118: 12.4
            }
        }
    
    def get_group_period(self, atomic_number: int) -> tuple:
        """Get group and period for an element"""
        # Period mapping
        if atomic_number <= 2:
            period = 1
        elif atomic_number <= 10:
            period = 2
        elif atomic_number <= 18:
            period = 3
        elif atomic_number <= 36:
            period = 4
        elif atomic_number <= 54:
            period = 5
        elif atomic_number <= 86:
            period = 6
        else:
            period = 7
        
        # Group mapping (simplified)
        group_map = {
            1: 1, 2: 18,  # H, He
            3: 1, 4: 2, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18,  # Li-Ne
            11: 1, 12: 2, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,  # Na-Ar
            19: 1, 20: 2, 21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10,  # K-Ni
            29: 11, 30: 12, 31: 13, 32: 14, 33: 15, 34: 16, 35: 17, 36: 18,  # Cu-Kr
            37: 1, 38: 2, 39: 3, 40: 4, 41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10,  # Rb-Pd
            47: 11, 48: 12, 49: 13, 50: 14, 51: 15, 52: 16, 53: 17, 54: 18,  # Ag-Xe
            55: 1, 56: 2,  # Cs, Ba
            # Lanthanides (57-71) - group 3
            57: 3, 58: 3, 59: 3, 60: 3, 61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3,
            67: 3, 68: 3, 69: 3, 70: 3, 71: 3,
            # Continue with period 6
            72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10, 79: 11, 80: 12,
            81: 13, 82: 14, 83: 15, 84: 16, 85: 17, 86: 18,  # Hf-Rn
            87: 1, 88: 2,  # Fr, Ra
            # Actinides (89-103) - group 3
            89: 3, 90: 3, 91: 3, 92: 3, 93: 3, 94: 3, 95: 3, 96: 3, 97: 3, 98: 3,
            99: 3, 100: 3, 101: 3, 102: 3, 103: 3,
            # Super heavy elements
            104: 4, 105: 5, 106: 6, 107: 7, 108: 8, 109: 9, 110: 10, 111: 11, 112: 12,
            113: 13, 114: 14, 115: 15, 116: 16, 117: 17, 118: 18
        }
        
        group = group_map.get(atomic_number, 1)
        
        return group, period
    
    def encode_atomic_features(self, atomic_numbers: List[int]) -> torch.Tensor:
        """
        Encode atomic features following the paper's approach:
        - One-hot group (18 dimensions)
        - One-hot period (7 dimensions) 
        - Binned electronegativity (10 dimensions)
        - Binned covalent radius (10 dimensions)
        - Binned ionization energy (10 dimensions)
        - Binned atomic volume (10 dimensions)
        Total: 65 dimensions
        """
        features = []
        
        for atomic_num in atomic_numbers:
            feature_vec = []
            
            # Group and period one-hot encoding
            group, period = self.get_group_period(atomic_num)
            group_onehot = torch.zeros(18)
            period_onehot = torch.zeros(7)
            group_onehot[group - 1] = 1.0
            period_onehot[period - 1] = 1.0
            
            feature_vec.extend([group_onehot, period_onehot])
            
            # Binned continuous properties
            for prop_name, prop_dict in self.atomic_properties.items():
                value = prop_dict.get(atomic_num, 0.0)
                binned = self._bin_property(value, prop_name)
                feature_vec.append(binned)
            
            features.append(torch.cat(feature_vec))
        
        return torch.stack(features)
    
    def _bin_property(self, value: float, property_name: str) -> torch.Tensor:
        """Bin continuous property into 10 segments"""
        # Define ranges for each property
        ranges = {
            'electronegativity': (0.5, 4.5),
            'covalent_radius': (0.3, 2.5),
            'ionization_energy': (3.0, 25.0),
            'atomic_volume': (5.0, 50.0)
        }
        
        min_val, max_val = ranges.get(property_name, (0.0, 1.0))
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0,1]
        
        bin_idx = min(9, int(normalized * 10))
        onehot = torch.zeros(10)
        onehot[bin_idx] = 1.0
        
        return onehot


class EnhancedCGCNNEncoder(nn.Module):
    """
    Enhanced CGCNN encoder with rich atomic features and improved architecture
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        
        self.atomic_features = EnhancedAtomicFeatures()
        
        # Input projection for rich atomic features
        self.node_embedding = nn.Sequential(
            nn.Linear(65, hidden_dim),  # 65-dim rich features -> hidden_dim
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Graph convolution layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(0.1))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x, edge_index, edge_attr, batch, atomic_numbers=None):
        # If atomic numbers provided, use enhanced features
        if atomic_numbers is not None:
            x = self.atomic_features.encode_atomic_features(atomic_numbers)
        
        # Embed node features
        x = self.node_embedding(x)
        
        # Embed edge features
        edge_features = self.edge_embedding(edge_attr)
        
        # Graph convolution with residual connections
        for conv, bn, dropout in zip(self.conv_layers, self.batch_norms, self.dropouts):
            # Aggregate edge information
            row, col = edge_index
            edge_msg = torch.cat([x[row], edge_features], dim=1)
            
            # Apply convolution
            x_new = conv(edge_msg)
            
            # Aggregate messages (simple mean aggregation)
            x_agg = torch.zeros_like(x)
            x_agg.index_add_(0, col, x_new)
            
            # Residual connection + normalization
            x = x + x_agg
            x = dropout(bn(x))
        
        # Global pooling
        from torch_geometric.nn import global_mean_pool, global_max_pool
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # Combine mean and max pooling
        x = torch.cat([x_mean, x_max], dim=1)
        
        return self.output_proj(x)