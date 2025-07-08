"""
Topological ML Encoder inspired by arXiv:1805.10503v2
Uses deep learning to predict topological invariants from Hamiltonians in momentum space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import warnings

class TopologicalMLEncoder(nn.Module):
    """
    Deep learning encoder for topological invariants from Hamiltonians.
    Inspired by arXiv:1805.10503v2 - "Deep Learning Topological Invariants of Band Insulators"
    
    This encoder takes Hamiltonians in momentum space and predicts:
    1. Topological invariants (winding number, Chern number)
    2. Local Berry curvature/winding angle (for interpretability)
    3. Topological features for downstream classification
    """
    
    def __init__(
        self,
        input_dim: int = 8,  # Real/imaginary parts of 2x2 matrix D(k)
        k_points: int = 32,  # Number of k-points in Brillouin zone
        hidden_dims: list = [64, 128, 256, 512],
        num_classes: int = 3,  # Number of topological classes
        output_features: int = 128,  # Features for downstream tasks
        use_auxiliary_loss: bool = True,
        extract_local_features: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.k_points = k_points
        self.num_classes = num_classes
        self.output_features = output_features
        self.use_auxiliary_loss = use_auxiliary_loss
        self.extract_local_features = extract_local_features
        
        # Input shape: (batch_size, input_dim, k_points)
        # For 2x2 matrix D(k): 8 channels (Re/Im of D11, D12, D21, D22)
        
        # Convolutional layers for 1D k-space (like the paper)
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim
        
        for hidden_dim in hidden_dims:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )
            in_channels = hidden_dim
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers for final prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1] // 2, hidden_dims[-1] // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output heads
        self.topological_classifier = nn.Linear(hidden_dims[-1] // 4, num_classes)
        self.feature_projection = nn.Linear(hidden_dims[-1] // 4, output_features)
        
        # For extracting local features (like winding angle/Berry curvature)
        if self.extract_local_features:
            self.local_feature_extractor = nn.ModuleList([
                nn.Conv1d(hidden_dims[-1], 20, kernel_size=1),  # H1 layer from paper
                nn.Conv1d(20, 10, kernel_size=1)  # H2 layer from paper
            ])
    
    def forward(self, hamiltonian_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the topological ML encoder.
        
        Args:
            hamiltonian_data: Tensor of shape (batch_size, input_dim, k_points)
                             For 2x2 matrix D(k): (batch_size, 8, k_points)
        
        Returns:
            Dictionary containing:
            - topological_logits: Classification logits
            - topological_features: Features for downstream tasks
            - local_features: Local winding angle/Berry curvature (if enabled)
            - auxiliary_features: Intermediate features for analysis
        """
        batch_size = hamiltonian_data.shape[0]
        
        # Apply convolutional layers
        x = hamiltonian_data
        conv_features = []
        
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            conv_features.append(x)
        
        # Extract local features (like H1, H2 layers in the paper)
        local_features = None
        if self.extract_local_features:
            h1_features = self.local_feature_extractor[0](x)  # (batch, 20, k_points)
            h2_features = self.local_feature_extractor[1](h1_features)  # (batch, 10, k_points)
            local_features = {
                'h1_features': h1_features,  # Like winding angle α(k)
                'h2_features': h2_features   # Like Δα(k)
            }
        
        # Global pooling
        pooled = self.global_pool(x).squeeze(-1)  # (batch, hidden_dims[-1])
        
        # Fully connected layers
        fc_features = self.fc_layers(pooled)
        
        # Output predictions
        topological_logits = self.topological_classifier(fc_features)
        topological_features = self.feature_projection(fc_features)
        
        return {
            'topological_logits': topological_logits,
            'topological_features': topological_features,
            'local_features': local_features,
            'auxiliary_features': {
                'conv_features': conv_features,
                'fc_features': fc_features
            }
        }

class TopologicalMLEncoder2D(nn.Module):
    """
    2D version for 2D materials (like Chern number prediction).
    Takes 2D Hamiltonians in k-space and predicts Chern numbers.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # hx, hy, hz components of H(k) = h(k)·σ
        k_grid: int = 8,     # kx × ky grid size
        hidden_dims: list = [32, 64, 128, 256],
        num_classes: int = 5,  # Chern numbers: -2, -1, 0, 1, 2
        output_features: int = 128,
        extract_berry_curvature: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.k_grid = k_grid
        self.num_classes = num_classes
        self.output_features = output_features
        self.extract_berry_curvature = extract_berry_curvature
        
        # Input shape: (batch_size, input_dim, k_grid, k_grid)
        # For 2D H(k) = h(k)·σ: (batch_size, 3, k_grid, k_grid)
        
        # 2D Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim
        
        for hidden_dim in hidden_dims:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )
            in_channels = hidden_dim
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1] // 2, hidden_dims[-1] // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output heads
        self.chern_classifier = nn.Linear(hidden_dims[-1] // 4, num_classes)
        self.feature_projection = nn.Linear(hidden_dims[-1] // 4, output_features)
        
        # Berry curvature extractor (like H3 layer in the paper)
        if self.extract_berry_curvature:
            self.berry_curvature_extractor = nn.Conv2d(
                hidden_dims[-1], 3, kernel_size=1
            )  # 3 channels for positive/negative/zero Berry curvature
    
    def forward(self, hamiltonian_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for 2D topological ML encoder.
        
        Args:
            hamiltonian_data: Tensor of shape (batch_size, input_dim, k_grid, k_grid)
                             For 2D H(k) = h(k)·σ: (batch_size, 3, k_grid, k_grid)
        
        Returns:
            Dictionary containing:
            - chern_logits: Chern number classification logits
            - topological_features: Features for downstream tasks
            - berry_curvature: Local Berry curvature (if enabled)
        """
        batch_size = hamiltonian_data.shape[0]
        
        # Apply 2D convolutional layers
        x = hamiltonian_data
        conv_features = []
        
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            conv_features.append(x)
        
        # Extract Berry curvature (like H3 layer in the paper)
        berry_curvature = None
        if self.extract_berry_curvature:
            berry_curvature = self.berry_curvature_extractor(x)  # (batch, 3, k_grid, k_grid)
        
        # Global pooling
        pooled = self.global_pool(x).squeeze(-1).squeeze(-1)  # (batch, hidden_dims[-1])
        
        # Fully connected layers
        fc_features = self.fc_layers(pooled)
        
        # Output predictions
        chern_logits = self.chern_classifier(fc_features)
        topological_features = self.feature_projection(fc_features)
        
        return {
            'chern_logits': chern_logits,
            'topological_features': topological_features,
            'berry_curvature': berry_curvature,
            'auxiliary_features': {
                'conv_features': conv_features,
                'fc_features': fc_features
            }
        }

def create_hamiltonian_from_features(
    features: torch.Tensor, 
    k_points: int = 32,
    model_type: str = "1d_a3"
) -> torch.Tensor:
    """
    Create synthetic Hamiltonians from features for training/testing.
    
    Args:
        features: Input features (could be from your existing encoders)
        k_points: Number of k-points
        model_type: Type of model ("1d_a3", "2d_a")
    
    Returns:
        Hamiltonian tensor ready for topological ML encoder
    """
    batch_size = features.shape[0]
    feature_dim = features.shape[1]
    
    if model_type == "1d_a3":
        # 1D AIII class: 2x2 matrix D(k)
        # Use features to parameterize the Hamiltonian
        hamiltonian = torch.zeros(batch_size, feature_dim, k_points, device=features.device)
        
        # Simple parameterization: use features to create smooth functions
        for i in range(batch_size):
            # Create smooth functions in k-space
            k = torch.linspace(-np.pi, np.pi, k_points, device=features.device)
            
            # Use all features to parameterize the Hamiltonian
            # Each feature becomes a channel in the Hamiltonian
            for j in range(feature_dim):
                f = features[i, j]
                # Create different patterns for different features
                if j < 4:
                    # Use first 4 features for basic parameterization
                    hamiltonian[i, j] = f * torch.cos(k) + f * torch.sin(k)
                else:
                    # Use remaining features for additional complexity
                    hamiltonian[i, j] = f * torch.sin((j-3) * k * 0.5) + f * torch.cos((j-3) * k * 0.5)
    
    elif model_type == "2d_a":
        # 2D A class: H(k) = h(k)·σ
        k_grid = int(np.sqrt(k_points))
        hamiltonian = torch.zeros(batch_size, 3, k_grid, k_grid, device=features.device)
        
        for i in range(batch_size):
            # Create 2D k-grid
            kx = torch.linspace(-np.pi, np.pi, k_grid, device=features.device)
            ky = torch.linspace(-np.pi, np.pi, k_grid, device=features.device)
            KX, KY = torch.meshgrid(kx, ky, indexing='ij')
            
            # Use features to parameterize h(k) components
            f1, f2, f3 = features[i, :3]
            
            # hx(k) = f1 * sin(kx) * cos(ky)
            hamiltonian[i, 0] = f1 * torch.sin(KX) * torch.cos(KY)
            
            # hy(k) = f2 * cos(kx) * sin(ky)
            hamiltonian[i, 1] = f2 * torch.cos(KX) * torch.sin(KY)
            
            # hz(k) = f3 * (cos(kx) + cos(ky) - 2)
            hamiltonian[i, 2] = f3 * (torch.cos(KX) + torch.cos(KY) - 2)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return hamiltonian

def compute_topological_loss(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    auxiliary_weight: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Compute loss for topological ML encoder.
    
    Args:
        predictions: Output from topological ML encoder
        targets: Ground truth topological labels
        auxiliary_weight: Weight for auxiliary loss terms
    
    Returns:
        Dictionary containing loss components
    """
    # Main classification loss
    if 'topological_logits' in predictions:
        main_loss = F.cross_entropy(predictions['topological_logits'], targets)
    elif 'chern_logits' in predictions:
        main_loss = F.cross_entropy(predictions['chern_logits'], targets)
    else:
        raise ValueError("No topological logits found in predictions")
    
    # Feature consistency loss (optional)
    feature_loss = 0.0
    if 'topological_features' in predictions:
        # Encourage features to be discriminative
        features = predictions['topological_features']
        feature_loss = torch.mean(torch.var(features, dim=0))  # Encourage feature diversity
    
    # Total loss
    total_loss = main_loss + auxiliary_weight * feature_loss
    
    return {
        'total_loss': total_loss,
        'main_loss': main_loss,
        'feature_loss': feature_loss
    } 