"""
Mixup augmentation for better generalization in topology classification
"""

import torch
import numpy as np

def mixup_data(x, y, alpha=0.2):
    """
    Apply mixup augmentation to features and labels
    
    Args:
        x: Input features (batch_size, feature_dim)
        y: Labels (batch_size,)
        alpha: Mixup parameter (higher = more mixing)
    
    Returns:
        mixed_x: Mixed features
        y_a, y_b: Original labels for mixed samples
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute mixup loss
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Original labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MixupTrainer:
    """
    Trainer with mixup augmentation
    """
    
    def __init__(self, model, criterion, alpha=0.2):
        self.model = model
        self.criterion = criterion
        self.alpha = alpha
    
    def train_step(self, x, y):
        """
        Single training step with mixup
        """
        # Apply mixup
        mixed_x, y_a, y_b, lam = mixup_data(x, y, self.alpha)
        
        # Forward pass
        outputs = self.model({'mixed_features': mixed_x})
        
        # Compute mixup loss
        loss = mixup_criterion(self.criterion, outputs['logits'], y_a, y_b, lam)
        
        return loss, outputs