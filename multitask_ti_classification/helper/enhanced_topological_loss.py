import torch.nn as nn
import torch 

class EnhancedTopologicalLoss(nn.Module):
    """
    Specialized loss function for topological classification that incorporates
    physical constraints and topological invariant consistency.
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3):
        super().__init__()
        self.alpha = alpha  # Classification loss weight
        self.beta = beta    # Topological consistency weight
        self.gamma = gamma  # Regularization weight
        
        self.classification_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits, targets, topological_features=None):
        # Standard classification loss
        class_loss = self.classification_loss(logits, targets)
        
        total_loss = self.alpha * class_loss
        
        if topological_features is not None:
            # Topological consistency loss
            # Encourage physical constraints (e.g., Chern numbers should be integers)
            chern_features = topological_features[:, 0:1]  # First feature is Chern-like
            chern_consistency = torch.mean((chern_features - torch.round(chern_features))**2)
            
            # Z2 features should be binary-like
            z2_features = topological_features[:, 1:5]  # Next 4 features are Z2-like
            z2_consistency = torch.mean((z2_features - torch.round(z2_features))**2)
            
            topo_loss = chern_consistency + z2_consistency
            total_loss += self.beta * topo_loss
            
            # Regularization term to prevent overfitting
            reg_loss = torch.mean(torch.norm(topological_features, dim=-1))
            total_loss += self.gamma * reg_loss
        
        return total_loss

