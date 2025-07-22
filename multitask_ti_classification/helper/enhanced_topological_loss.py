import torch.nn as nn
import torch 

class EnhancedTopologicalLoss(nn.Module):
    """
    Enhanced loss function incorporating multiple topological constraints
    and consistency checks based on the Nature paper approach.
    """
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.2, delta=0.1, epsilon=0.1):
        super().__init__()
        self.alpha = alpha    # Main classification loss weight
        self.beta = beta      # Auxiliary classification loss weight  
        self.gamma = gamma    # Topological consistency weight
        self.delta = delta    # Confidence regularization weight
        self.epsilon = epsilon # Feature regularization weight
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.aux_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        """
        Enhanced loss computation
        Args:
            outputs: dict with 'logits', 'aux_logits', 'confidence', 'topo_features'
            targets: ground truth labels
        """
        # Main classification loss
        main_loss = self.classification_loss(outputs['logits'], targets)
        total_loss = self.alpha * main_loss
        
        # Auxiliary topological classification loss
        if 'aux_logits' in outputs:
            aux_loss = self.aux_loss(outputs['aux_logits'], targets)
            total_loss += self.beta * aux_loss
        
        # Topological consistency constraints
        if 'topo_features' in outputs:
            topo_loss = self._compute_topological_consistency(outputs['topo_features'], targets)
            total_loss += self.gamma * topo_loss
        
        # Confidence regularization
        if 'confidence' in outputs:
            conf_loss = self._compute_confidence_loss(outputs['confidence'], outputs['logits'], targets)
            total_loss += self.delta * conf_loss
        
        # Feature regularization
        if 'graph_features' in outputs:
            reg_loss = self._compute_regularization_loss(outputs['graph_features'])
            total_loss += self.epsilon * reg_loss
        
        return total_loss
    
    def _compute_topological_consistency(self, topo_features, targets):
        """Compute topological invariant consistency losses"""
        consistency_loss = 0.0
        
        # Chern number consistency (should be integer-like)
        if 'chern' in topo_features:
            chern = topo_features['chern']
            chern_consistency = torch.mean((chern - torch.round(chern))**2)
            consistency_loss += chern_consistency
        
        # Z2 invariant consistency (should be binary-like)
        if 'z2' in topo_features:
            z2 = topo_features['z2']
            z2_consistency = torch.mean((z2 - torch.round(z2))**2)
            consistency_loss += z2_consistency
        
        # Gap consistency (non-negative for insulators)
        if 'gap' in topo_features:
            gap = topo_features['gap']
            # Encourage positive gaps for trivial materials (class 0)
            trivial_mask = (targets == 0).float().unsqueeze(1)
            gap_consistency = torch.mean(trivial_mask * F.relu(-gap))  # Penalty for negative gaps in trivial
            consistency_loss += gap_consistency
        
        # Physical constraint: topological materials should have small gaps
        if 'gap' in topo_features:
            gap = topo_features['gap']
            topo_mask = (targets > 0).float().unsqueeze(1)  # Topological materials
            large_gap_penalty = torch.mean(topo_mask * F.relu(gap - 0.5))  # Penalty for large gaps in topo
            consistency_loss += large_gap_penalty
        
        return consistency_loss
    
    def _compute_confidence_loss(self, confidence, logits, targets):
        """Compute confidence calibration loss"""
        # Get prediction probabilities
        probs = F.softmax(logits, dim=1)
        max_probs, predictions = torch.max(probs, dim=1)
        
        # Correct predictions should have high confidence
        correct_mask = (predictions == targets).float()
        
        # Confidence should correlate with prediction accuracy
        confidence_loss = F.mse_loss(confidence.squeeze(), correct_mask * max_probs)
        
        # Encourage confident predictions for correct classifications
        confidence_reg = -torch.mean(correct_mask * torch.log(confidence.squeeze() + 1e-8))
        
        return confidence_loss + 0.1 * confidence_reg
    
    def _compute_regularization_loss(self, graph_features):
        """Compute feature regularization to prevent overfitting"""
        # L2 regularization on graph features
        l2_reg = torch.mean(torch.norm(graph_features, p=2, dim=1))
        
        # Encourage diverse feature representations
        feature_diversity = -torch.mean(torch.std(graph_features, dim=0))
        
        return l2_reg + 0.1 * feature_diversity


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in topological materials
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting that adjusts based on training progress
    """
    def __init__(self, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses):
        """
        Args:
            losses: list of loss values [main_loss, aux_loss, consistency_loss]
        """
        precision = torch.exp(-self.log_vars)
        weighted_losses = []
        
        for i, loss in enumerate(losses):
            weighted_loss = precision[i] * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
            
        return sum(weighted_losses)

