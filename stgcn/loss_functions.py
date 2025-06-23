# loss_functions.py
"""
Custom loss functions for golf swing classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseCategoricalFocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-class classification.
    
    Args:
        gamma (float): Focusing parameter. Higher gamma reduces the relative loss 
                      for well-classified examples.
        weight (Tensor, optional): Manual rescaling weight given to each class.
        reduction (str): Specifies the reduction to apply to the output.
                        'mean' | 'sum' | 'none'
    """
    
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(SparseCategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # Class weight (Tensor of shape [num_classes])
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Forward pass of focal loss.
        
        Args:
            logits (Tensor): Predicted logits of shape [batch_size, num_classes]
            targets (Tensor): Ground truth labels of shape [batch_size]
            
        Returns:
            Tensor: Computed focal loss
        """
        # logits: [batch_size, num_classes]
        # targets: [batch_size] (int labels)
        log_probs = F.log_softmax(logits, dim=-1)  # Log probabilities
        probs = torch.exp(log_probs)  # Probabilities
        
        # Gather the log-probability and prob of the true class
        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)
        
        # Focal loss modulation
        focal_term = (1 - pt) ** self.gamma
        
        # Class weights (if provided)
        if self.weight is not None:
            at = self.weight.gather(0, targets.squeeze(1))
            log_pt = log_pt * at
        
        loss = -focal_term * log_pt
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss