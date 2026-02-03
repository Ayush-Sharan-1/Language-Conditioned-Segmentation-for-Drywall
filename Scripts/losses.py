"""
Loss functions for binary segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    Computed on sigmoid outputs.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Logits [B, 1, H, W]
            targets: Binary masks [B, 1, H, W] with values {0, 1}
        
        Returns:
            Dice loss scalar
        """
        # Apply sigmoid to predictions
        pred_probs = torch.sigmoid(predictions)
        
        # Flatten tensors
        pred_flat = pred_probs.view(-1)
        target_flat = targets.view(-1)
        
        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss: BCE + Dice
    """
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Logits [B, 1, H, W]
            targets: Binary masks [B, 1, H, W] with values {0, 1}
        
        Returns:
            Combined loss scalar
        """
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        return bce + dice

