import torch
import torch.nn as nn
import lpips
from typing import *


class GANLoss(nn.Module):
    def __init__(self, 
                 gan_mode: Literal['vanilla', 'lsgan'] = 'vanilla',
                 use_label_smoothing: bool = False,
                 smooth_real: float = 0.9,
                 smooth_fake: float = 0.1,
                 reduction: str = 'mean'):
        """
        Enhanced GAN loss with PatchGAN support, BCEWithLogitsLoss and label smoothing
        
        Args:
            gan_mode: 'vanilla' for BCEWithLogitsLoss, 'lsgan' for MSE loss
            use_label_smoothing: Whether to apply label smoothing
            smooth_real: Target value for real samples when smoothing is enabled
            smooth_fake: Target value for fake samples when smoothing is enabled  
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.gan_mode = gan_mode
        self.use_label_smoothing = use_label_smoothing
        self.smooth_real = smooth_real
        self.smooth_fake = smooth_fake
        self.reduction = reduction
        
        if gan_mode == 'vanilla':
            # BCEWithLogitsLoss is more numerically stable than Sigmoid + BCELoss
            self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        elif gan_mode == 'lsgan':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported GAN mode: {gan_mode}. Choose 'vanilla' or 'lsgan'")

    def get_target_tensor(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Create target tensor for PatchGAN output
        """        
        if is_real:
            if self.use_label_smoothing:
                target_val = self.smooth_real
            else:
                target_val = 1.0
        else:
            if self.use_label_smoothing:
                target_val = self.smooth_fake
            else:
                target_val = 0.0
        
        # For PatchGAN: create target tensor with same spatial dimensions as prediction
        return torch.full_like(prediction, target_val)

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Compute GAN loss for PatchGAN output
        
        Args:
            prediction: Output from discriminator (logits, not probabilities)
                        For PatchGAN: shape [batch, 1, height, width] 
            is_real: Whether the input is real (True) or fake (False)
        """
        if prediction is None:
            return torch.tensor(0, device=self.device)
        target_tensor = self.get_target_tensor(prediction, is_real)
        return self.criterion(prediction, target_tensor)