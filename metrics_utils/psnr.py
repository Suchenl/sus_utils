import torch
import torch.nn.functional as F

class PSNR:
    """
    Peak Signal-to-Noise Ratio
    Args:
        data_range: maximum possible pixel value (1.0 for normalized images)
        reduction: 'none' | 'mean' | 'sum'
    Input:
        img_pred, img_gt: [B, C, H, W], float tensors in [0, data_range]
    Output:
        [B] if reduction='none', scalar otherwise
    """
    def __init__(self, data_range=1.0, reduction='none'):
        self.data_range = data_range
        self.reduction = reduction

    def __call__(self, img_pred: torch.Tensor, img_gt: torch.Tensor):
        assert img_pred.shape == img_gt.shape, "Shapes must match"
        B = img_pred.shape[0]
        mse = ((img_pred - img_gt) ** 2).view(B, -1).mean(dim=1)
        psnr = 10.0 * torch.log10(self.data_range ** 2 / mse.clamp_min(1e-10))

        if self.reduction == 'mean':
            return psnr.mean()
        elif self.reduction == 'sum':
            return psnr.sum()
        return psnr
