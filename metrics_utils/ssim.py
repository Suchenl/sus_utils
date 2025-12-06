import torch
import torch.nn.functional as F

class SSIM:
    """
    Structural Similarity Index Measure
    Args:
        data_range: max pixel value
        window_size: Gaussian window size
        sigma: Gaussian sigma
        K: constants (K1, K2)
        reduction: 'none' | 'mean' | 'sum'
    Input:
        img_pred, img_gt: [B, C, H, W]
    Output:
        [B] if reduction='none', scalar otherwise
    """
    def __init__(self, data_range=1.0, window_size=11, sigma=1.5, K=(0.01, 0.03), reduction='none'):
        self.data_range = data_range
        self.window_size = window_size
        self.sigma = sigma
        self.K1, self.K2 = K
        self.reduction = reduction

        # create 1D Gaussian kernel
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        self.register_buffer = g.view(1, 1, 1, window_size)

    def gaussian_filter(self, x: torch.Tensor):
        g = self.register_buffer.to(x.device)
        x = F.conv2d(x, g.expand(x.shape[1], -1, 1, self.window_size), 
                     padding=(0, self.window_size//2), groups=x.shape[1])
        x = F.conv2d(x, g.transpose(-1, -2).expand(x.shape[1], -1, self.window_size, 1), 
                     padding=(self.window_size//2, 0), groups=x.shape[1])
        return x

    def __call__(self, img_pred: torch.Tensor, img_gt: torch.Tensor):
        assert img_pred.shape == img_gt.shape, "Shapes must match"
        B, C, H, W = img_pred.shape
        device = img_pred.device

        C1 = (self.K1 * self.data_range) ** 2
        C2 = (self.K2 * self.data_range) ** 2

        mu1 = self.gaussian_filter(img_pred)
        mu2 = self.gaussian_filter(img_gt)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.gaussian_filter(img_pred ** 2) - mu1_sq
        sigma2_sq = self.gaussian_filter(img_gt ** 2) - mu2_sq
        sigma12 = self.gaussian_filter(img_pred * img_gt) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim = ssim_map.view(B, -1).mean(dim=1)

        if self.reduction == 'mean':
            return ssim.mean()
        elif self.reduction == 'sum':
            return ssim.sum()
        return ssim