import torch
import torch.nn.functional as F
import math
from typing import Optional

class ConfidenceSimilarity:
    """
    Compute similarity metrics between two confidence maps (e.g., predicted vs. GT).

    Inputs:
        conf_pred, conf_gt: tensors of shape [B, 1, H, W] or [B, H, W], in [0,1]
        valid_mask: optional [B, 1, H, W] or [B, H, W], where 1=valid pixel

    Returns a dict with:
        'Soft Dice' : [B]
        'Dice@0.7'  : [B]
        'Dice@0.5'  : [B]
        'Dice@0.3'  : [B]
        'AUC(Dice)' : [B]
    """
    def __init__(self, threshold_num=20):
        self.threshold_num = threshold_num
        
    @torch.no_grad()
    def __call__(
        self,
        conf_pred: torch.Tensor,
        conf_gt: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            conf_pred: [B, 1, H, W] or [B, H, W]
            conf_gt: [B, 1, H, W] or [B, H, W]
            valid_mask: [B, 1, H, W] or [B, H, W] or None
        """
        eps = 1e-8
        device = conf_pred.device
        T = self.threshold_num

        # reshape to [B, H, W]
        if conf_pred.ndim == 4:
            conf_pred = conf_pred.squeeze(1)
        if conf_gt.ndim == 4:
            conf_gt = conf_gt.squeeze(1)
        if valid_mask is not None:
            if valid_mask.ndim == 4:
                valid_mask = valid_mask.squeeze(1)
        else:
            valid_mask = torch.ones_like(conf_gt, device=device)

        metrics = {}

        # 1️⃣ Soft Dice
        inter = (conf_pred * conf_gt * valid_mask).sum(dim=(1, 2))
        union = (conf_pred * valid_mask).sum(dim=(1, 2)) + (conf_gt * valid_mask).sum(dim=(1, 2))
        soft_dice = (2 * inter + eps) / (union + eps)
        metrics["Soft Dice"] = soft_dice

        # 2️⃣ Vectorized Dice Curve for thresholds [0,1]
        thresholds = torch.linspace(0.0, 1.0, T, device=device)  # finer sampling improves AUC
        B, H, W = conf_pred.shape

        # Broadcast thresholds: shape [B, T, H, W]
        pred_bin = (conf_pred.unsqueeze(1) >= thresholds.view(1, T, 1, 1)).float()
        gt_bin = (conf_gt.unsqueeze(1) >= thresholds.view(1, T, 1, 1)).float()
        mask = valid_mask.unsqueeze(1)

        inter = (pred_bin * gt_bin * mask).sum(dim=(2, 3))
        union = (pred_bin.sum(dim=(2, 3)) + gt_bin.sum(dim=(2, 3))).clamp_min(eps)
        dice_curve = (2 * inter + eps) / (union + eps)  # [B, T]

        # 3️⃣ AUC(Dice)
        auc_dice = torch.trapz(dice_curve, thresholds, dim=1)
        metrics["AUC(Dice)"] = auc_dice

        # 4️⃣ Dice@thresholds (0.7, 0.5, 0.3)
        query_ths = [0.7, 0.5, 0.3]
        for qt in query_ths:
            idx = (torch.abs(thresholds - qt)).argmin().item()
            if torch.abs(thresholds[idx] - qt) < 1e-3:
                metrics[f"Dice@{qt}"] = dice_curve[:, idx]
            else:
                # fallback (only rare case if threshold grid not aligned)
                pred_bin = (conf_pred >= qt).float()
                gt_bin = (conf_gt >= qt).float()
                inter = (pred_bin * gt_bin * valid_mask).sum(dim=(1, 2))
                union = (pred_bin.sum(dim=(1, 2)) + gt_bin.sum(dim=(1, 2))).clamp_min(eps)
                metrics[f"Dice@{qt}"] = (2 * inter + eps) / (union + eps)

        return metrics
