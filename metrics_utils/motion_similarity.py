import math
import torch
import torch.nn.functional as F
from .data_auc import compute_data_auc
from typing import Optional

class MotionSimilarity:
    """
    Compute optical flow evaluation metrics between predicted and GT flow.

    Inputs:
        flow_pred, flow_gt: tensors of shape [B, 2, H, W], units in pixels.

    Returns a dict with:
        'EPE' : [B, H, W]          (per-pixel endpoint error)
        'AEE' : [B]                (mean EPE per sample)
        'AAE' : [B]                (mean angular error per sample, in radians)
        'AUC(EPE)' : [B]                (normalized CDF AUC via external compute_auc)
        'Fl-all' : [B]             (KITTI outlier rate)
        '1px s0-10' : [B], '1px s10-40' : [B], '1px s40+' : [B]
        '3px s0-10' : [B], '3px s10-40' : [B], '3px s40+' : [B]
        '5px s0-10' : [B], '5px s10-40' : [B], '5px s40+' : [B]
    """
    def __call__(self, 
                 flow_pred: torch.Tensor, 
                 flow_gt: torch.Tensor,
                 valid_mask: Optional[torch.Tensor] = None):
        """
        Compute all metrics. If valid_mask is provided, only mask==1 pixels are considered.
        Args:
            flow_pred (torch.Tensor): [B, 2, H, W]
            flow_gt (torch.Tensor): [B, 2, H, W]
            valid_mask (torch.Tensor): [B, H, W] or [B, 1, H, W]
        """
        assert flow_pred.shape == flow_gt.shape, "Flow shapes must match"
        B, C, H, W = flow_pred.shape
        assert C == 2, "Flow should have 2 channels (u,v)"

        # handle valid mask
        if valid_mask is not None:
            if valid_mask.ndim == 4:
                valid_mask = valid_mask.squeeze(1)
            assert valid_mask.shape == (B, H, W), "Mask shape must match spatial dimensions"
            valid_mask = valid_mask.float()
        else:
            valid_mask = torch.ones((B, H, W), device=flow_gt.device)

        eps = 1e-6

        # --- 1) EPE ---
        epe = torch.norm(flow_pred - flow_gt, dim=1)  # [B, H, W], L2 norm

        # --- 2) AEE ---
        aee = (epe * valid_mask).sum(dim=(1, 2)) / valid_mask.sum(dim=(1, 2)).clamp_min(1.0)

        # --- 3) AAE ---
        dot = (flow_pred * flow_gt).sum(dim=1) # v1·v2
        norm_pred = flow_pred.norm(dim=1) # |v1|
        norm_gt = flow_gt.norm(dim=1)   # |v2|
        denom = (norm_pred * norm_gt).clamp_min(eps) # |v1||v2|
        cos_angle = (dot / denom).clamp(-1.0, 1.0) # cos(θ) = v1·v2 / (|v1||v2|)
        aae = (torch.acos(cos_angle) * valid_mask).sum(dim=(1, 2)) / valid_mask.sum(dim=(1, 2)).clamp_min(1.0)
        aae = aae * (180.0 / math.pi)  # convert radians to degrees
        
        # --- 4) Speed bins ---
        s = norm_gt
        bins = {
            "s0-10": (s <= 10.0),
            "s10-40": (s > 10.0) & (s <= 40.0),
            "s40+": (s > 40.0),
        }

        # --- 5) Initialize metrics dict ---
        metrics = {
            "EPE": epe,
            "AEE": aee,
            "AAE": aae,
        }

        # --- 6) multi-threshold statistics (1px, 3px, 5px) ---
        thresholds = [1.0, 3.0, 5.0]
        for t in thresholds:
            for name, mask in bins.items():
                valid_bin = valid_mask * mask
                num_valid = valid_bin.sum(dim=(1, 2)).clamp_min(1.0)
                num_ok = ((epe > t) * valid_bin).sum(dim=(1, 2))
                frac = num_ok / num_valid
                metrics[f"{int(t)}px {name}"] = frac

        # --- 7) Fl-all (KITTI-style outlier) ---
        mag = norm_gt
        outlier_mask = ((epe > 3.0) & ((epe / (mag + eps)) > 0.05)).float() * valid_mask
        fl_all = outlier_mask.sum(dim=(1, 2)) / valid_mask.sum(dim=(1, 2)).clamp_min(1.0)
        metrics["Fl-all"] = fl_all

        # --- 8) AUC ---
        metrics["AUC(EPE)"] = compute_data_auc(
            data=epe * valid_mask,  # masked input
            mode="fixed",
            fixed_value=5.0,
            steps=100,
            )

        # --- 9) Pixel count statistics ---
        total_pixels = torch.full((B,), H * W, device=flow_gt.device, dtype=torch.float32)
        valid_pixels = valid_mask.sum(dim=(1, 2))

        s0_10_pixels = (valid_mask * bins["s0-10"]).sum(dim=(1, 2))
        s10_40_pixels = (valid_mask * bins["s10-40"]).sum(dim=(1, 2))
        s40p_pixels = (valid_mask * bins["s40+"]).sum(dim=(1, 2))

        metrics.update({
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
            "s0-10_pixels": s0_10_pixels,
            "s10-40_pixels": s10_40_pixels,
            "s40+_pixels": s40p_pixels,
        })
        
        return metrics