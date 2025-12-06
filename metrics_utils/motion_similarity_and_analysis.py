import math
import torch
import torch.nn.functional as F
from .data_auc import compute_data_auc
from typing import Optional, List

class MotionSimilarityAndAnalysis:
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
        
        and several optional motion distribution and complexity statistics.
    """
    def __init__(self,
                 return_gt_bins: bool = False,
                 return_pred_bins: bool = False,
                 speed_bins: Optional[List[float]] = None,
                 return_gt_angle_var: bool = False,
                 return_pred_angle_var: bool = False,
                 return_valid_pxs: bool = False,
                 return_total_pxs: bool = False):
        """
        Args:
            - return_gt_bins: Whether to return the count of ground-truth flow magnitudes per bin.
            - return_pred_bins: Whether to return the count of predicted flow magnitudes per bin.
            - speed_bins: List of magnitude bin boundaries (e.g., [0,10,20,...,inf]).
            - return_gt_angle_var: Whether to return variance of ground-truth flow angles.
            - return_pred_angle_var: Whether to return variance of predicted flow angles.
            - return_valid_pxs: Whether to return the number of valid pixels (mask==1).
            - return_total_pxs: Whether to return the total number of pixels (H * W).
        """
        self.return_gt_bins = return_gt_bins
        self.return_pred_bins = return_pred_bins
        self.speed_bins = speed_bins or [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
        self.return_gt_angle_var = return_gt_angle_var
        self.return_pred_angle_var = return_pred_angle_var
        self.return_valid_pxs = return_valid_pxs
        self.return_total_pxs = return_total_pxs
        
    def __call__(self, 
                 flow_pred: torch.Tensor, 
                 flow_gt: torch.Tensor,
                 valid_mask: Optional[torch.Tensor] = None):
        """
        Compute all motion similarity metrics on a per-sample basis.
        Args:
            flow_pred: Predicted optical flow [B, 2, H, W].
            flow_gt: Ground-truth optical flow [B, 2, H, W].
            valid_mask: Optional binary mask [B, H, W] or [B, 1, H, W]; 1 = valid pixel.
        Returns:
            metrics: dict of computed statistics (each item is [B] or [B, ...]).
        """
        assert flow_pred.shape == flow_gt.shape, "Flow shapes must match"
        B, C, H, W = flow_pred.shape
        assert C == 2, "Flow must have 2 channels (u,v)"
        
        # --- Handle valid mask ---
        if valid_mask is not None:
            if valid_mask.ndim == 4:
                valid_mask = valid_mask.squeeze(1)
            assert valid_mask.shape == (B, H, W), "Mask shape must match spatial dimensions"
            valid_mask = valid_mask.float()
        else:
            valid_mask = torch.ones((B, H, W), device=flow_gt.device)

        eps = 1e-6

        # --- 1) EPE (End-Point Error) ---
        epe = torch.norm(flow_pred - flow_gt, dim=1)  # [B, H, W], L2 norm

        # --- 2) AEE (Average Endpoint Error) ---
        aee = (epe * valid_mask).sum(dim=(1, 2)) / valid_mask.sum(dim=(1, 2)).clamp_min(1.0)

        # --- 3) AAE (Average Angular Error, in degrees) ---
        dot = (flow_pred * flow_gt).sum(dim=1) # v1·v2
        norm_pred = flow_pred.norm(dim=1) # |v1|
        norm_gt = flow_gt.norm(dim=1)   # |v2|
        denom = (norm_pred * norm_gt).clamp_min(eps) # |v1||v2|
        cos_angle = (dot / denom).clamp(-1.0, 1.0) # cos(θ) = v1·v2 / (|v1||v2|)
        aae = (torch.acos(cos_angle) * valid_mask).sum(dim=(1, 2)) / valid_mask.sum(dim=(1, 2)).clamp_min(1.0)
        aae = aae * (180.0 / math.pi)  # convert radians to degrees
        
        # --- 4) Speed bins (KITTI-style categories) ---
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

        # --- 8) Optional: magnitude distribution (speed bins) ---
        if self.return_gt_bins or self.return_pred_bins:
            def bin_count(mag):
                counts = []
                for i in range(len(self.speed_bins) - 1):
                    low, high = self.speed_bins[i], self.speed_bins[i + 1]
                    mask_bin = (mag >= low) & (mag < high)
                    counts.append((mask_bin * valid_mask).sum(dim=(1, 2)))
                return torch.stack(counts, dim=1)  # [B, num_bins]

            if self.return_gt_bins:
                metrics["gt_speed_bins"] = bin_count(norm_gt)
            if self.return_pred_bins:
                metrics["pred_speed_bins"] = bin_count(norm_pred)

        # --- 9) Optional: angular variance (motion complexity) ---
        if self.return_gt_angle_var or self.return_pred_angle_var:
            def angle_variance(flow): # cirular variance
                angles = torch.atan2(flow[:, 1], flow[:, 0])  # [-pi, pi]
                vx = torch.cos(angles)
                vy = torch.sin(angles)
                mean_vx = (vx * valid_mask).sum(dim=(1, 2)) / valid_mask.sum(dim=(1, 2)).clamp_min(1.0)
                mean_vy = (vy * valid_mask).sum(dim=(1, 2)) / valid_mask.sum(dim=(1, 2)).clamp_min(1.0)
                R = torch.sqrt(mean_vx**2 + mean_vy**2)
                circular_variance = 1 - R
                return circular_variance

            if self.return_gt_angle_var:
                metrics["gt_angle_var"] = angle_variance(flow_gt)
            if self.return_pred_angle_var:
                metrics["pred_angle_var"] = angle_variance(flow_pred)

        # --- 10) Optional: valid and total pixel counts ---
        if self.return_valid_pxs:
            metrics["valid_pxs"] = valid_mask.sum(dim=(1, 2)).int()
        if self.return_total_pxs:
            metrics["total_pxs"] = torch.full((B,), H * W, device=flow_gt.device, dtype=torch.int)
                    
        return metrics