import torch, math
from typing import Literal

def compute_data_auc(
    data: torch.Tensor,
    mode: Literal["diag", "fixed", "observed_max", "percentile"] = "diag",
    fixed_value: float = 5.0,
    percentile: float = 99.0,
    steps: int = 100,
    ) -> torch.Tensor:
    """
    General-purpose AUC computation for spatial tensors.
    Computes normalized area under cumulative distribution curve (CDF).
    
    Args:
        data: [B, H, W] tensor (e.g., error map, EPE map, confidence map, etc.)
        mode: one of ["diag", "fixed", "observed_max", "percentile"]
        fixed_value: used when mode == "fixed"
        percentile: percentile cutoff when mode == "percentile"
        steps: number of threshold steps for numerical integration
        device: computation device
        
    Returns:
        auc: [B] tensor, normalized AUC values in [0, 1]
    """
    B, H, W = data.shape
    device = data.device
    
    data_flat = data.view(B, -1)
    # === Determine integration upper bound ===
    if mode == "diag":
        max_val = math.sqrt(H * H + W * W)
        upper = torch.full((B,), max_val, device=device)
    elif mode == "fixed":
        upper = torch.full((B,), fixed_value, device=device)
    elif mode == "percentile":
        upper = torch.quantile(data_flat, q=percentile / 100.0, dim=1).clamp_min(1e-6)
    elif mode == "observed_max":
        upper = data_flat.max(dim=1).values.clamp_min(1e-6)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # === Build global threshold grid (shared among all samples) ===
    max_global = upper.max().item()
    thresholds = torch.linspace(0, float(max_global), steps, device=device)  # [steps]

    # === Compute per-sample CDF ===
    cdf = torch.stack([(data_flat < th).float().mean(dim=1) for th in thresholds], dim=1)  # [B, steps]

    # === Integrate using trapezoidal rule ===
    auc_raw = torch.trapz(cdf, thresholds, dim=1)  # [B]
    auc = auc_raw / upper  # normalize to [0,1] range (approx)
    return auc
