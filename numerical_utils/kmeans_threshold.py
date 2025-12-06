import torch

def kmeans_threshold(x: torch.Tensor, max_iters: int = 100, tol: float = 1e-4) -> float:
    """
    Simple 1D k-means with k=2. Threshold = midpoint of the two cluster centers.
    Works well when there's a small tail cluster separated from a dominant background.
    """
    x = x.flatten().to(torch.float32)
    n = x.numel()
    if n == 0:
        return 0.0
    x_min = float(x.min()); x_max = float(x.max())
    if x_min == x_max:
        return x_min

    # init centers to min and max
    c0 = x_min
    c1 = x_max
    for _ in range(max_iters):
        # assign to nearest center
        d0 = torch.abs(x - c0)
        d1 = torch.abs(x - c1)
        mask = (d1 < d0)  # True -> cluster1 (center c1)
        # if one cluster becomes empty, break
        if mask.sum().item() == 0 or mask.sum().item() == n:
            break
        new_c0 = float(x[~mask].mean().item()) if (~mask).sum().item() > 0 else c0
        new_c1 = float(x[mask].mean().item()) if mask.sum().item() > 0 else c1
        if abs(new_c0 - c0) < tol and abs(new_c1 - c1) < tol:
            c0, c1 = new_c0, new_c1
            break
        c0, c1 = new_c0, new_c1
    threshold = (c0 + c1) / 2.0
        
    return float(threshold)


