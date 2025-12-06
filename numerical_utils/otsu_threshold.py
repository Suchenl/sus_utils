import torch

def otsu_threshold(x: torch.Tensor, nbins: int = 256, save_hist: bool = False, save_path: str = None) -> float:
    """
    Compute Otsu's threshold for a 1D tensor.

    Otsu's method:
        - It assumes the data can be separated into two classes (e.g., foreground vs background).
        - For each possible threshold (candidate split point), it computes the "between-class variance":
              variance = w1 * w2 * (mean1 - mean2)^2
          where:
              w1 = probability (weight) of class 1
              w2 = probability (weight) of class 2
              mean1 = mean of values in class 1
              mean2 = mean of values in class 2
        - The optimal threshold is the one that maximizes this variance.
        - Intuition: the two groups are best separated when their means are far apart 
          and each group has enough samples.

    Args:
        x (torch.Tensor): 1D tensor of shape [n].
        nbins (int): number of histogram bins.

    Returns:
        float: optimal threshold value that best separates the data into two groups.
    """
    if x.ndim != 1:
        x = x.flatten()
    # compute histogram of values
    x_min, x_max = float(x.min()), float(x.max())
    hist = torch.histc(x, bins=nbins, min=x_min, max=x_max).float()
    bin_edges = torch.linspace(x_min, x_max, nbins+1, device=x.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # cumulative sums of weights for each threshold
    weight1 = torch.cumsum(hist, dim=0)
    weight2 = hist.sum() - weight1

    # cumulative means for each threshold
    mean1 = torch.cumsum(hist * bin_centers, dim=0) / (weight1 + 1e-6)
    mean2 = (hist.sum() * bin_centers.mean() - torch.cumsum(hist * bin_centers, dim=0)) / (weight2 + 1e-6)

    # between-class variance for each possible threshold
    between_var = weight1 * weight2 * (mean1 - mean2) ** 2

    # best threshold is the one that maximizes between-class variance
    idx = torch.argmax(between_var)
    threshold = bin_centers[idx].item()

    if save_hist:
        viz_hist(x, nbins=nbins, threshold=threshold, save_path=save_path)
        
    return threshold

import matplotlib.pyplot as plt
def viz_hist(x: torch.Tensor, nbins: int = 256, threshold: float = None, save_path: str = None):
    plt.figure(figsize=(6,4))
    plt.hist(x.cpu().numpy(), bins=nbins, color='skyblue', alpha=0.7, density=True)
    if threshold is not None:
        plt.axvline(threshold, color='red', linestyle='--', label=f'Otsu threshold = {threshold:.4f}')
    plt.xlabel("Error value")
    plt.ylabel("Density")
    plt.title("Histogram of error with Otsu threshold")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else: 
        plt.show()
    plt.close()