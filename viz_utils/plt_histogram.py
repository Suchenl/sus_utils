import torch
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