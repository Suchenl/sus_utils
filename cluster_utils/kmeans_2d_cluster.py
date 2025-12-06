import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Literal

def kmeans_2d_cluster(data, n_clusters=2, use_intensity=True, 
                      data_norm: Literal['minmax', 'zscore', 'log', 'none'] = 'minmax',
                      viz=False, viz_path=None):
    """
    Perform KMeans clustering on a 2D matrix [H, W].

    Args:
        data (np.ndarray or torch.Tensor): input matrix [H, W].
        n_clusters (int): number of clusters for KMeans (default=2).
        use_intensity (bool): whether to include pixel intensity as a feature (default=True).
        norm (str): preprocessing method for intensity values:
                    - "minmax" : scale to [0,1] (default)
                    - "zscore" : standardize to mean=0, std=1
                    - "log"    : log(1+x) then min-max scale
                    - "none"   : no normalization
        viz (bool): whether to visualize result (imshow).
        viz_path (str): path to save visualization (default=None).

    Returns:
        cluster_map (np.ndarray): [H, W] cluster labels.
    """
    # convert torch → numpy
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    assert data.ndim == 2, "Input must be a 2D matrix [H, W]."

    H, W = data.shape
    values = data.astype(np.float32).flatten()

    # ----- preprocess intensity -----
    if data_norm == "minmax":
        vmin, vmax = values.min(), values.max()
        if vmax > vmin:
            values = (values - vmin) / (vmax - vmin)
    elif data_norm == "zscore":
        mean, std = values.mean(), values.std()
        if std > 1e-6:
            values = (values - mean) / std
    elif data_norm == "log":
        values = np.log1p(values)
        vmin, vmax = values.min(), values.max()
        if vmax > vmin:
            values = (values - vmin) / (vmax - vmin)
    elif data_norm == "none":
        pass
    else:
        raise ValueError(f"Unsupported norm: {data_norm}")

    # ----- build features -----
    xs, ys = np.meshgrid(np.arange(W) / W, np.arange(H) / H)
    if use_intensity:
        X = np.stack([xs.flatten(), ys.flatten(), values], axis=1)
    else:
        X = np.stack([xs.flatten(), ys.flatten()], axis=1)

    # ----- run kmeans -----
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)

    # reshape back to H×W
    cluster_map = labels.reshape(H, W)

    # visualize
    if viz:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input data")
        plt.imshow(data, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"KMeans clustering (k={n_clusters})")
        plt.imshow(cluster_map, cmap="tab20")
        plt.axis("off")

        plt.tight_layout()
        if viz_path is None:
            plt.show()
        else:
            plt.savefig(viz_path)
        plt.close()
    else:
        pass
    return cluster_map
