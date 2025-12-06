import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Literal


def data_to_heatmap(
    data: Optional[Union[np.ndarray, torch.Tensor]],
    cmap_name: str = 'viridis',
    out_type: Literal['ndarray', 'tensor'] = 'tensor'
):
    """ 
    Convert scalar maps to RGB heatmaps using matplotlib colormaps.

    Args:
        data: Input array, supports shapes [H, W], [1, H, W], or [N, 1, H, W].
        cmap_name: Name of the matplotlib colormap (e.g., 'viridis', 'plasma').
        out_type: Return type: 'tensor' or 'ndarray'.

    Returns:
        Heatmap with shape [N, 3, H, W], range [0, 1].
    """
    # ---- 1. Convert to numpy
    if isinstance(data, torch.Tensor):
        npdata = data.detach().cpu().numpy()
    else:
        npdata = np.asarray(data)

    # ---- 2. Normalize shape to [N, 1, H, W]
    if npdata.ndim == 2:  # [H, W]
        npdata = npdata[None, None, :, :]
    elif npdata.ndim == 3:  # [1, H, W] or [C, H, W]
        if npdata.shape[0] != 1:
            raise ValueError(f"Expected single-channel data, got shape {npdata.shape}")
        npdata = npdata[None, :, :, :]
    elif npdata.ndim != 4:
        raise ValueError(f"Unsupported input shape {npdata.shape}, expected [N,1,H,W], [1,H,W], or [H,W]")

    N, C, H, W = npdata.shape
    assert C == 1, f"Expected 1 channel, got {C}"

    # ---- 3. Normalize values to [0,1]
    if npdata.min() < -0.1:  # handle [-1,1] range
        npdata = (npdata + 1) / 2
    npdata = np.clip(npdata, 0, 1)

    # ---- 4. Apply colormap
    cmap = plt.get_cmap(cmap_name)
    heatmaps = []
    for i in range(N):
        rgba = cmap(npdata[i, 0])[:, :, :3]  # [H, W, 3]
        heatmaps.append(rgba)
    heatmaps = np.stack(heatmaps, axis=0).transpose(0, 3, 1, 2)  # [N, 3, H, W]

    # ---- 5. Output format
    if out_type == 'tensor':
        return torch.from_numpy(heatmaps).float()
    return heatmaps
