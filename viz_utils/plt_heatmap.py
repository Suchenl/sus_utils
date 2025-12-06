import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Literal
from PIL import Image

def plot_easy_heatmap(npdata, cmap_name='viridis'):
    """ 
    Plot a heatmap from a numpy array.
    Args:
        npdata: numpy array to plot, range [0, 1]
    """
    cmap = plt.get_cmap(cmap_name)
    heatmap_rgba = cmap(npdata.squeeze())
    heatmap_rgb_uint8 = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb_uint8)
    return heatmap_pil
        
def plot_heatmap(data: Union[torch.Tensor, np.ndarray, Image.Image], 
                 title: str = "Heatmap", 
                 cmap: Literal['viridis', 'hot', 'jet'] = 'viridis', 
                 save_path: Optional[str] = None, 
                 show: bool = False):
    """
    Visualizes a 2D data structure (Tensor, NumPy array, or PIL Image) as a heatmap.

    - For torch.Tensor and np.ndarray, it squeezes the data to remove singleton dimensions.
    - For PIL.Image, it converts color images (RGB/RGBA) to grayscale before plotting.

    Args:
        data (Union[torch.Tensor, np.ndarray, Image.Image]): 
            The input 2D data to visualize. It can be a PyTorch Tensor, a NumPy array,
            or a PIL Image object.
        title (str): 
            The title of the plot. Defaults to "Heatmap".
        cmap (str): 
            The colormap to use for the heatmap (e.g., 'viridis', 'hot', 'jet'). 
            Defaults to 'viridis'.
        save_path (Optional[str]): 
            If provided, the plot will be saved to this file path (e.g., 'my_heatmap.png'). 
            Defaults to None.
        show (bool): 
            If True, the plot will be displayed in an interactive window. 
            Defaults to True.
    """
    # --- 1. Input Validation and Data Preparation ---
    data_np = None

    if isinstance(data, torch.Tensor):
        # Handle PyTorch Tensor
        # Squeeze the tensor, move to CPU, detach, and convert to NumPy
        tensor_squeezed = data.squeeze()
        data_np = tensor_squeezed.cpu().detach().numpy()

    elif isinstance(data, np.ndarray):
        # Handle NumPy Array
        # Squeeze the array to remove any singleton dimensions
        data_np = data.squeeze()

    elif isinstance(data, Image.Image):
        # Handle PIL Image
        # If the image is in color (RGB/RGBA), convert it to grayscale ('L' mode)
        # because a heatmap visualizes single-channel intensity.
        if data.mode in ['RGB', 'RGBA']:
            print("Info: Color PIL Image provided. Converting to grayscale for heatmap visualization.")
            data = data.convert('L')
        data_np = np.array(data)

    else:
        raise TypeError(f"Unsupported data type: {type(data)}. "
                        "Supported types are torch.Tensor, np.ndarray, and PIL.Image.Image.")

    # Final check to ensure the processed data is 2D
    if data_np.ndim != 2:
        raise ValueError(f"Processed data must be 2D, but got shape {data_np.shape} "
                         f"from original input of type {type(data)}.")

    # --- 2. Create the Plot ---
    plt.figure(figsize=(8, 6))
    im = plt.imshow(data_np, cmap=cmap)
    
    # Add a color bar to show the mapping of values to colors
    plt.colorbar(im, label="Value")
    
    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Height", fontsize=12)
    
    # --- 3. Save or Display the Plot ---
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
        
    if show:
        plt.show()
    
    # Close the plot to free up memory
    plt.close()
