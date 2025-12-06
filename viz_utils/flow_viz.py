from torchvision.utils import flow_to_image
import torch
from PIL import Image
import numpy as np
from typing import *
import matplotlib.pyplot as plt

def viz_flow(flow: torch.Tensor, 
             viz_type: Literal["rgb", "arrows", "both"] = "rgb",
             background_image: Optional[Union[torch.Tensor, Image.Image]] = None,
             step: int = 32,
             output_type: Literal["pil", "numpy"] = "pil",
             ) -> [Image.Image, np.ndarray]:
    """ Visualizes a dense optical flow field using arrows (quiver plot) on an optional background.
    Args:
        flow (torch.Tensor): The optical flow tensor, shape [1, 2, H, W].
        output_type (Literal["numpy", "pil"], optional): The desired output format.
                                                        Defaults to "pil".
        viz_type (Literal["rgb", "arrows", "both"], optional): The desired output format.
                                                        Defaults to "rgb".
        background_image (Optional[Union[torch.Tensor, Image.Image]], optional):
            Background image, either:
                - Tensor: shape [1, 3, H, W] or [1, 1, H, W], values in [-1, 1] or [0, 1].
                - PIL.Image: will be resized to (W, H).
            Defaults to white background.
        step (int, optional): The grid step for drawing arrows. A higher value means fewer arrows.
                                                  Defaults to 32.
    Returns:
        Union[np.ndarray, Image.Image]: The resulting visualization.
    """                 
    if viz_type == "rgb":
        return viz_flow_with_rgb(flow, output_type)
    elif viz_type == "arrows":
        return viz_flow_with_arrows(flow, background_image, step, output_type)
    elif viz_type == "both":
        return viz_flow_with_rgb(flow, output_type), viz_flow_with_arrows(flow, background_image, step, output_type)
    
def viz_flow_with_rgb(flow: torch.Tensor, output_type: Literal["pil", "numpy"] = "pil") -> [Image.Image, np.ndarray]:
    flow_img_tensor = flow_to_image(flow)
    flow_img_np = np.transpose(flow_img_tensor.cpu().detach().numpy(), [1, 2, 0])
    if output_type == "numpy":
        return flow_img_np
    elif output_type == "pil":
        flow_img_pil = Image.fromarray(flow_img_np)
        return flow_img_pil
    else:
        raise ValueError(f"Invalid output type: {output_type}. Please choose from 'numpy' or 'pil'.")

def viz_flow_with_arrows(
    flow: torch.Tensor,
    background_image: Optional[Union[torch.Tensor, Image.Image]] = None,
    step: int = 32,
    output_type: Literal["pil", "numpy"] = "pil"
    ) -> [Image.Image, np.ndarray]:
    """
    Visualizes a dense optical flow field using arrows (quiver plot) on an optional background.

    Args:
        flow (torch.Tensor): The optical flow tensor, shape [1, 2, H, W].
        background_image (Optional[Union[torch.Tensor, Image.Image]], optional):
            Background image, either:
                - Tensor: shape [1, 3, H, W] or [1, 1, H, W], values in [-1, 1] or [0, 1].
                - PIL.Image: will be resized to (W, H).
            Defaults to white background.
        step (int, optional): The grid step for drawing arrows. A higher value means fewer arrows.
                              Defaults to 32.
        output_type (Literal["numpy", "pil"], optional): The desired output format.
                                                        Defaults to "pil".

    Returns:
        Union[np.ndarray, Image.Image]: The resulting visualization.
    """
    # --- 1. Input Tensor Preparation ---
    # Ensure flow is on CPU and in NumPy format, shape (H, W, 2)
    if flow.ndim == 4:
        flow = flow.squeeze(0)
    flow_np = flow.detach().cpu().numpy().transpose(1, 2, 0)
    H, W, _ = flow_np.shape

    # --- 2. Prepare background ---
    if background_image is not None:
        if isinstance(background_image, torch.Tensor):
            bg_np = background_image.detach().cpu().numpy().transpose(1, 2, 0)
            # Normalize from [-1,1] → [0,1] if needed
            if bg_np.min() < -0.0001:
                bg_np = (bg_np + 1.0) / 2.0
            # Ensure shape (H, W, 3)
            if bg_np.shape[2] == 1:
                bg_np = np.repeat(bg_np, 3, axis=2)
        elif isinstance(background_image, Image.Image):
            bg_np = np.array(background_image.convert("RGB").resize((W, H))) / 255.0
        else:
            raise TypeError("background_image must be torch.Tensor or PIL.Image.Image")
    else:
        bg_np = np.ones((H, W, 3), dtype=np.float32)  # White background

    # --- 3. Create Grid for Quiver Plot ---
    y, x = np.mgrid[0:H:step, 0:W:step]
    u = flow_np[y, x, 0]
    v = flow_np[y, x, 1]

    # --- 4. Plotting with Matplotlib ---
    dpi = 80
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove padding

    ax.imshow(bg_np, cmap='gray', extent=[0, W, H, 0])
    ax.quiver(x, y, u, v, color='green',
              angles='xy', scale_units='xy', scale=1,
              headwidth=4.5, headlength=4)
    ax.axis('off')

    # --- 5. Convert Plot to Image (NumPy/PIL) ---
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plot_np = image_from_plot[:, :, :3]

    plt.close(fig)

    if output_type == "numpy":
        return plot_np
    elif output_type == "pil":
        return Image.fromarray(plot_np)
    else:
        raise ValueError(f"Invalid output type: {output_type}. Please choose from 'numpy' or 'pil'.")


def calc_mol_confidence(info_pred, var_max=10.0, var_min=0.0, epsilon=1e-6):
    """
    Calculates and normalizes a confidence map from the model's info tensor.

    This method is based on the probability density of the mixture of Laplacians
    distribution at the predicted mean. A higher probability density signifies
    higher model confidence.

    Args:
        info_pred (torch.Tensor): The model's info prediction output, with a
                                shape of [N, 4, H, W]. The channel order is
                                expected to be [w_logit0, w_logit1, raw_b0, raw_b1].
        var_max (float): The maximum value for the large b component.
        var_min (float): The minimum value for the small b component.
        epsilon (float): A small constant to prevent division by zero.

    Returns:
        torch.Tensor: The normalized confidence map, with a shape of [N, 1, H, W]
                    and values in the range [0, 1]. Higher values indicate
                    higher confidence.
    """
    if info_pred.shape[1] != 4:
        raise ValueError(f"Input info_pred is expected to have 4 channels, but got {info_pred.shape[1]}")

    # --- 1. Extract Parameters ---
    # Separate the mixture weights (logits) and the raw scale parameters.
    weights_logits = info_pred[:, :2, :, :]
    raw_b = info_pred[:, 2:, :, :]

    # Convert logits to actual mixture weights using Softmax.
    weights = torch.softmax(weights_logits, dim=1)  # -> Shape [N, 2, H, W]

    # --- 2. Process Scale Parameter b ---
    # Process and clamp b according to the original code's logic.
    log_b = torch.zeros_like(raw_b)
    
    # Large b component (corresponding to higher uncertainty).
    log_b[:, 0, :, :] = torch.clamp(raw_b[:, 0, :, :], min=0, max=var_max)
    # Small b component (corresponding to lower uncertainty).
    log_b[:, 1, :, :] = torch.clamp(raw_b[:, 1, :, :], min=var_min, max=0)
    
    # Convert from log space back to the real scale b.
    b = torch.exp(log_b)  # -> Shape [N, 2, H, W]

    # --- 3. Calculate Confidence (as Probability Density) ---
    # The Probability Density Function (PDF) of a Laplace distribution at its mean is 1 / (2*b).
    pdf_at_mean_0 = 1.0 / (2 * b[:, 0, :, :] + epsilon)
    pdf_at_mean_1 = 1.0 / (2 * b[:, 1, :, :] + epsilon)

    # The PDF of the mixture distribution is the weighted sum of the components' PDFs.
    # This serves as our unnormalized confidence map.
    confidence_unnormalized = weights[:, 0, :, :] * pdf_at_mean_0 + weights[:, 1, :, :] * pdf_at_mean_1
    # -> Shape [N, H, W]

    # --- 4. Normalize Each Image in the Batch ---
    # Scale the confidence map to the [0, 1] range for visualization.
    N, H, W = confidence_unnormalized.shape
    
    # Reshape to (N, H*W) to find min/max per image
    view_for_norm = confidence_unnormalized.view(N, -1)
    
    # Find min and max for each image in the batch. Shapes become [N, 1]
    c_min = torch.min(view_for_norm, dim=1, keepdim=True)[0]
    c_max = torch.max(view_for_norm, dim=1, keepdim=True)[0]

    # Reshape min/max to (N, 1, 1) for broadcasting
    c_min = c_min.view(N, 1, 1)
    c_max = c_max.view(N, 1, 1)
    
    # Apply min-max normalization in a single, vectorized operation
    confidence_normalized = (confidence_unnormalized - c_min) / (c_max - c_min + epsilon)

    return confidence_normalized.unsqueeze(1)  # [N, 1, H, W]
        