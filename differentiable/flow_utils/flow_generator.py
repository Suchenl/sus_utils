#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2025 Yuzhuo Chen (Suchenl)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This file contains some functions that automatically generate dense optical flow fields / displacement fields.
The return type is a tensor of [B, 2, H, W].
"""

import torch
import math
from typing import Union, Tuple, Optional
from torchvision import transforms
from torch.nn import functional as F

def separable_gaussian_blur(tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Applies a Gaussian blur using a separable filter with 'reflect' padding
    to match the behavior of torchvision's implementation.
    
    This version correctly handles padding for the functional F.conv2d API.
    """
    # Step 1: Create the 1D Gaussian kernel (this part remains the same)
    kernel_1d = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device)
    kernel_1d -= kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()

    # Step 2: Prepare kernels for convolution (this part also remains the same)
    in_channels = tensor.shape[1]
    kernel_h = kernel_1d.view(1, 1, 1, kernel_size).expand(in_channels, 1, 1, kernel_size)
    kernel_v = kernel_1d.view(1, 1, kernel_size, 1).expand(in_channels, 1, kernel_size, 1)

    # --- Step 3: Pad first, then convolve with padding=0 ---
    
    # --- Horizontal Blur ---
    # Define the padding amounts for the left and right sides.
    # F.pad expects a tuple in the format (pad_left, pad_right, pad_top, pad_bottom).
    pad_h = (kernel_size // 2, kernel_size // 2, 0, 0)
    # Manually pad the tensor using reflection padding.
    padded_tensor = F.pad(tensor, pad_h, mode='reflect')
    # Convolve the padded tensor with zero padding.
    blurred_h = F.conv2d(padded_tensor, kernel_h, padding=0, groups=in_channels)
    
    # --- Vertical Blur ---
    # Define the padding amounts for the top and bottom sides.
    pad_v = (0, 0, kernel_size // 2, kernel_size // 2)
    # Manually pad the horizontally-blurred tensor.
    padded_blurred_h = F.pad(blurred_h, pad_v, mode='reflect')
    # Convolve the padded tensor with zero padding.
    blurred_hv = F.conv2d(padded_blurred_h, kernel_v, padding=0, groups=in_channels)
    
    return blurred_hv

def random_flow(B: int, 
                H: int, 
                W: int,
                sigma: float = 10.0,
                strength: float | Tuple[float, float] = 50.0,
                kernel_scale: float = 3.0,
                device: str = 'cpu', dtype: torch.dtype = torch.float32,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Generates a smooth, random optical flow field for elastic transformations.

    This function creates a random displacement field by generating random noise and
    then smoothing it with a Gaussian filter.

    Args:
        B (int): The batch size.
        H (int): The height of the image/flow field.
        W (int): The width of the image/flow field.
        sigma (float, optional): The standard deviation of the Gaussian filter.
                                 A larger sigma results in smoother distortions.
                                 Defaults to 10.0.
        strength (Union[float, Tuple[float, float]], optional):
                                 Controls the displacement intensity.
                                 - If a single float is provided, it is used for both
                                   horizontal (u) and vertical (v) strength.
                                 - If a tuple (max_u, max_v) is provided, it controls
                                   the horizontal and vertical strength independently.
                                 Defaults to 50.0.
        kernel_scale (float, optional): The scaling ratio of the Gaussian blur kernel relative to sigma.
        device (str, optional): The device to create the tensor on. Defaults to 'cpu'.
        dtype (torch.dtype, optional): The data type of the tensor. Defaults to torch.float32.
        generator (Optional[torch.Generator], optional): A PyTorch random number
                                                       generator. If None, the global
                                                       RNG is used. Defaults to None.
    Returns:
        torch.Tensor: The generated random optical flow tensor with a shape of [B, 2, H, W].
                      flow[:, 0] corresponds to the horizontal displacement (u).
                      flow[:, 1] corresponds to the vertical displacement (v).
    """
    # --- Step 1: Parse the 'strength' parameter ---
    # This is the new logic based on your suggestion.
    if isinstance(strength, (int, float)):
        strength_u = strength
        strength_v = strength
    elif isinstance(strength, (list, tuple)) and len(strength) == 2:
        strength_u, strength_v = strength
    else:
        raise ValueError(
            "strength must be a single float or a tuple/list of two floats."
        )

    # 2. Generate random noise.
    noise = torch.randn(B, 2, H, W, device=device, dtype=dtype, generator=generator)

    # 3. Smooth the noise with a Gaussian filter.
    kernel_size = 2 * int(kernel_scale * sigma) + 1
    # gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    # flow = gaussian_blur(noise)
    flow = separable_gaussian_blur(noise, kernel_size=kernel_size, sigma=sigma)

    # 4. Normalize the flow to the range [-1, 1] for each channel independently.
    min_vals = torch.amin(flow, dim=(-2, -1), keepdim=True)
    max_vals = torch.amax(flow, dim=(-2, -1), keepdim=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    flow_normalized = 2 * (flow - min_vals) / range_vals - 1

    # --- Step 5: Apply anisotropic strength ---
    # Create a strength tensor for broadcasting: [1, 2, 1, 1]
    strength_tensor = torch.tensor([strength_u, strength_v], device=device, dtype=dtype).view(1, 2, 1, 1)
    
    # Scale each channel (u and v) by its respective strength.
    # Broadcasting takes care of the element-wise multiplication.
    flow_scaled = flow_normalized * strength_tensor

    return flow_scaled

def zoom_flow(B: int, H: int, W: int, strength: float = 0.1, 
              device: str = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generates an optical flow field that makes the image expand from the center outwards (Zoom effect).

    Args:
        B (int): The batch size.
        H (int): The height of the image.
        W (int): The width of the image.
        strength (float, optional): The intensity of the expansion.
                                    - A 'positive' value causes expansion (zoom-out).
                                    - A 'negative' value causes contraction (zoom-in).
                                    - The larger the absolute value, the stronger the effect.
                                    Defaults to 0.1.
        device (str, optional): The device to create the tensor on ('cpu' or 'cuda'). Defaults to 'cpu'.
        dtype (torch.dtype, optional): The data type of the tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: The generated optical flow tensor with a shape of [B, 2, H, W].
                      flow[:, 0] corresponds to the horizontal displacement (x-axis).
                      flow[:, 1] corresponds to the vertical displacement (y-axis).
    """
    # 1. Calculate the coordinates of the image center.
    center_y = (H - 1) / 2.0
    center_x = (W - 1) / 2.0

    # 2. Create a grid to get the (x, y) coordinates for each pixel.
    ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # Shape: [H, W]

    # 3. Calculate the vector from the center to each pixel and scale it by strength.
    flow_x = (grid_x - center_x) * strength
    flow_y = (grid_y - center_y) * strength

    # 4. Combine the x and y flows into the final flow tensor.
    # Shape: [2, H, W]
    flow = torch.stack([flow_x, flow_y], dim=0)

    # 5. Expand to the specified batch size.
    # Shape: [1, 2, H, W] -> [B, 2, H, W]
    flow = flow.unsqueeze(0).repeat(B, 1, 1, 1)

    return flow

def rotation_flow(B: int, H: int, W: int, angle_deg: float,
                  device: str = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generates an optical flow field for rotating an image around its center.

    Args:
        B (int): Batch size.
        H (int): Image height.
        W (int): Image width.
        angle_deg (float): The rotation angle in degrees.
                           - Positive values rotate clockwise.
                           - Negative values rotate counter-clockwise.
        device (str, optional): Device for tensor creation. Defaults to 'cpu'.
        dtype (torch.dtype, optional): Data type for the tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: The generated optical flow tensor, shape [B, 2, H, W].
    """
    # 1. Image center
    center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0

    # 2. Create original coordinate grid
    ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    # 3. Convert angle to radians
    theta = math.radians(angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # 4. Apply rotation transformation
    # First, translate to origin, then rotate, then translate back
    x_shifted = grid_x - center_x
    y_shifted = grid_y - center_y
    
    x_new = x_shifted * cos_theta - y_shifted * sin_theta + center_x
    y_new = x_shifted * sin_theta + y_shifted * cos_theta + center_y

    # 5. Calculate flow (new_pos - old_pos)
    flow_x = x_new - grid_x
    flow_y = y_new - grid_y
    
    flow = torch.stack([flow_x, flow_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    return flow

def shear_flow(B: int, H: int, W: int, shear_x: float = 0.0, shear_y: float = 0.0, 
               device: str = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generates an optical flow field for shearing an image relative to its center.

    Args:
        B (int): Batch size.
        H (int): Image height.
        W (int): Image width.
        shear_x (float): Horizontal shear factor. Tilts vertical lines.
        shear_y (float): Vertical shear factor. Tilts horizontal lines.
        device (str, optional): Device for tensor creation. Defaults to 'cpu'.
        dtype (torch.dtype, optional): Data type for the tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: The generated optical flow tensor, shape [B, 2, H, W].
    """
    # 1. Image center
    center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0
    
    # 2. Create original coordinate grid
    ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    # 3. Apply shear transformation relative to the center
    x_new = grid_x + shear_x * (grid_y - center_y)
    y_new = grid_y + shear_y * (grid_x - center_x)

    # 4. Calculate flow (new_pos - old_pos)
    flow_x = x_new - grid_x
    flow_y = y_new - grid_y

    flow = torch.stack([flow_x, flow_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    return flow

def projective_flow(B: int, H: int, W: int, matrix: torch.Tensor, 
                     device: str = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generates an optical flow field from a 3x3 projective transformation matrix.

    Args:
        B (int): Batch size.
        H (int): Image height.
        W (int): Image width.
        matrix (torch.Tensor): The 3x3 projective transformation matrix.
        device (str, optional): Device for tensor creation. Defaults to 'cpu'.
        dtype (torch.dtype, optional): Data type for the tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: The generated optical flow tensor, shape [B, 2, H, W].
    """
    # 1. Ensure matrix is on the correct device and dtype
    M = matrix.to(device=device, dtype=dtype)

    # 2. Create original coordinate grid
    ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    # 3. Create homogeneous coordinates (x, y, 1)
    coords = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0) # Shape: [3, H, W]
    coords = coords.view(3, -1) # Shape: [3, H*W]

    # 4. Apply projective transformation
    # M @ coords -> [x', y', w']
    transformed_coords = M @ coords
    
    # 5. Normalize by the w' component
    w = transformed_coords[2, :]
    x_new = transformed_coords[0, :] / w
    y_new = transformed_coords[1, :] / w

    # Reshape back to image grid
    x_new = x_new.view(H, W)
    y_new = y_new.view(H, W)
    
    # 6. Calculate flow
    flow_x = x_new - grid_x
    flow_y = y_new - grid_y

    flow = torch.stack([flow_x, flow_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    return flow

if __name__ == "__main__":
    # --- Example Usage ---
    from utils.viz_utils import viz_flow_with_rgb, viz_flow_with_arrows
    import os
    
    # Now, use the function to generate the flow
    batch_size = 1
    height = 540
    width = 960

    os.makedirs("outputs/check_flow_generator", exist_ok=True)

    # flow = zoom_flow(B=batch_size, H=height, W=width, strength=0.2, device='cpu')
    
    # 1. generate a random flow
    ## sample 1
    sigma = 20
    strength = 50
    kernel_scale = 3.0
    generator = torch.Generator(device='cpu').manual_seed(1)
    flow = random_flow(B=batch_size, H=height, W=width, sigma=sigma, strength=strength, kernel_scale=kernel_scale, device='cpu', generator=generator)
    flow_rgb = viz_flow_with_rgb(flow.squeeze(0))
    flow_arrows = viz_flow_with_arrows(flow.squeeze(0))
    flow_rgb.save(f"outputs/check_flow_generator/flow1_sigma{sigma}_strength{strength}_kernel_scale{kernel_scale}_rgb.png")
    flow_arrows.save(f"outputs/check_flow_generator/flow1_sigma{sigma}_strength{strength}_kernel_scale{kernel_scale}_arrows.png")

    ## sample 2
    sigma = 50
    strength = 50
    kernel_scale = 3.0
    generator = torch.Generator(device='cpu').manual_seed(1)
    flow = random_flow(B=batch_size, H=height, W=width, sigma=sigma, strength=strength, kernel_scale=kernel_scale, device='cpu', generator=generator)
    flow_rgb = viz_flow_with_rgb(flow.squeeze(0))
    flow_arrows = viz_flow_with_arrows(flow.squeeze(0))
    flow_rgb.save(f"outputs/check_flow_generator/flow2_sigma{sigma}_strength{strength}_kernel_scale{kernel_scale}_rgb.png")
    flow_arrows.save(f"outputs/check_flow_generator/flow2_sigma{sigma}_strength{strength}_kernel_scale{kernel_scale}_arrows.png")
    
    # Print some info to verify
    print("Flow tensor shape:", flow.shape)
    # 
    print(f"Range of u: [{flow[:, 0, :, :].min().item()}, {flow[:, 0, :, :].max().item()}]")
    print(f"Range of v: [{flow[:, 1, :, :].min().item()}, {flow[:, 1, :, :].max().item()}]")
    # Check the flow value near the center (should be close to 0)
    print("Flow at center (approx):", flow[0, :, height // 2, width // 2].numpy())
    # Check the flow value at the top-left corner (should be a large negative vector)
    print("Flow at top-left corner:", flow[0, :, 0, 0].numpy())
    # Check the flow value at the bottom-right corner (should be a large positive vector)
    print("Flow at bottom-right corner:", flow[0, :, -1, -1].numpy())

    # # If you want to create a contracting flow (zoom-in), use a negative strength
    # zoom_in_flow = zoom_flow(B=1, H=1080, W=1920, strength=-0.1)
    # print("\n--- Zoom-In Flow Example ---")
    # print("Zoom-in flow at top-left corner:", zoom_in_flow[0, :, 0, 0].numpy())
    # print("Zoom-in flow at bottom-right corner:", zoom_in_flow[0, :, -1, -1].numpy())