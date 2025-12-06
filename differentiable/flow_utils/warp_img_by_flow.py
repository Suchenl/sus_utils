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
# Explanation:
    ## This file contains two functions for warping images by optical flow.
    1. The function `backward_warp` which warps an image by a backward flow.
        It uses `torch.nn.functional.grid_sample` to perform the warping.
    2. The function `forward_warp` which warps an image by a forward flow.
        It uses a custom implementation of the splatting algorithm to perform the warping.
        
# Run this file to test the functions.
"""
import torch
import torch.nn.functional as F
from typing import *

def backward_warp(img: torch.Tensor,
                  flow: torch.Tensor,
                  flow_is_normalized: bool = False,
                  align_corners: bool = False,
                  sample_mode: Literal['bilinear', 'nearest'] = 'bilinear',
                  padding_mode: Literal['border', 'zeros', 'reflection'] = 'zeros',
                  dtype=torch.float32):
    """
    'backward' warping via 'grid_sample' (inverse mapping / sampling).
    * Expectation: `flow` maps TARGET -> SOURCE (i.e., backward flow),
        so for each target pixel p_t we sample source at p_s = p_t + flow(p_t).
    Args:
        img: [B, C, H, W], source image (values to be sampled / splatted)
        flow: [B, 2, H, W], flow in pixel units (or normalized if flow_is_normalized=True)
              flow[:,0] is u (horizontal-x direction), flow[:,1] is v (vertical-y direction).
        flow_is_normalized: if True, flow is assumed normalized: flow_x in [-1,1] representing fraction of W,
                            flow_y in [-1,1] representing fraction of H. We'll denormalize by *W and *H.
        align_corners: passed to grid_sample for 'backward' branch; used to convert pixel coords->[-1,1].
        sample_mode: 'bilinear' or 'nearest' sampling mode.
        padding_mode: for backward (grid_sample) supports 'border'|'zeros'|'reflection'.
        dtype: output dtype (defaults to img.dtype)
    Returns:
        img_warped: [B, C, H, W]
        valid_mask: [B, 1, H, W]  (1 where any contribution exists / sampled coords in bounds, 
                                    0 where added content)
    """
    if dtype is None:
        dtype = img.dtype
    B, C, H, W = img.shape
    device = img.device
    assert flow.shape == (B, 2, H, W)

    # If flow is normalized, then denormalize it before adding to base_grid
    # assume flow_x in [-1,1] corresponds to fraction of W and flow_y in [-1,1] corresponds to fraction of H.
    if flow_is_normalized:
        flow = flow.clone()
        flow_x = flow[:, 0, :, :] * W    # normalized -> pixel units
        flow_y = flow[:, 1, :, :] * H
        flow = torch.stack([flow_x, flow_y], dim=1)
        
    # base grid of pixel centers: x in [0..W-1], y in [0..H-1]
    ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base_grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
    
    # compute source sampling positions p_s in pixel coords
    p = base_grid + flow

    # normalize pixel coords to [-1, 1] for grid_sample
    p_x, p_y = p[:, 0], p[:, 1] # [B, H, W]
    if align_corners:
        # If True, the range [-1, 1] corresponds to the pixel centers [0, W-1] or [0, H-1].
        # If set to True, -1 and 1 are considered the center of the corner pixels, rather than the corners of the image.
        p_xn = (p_x / (W - 1)) * 2 - 1
        p_yn = (p_y / (H - 1)) * 2 - 1
    else:
        # If False, the range [-1, 1] corresponds to the pixel edges [-0.5, (W-1)+0.5] or [-0.5, (H-1)+0.5] = 
        # [-0.5, W-0.5] or [-0.5, H-0.5].
        p_xn = ((p_x + 0.5) / W) * 2 - 1
        p_yn = ((p_y + 0.5) / H) * 2 - 1
        
    sample_grid = torch.stack((p_xn, p_yn), dim=-1)  # [B, H, W, 2], format required by grid_sample
    # Grid sampling: sample source image at target positions p_s in pixel coords
    img_warped = F.grid_sample(img.to(dtype), sample_grid, mode=sample_mode, padding_mode=padding_mode, align_corners=align_corners)

    # valid mask: whether sampled source coordinate falls inside image bounds (in pixel coords)
    # if the sampled source coordinate is outside the image, the target pixel is 'added content' belong to forgery content.
    # it is say that the ~valid_mask is the mask of the 'Added Forgery Content'.
    mask_x = (p_x >= 0) & (p_x <= W - 1)
    mask_y = (p_y >= 0) & (p_y <= H - 1)
    valid_mask = (mask_x & mask_y).unsqueeze(1)  # [B, 1, H, W]，value in {0, 1}
    return img_warped, valid_mask
    
def forward_warp(img: torch.Tensor,
                 flow: torch.Tensor,
                 flow_is_normalized: bool = False,
                 padding_mode: Literal['border', 'zeros'] = 'zeros',
                 dtype=torch.float32,
                 eps: float = 1e-6,
                 threshold: float = 0.25):
    """
    Forward warping (splatting): source -> target

    * Expectation:
        - flow maps SOURCE -> TARGET.
        - For each source pixel p_s, it contributes to target position p_t = p_s + flow(p_s).
    * Implementation:
        - Supports "nearest" and "bilinear" splatting.
        - If multiple source pixels map to the same target location, contributions are accumulated.
        - Final pixel values are normalized by total weight (i.e., average splatting).
    * Notes:
      - This implementation distributes each source pixel to its 4 bilinear neighbors
        using continuous bilinear weights. The distribution is differentiable w.r.t.
        the input image and (partially) the flow (gradients flow through weights).
    Args:
        img: [B, C, H, W], source image (values to be splatted).
        flow: [B, 2, H, W], flow field in pixel units
              (or normalized if flow_is_normalized=True).
              flow[:,0] = u (horizontal-x), flow[:,1] = v (vertical-y).
        flow_is_normalized: if True, flow is assumed normalized in [-1,1] w.r.t. width/height.
        align_corners: unused here (kept for symmetry with backward warping).
        padding_mode: 'border' (clamp out-of-bound coords) or 'zeros' (discard OOB). (OOB means Out Of Boundary)
        dtype: output dtype.
        eps: small constant to avoid division by zero.
        threshold: threshold for content mask (see below).
    Returns:
        img_warped: [B, C, H, W], warped target image.
        valid_mask: [B, 1, H, W], binary mask (1 if any contribution exists).
    """
    if dtype is None:
        dtype = img.dtype
    B, C, H, W = img.shape
    device = img.device
    if flow.shape != (B, 2, H, W):
        print(f"flow must be {(B, 2, H, W)}, but got {tuple(flow.shape)}")
        raise AssertionError

    # If flow is normalized, then denormalize it before adding to base_grid
    # assume flow_x in [-1,1] corresponds to fraction of W and flow_y in [-1,1] corresponds to fraction of H.
    if flow_is_normalized:
        flow = flow.clone()
        flow_x = flow[:, 0, :, :] * W    # normalized -> pixel units
        flow_y = flow[:, 1, :, :] * H
        flow = torch.stack([flow_x, flow_y], dim=1)

    # base grid of pixel centers: x in [0..W-1], y in [0..H-1]
    ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base_grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

    # Compute target positions (float) where each source pixel moves to:
    tgt = base_grid + flow  # p_t = p_s + flow(p_s) # [B, 2, H, W]
    tgt_x = tgt[:, 0]  # [B, H, W]
    tgt_y = tgt[:, 1]  # [B, H, W]

    # For bilinear soft-splat: 4 neighbor integer corners
    x0 = torch.floor(tgt_x).long()  # B x H x W
    y0 = torch.floor(tgt_y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    # Compute bilinear weights (continuous expressions, differentiable wrt tgt_x/tgt_y)
    # weight = (x1 - x) * (y1 - y) 
    wa = (x1.to(dtype) - tgt_x) * (y1.to(dtype) - tgt_y)  # top-left:       (1-dx) * (1-dy)
    wb = (tgt_x - x0.to(dtype)) * (y1.to(dtype) - tgt_y)  # top-right:      (dx) * (1-dy)
    wc = (x1.to(dtype) - tgt_x) * (tgt_y - y0.to(dtype))  # bottom-left:    (1-dx) * (dy)
    wd = (tgt_x - x0.to(dtype)) * (tgt_y - y0.to(dtype))  # bottom-right:   (dx) * (dy)

    # Flatten source image to [B, C, N]
    #   - Each of the N source pixels is treated as a particle.
    #   - This makes scatter/splat operations easier.
    img_flat = img.reshape(B, C, -1)  # [B, C, N]
    N = H * W
    warped = torch.zeros((B, C, H, W), device=device, dtype=dtype)
    weight_sum = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

    # helper: scatter-add weighted contributions for one corner
    def scatter_corner(x_idx, y_idx, weight):
        """
        x_idx, y_idx: long tensors shape [B, H, W]
        weight: float tensor shape [B, H, W]
        Add img * weight to warped at integer locations (x_idx, y_idx).
        """
        # compute boolean mask of indices that are inside image bounds BEFORE clamping
        in_bounds = (x_idx >= 0) & (x_idx <= (W - 1)) & (y_idx >= 0) & (y_idx < (H - 1))  # [B, H, W]

        if padding_mode == 'border':
            # clamp to border; treat everything as valid (we will map OOB to border)
            x_safe = torch.clamp(x_idx, 0, W - 1)
            y_safe = torch.clamp(y_idx, 0, H - 1)
            mask = torch.ones_like(in_bounds, dtype=torch.bool, device=device)
        else:
            # zeros: drop out-of-bounds contributions
            x_safe = x_idx
            y_safe = y_idx
            mask = in_bounds

        # flatten safe indices to linear index for scatter
        safe_idx = (y_safe * W + x_safe).reshape(B, -1)  # [B, N]

        # for each batch, scatter-add the weighted source pixels to warped / weight_sum
        # Note: we only add entries where mask is True. Use view/reshape to access flattened source pixels.
        for b in range(B):
            mb = mask[b].reshape(-1)                     # (N,)
            if not mb.any():
                continue
            safe_idx_b = safe_idx[b, mb].long()                   # (K,)
            vals = img_flat[b, :, mb] * weight[b].reshape(1, -1)[0, mb].to(dtype)  # careful shape
            # expand safe idx for channels
            # accumulate pixel values
            warped[b].view(C, -1).index_add_(1, safe_idx_b, vals)
            # accumulate weights
            weight_sum[b, 0].view(-1).index_add_(0, safe_idx_b, weight[b].reshape(-1)[mb])

    # Scatter for the 4 neighbors using the computed weights
    scatter_corner(x0, y0, wa)  # top-left
    scatter_corner(x1, y0, wb)  # top-right
    scatter_corner(x0, y1, wc)  # bottom-left
    scatter_corner(x1, y1, wd)  # bottom-right 
    
    # Normalize accumulated values by weights (avoid division by zero)
    #    Use where to avoid NaN; positions with zero weight remain zero.
    denom = weight_sum + eps
    warped_normed = warped / denom

    # Build the masks
    #    - soft mask: weight_sum (float) — how much total contribution each target pixel received
    #    - hard binary mask: content_mask = weight_sum >= threshold
    content_mask = get_content_mask(weight_sum, threshold) # [B, 1, H, W]

    # 10) OPTIONAL: geometric mask indicating whether the *float target position* p_t
    #     fell inside the image boundaries (useful when you want to treat clamped border
    #     contributions as invalid)
    valid_mask = (tgt_x >= 0.0) & (tgt_x <= (W - 1)) & (tgt_y >= 0.0) & (tgt_y <= (H - 1))
    valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]

    # Return normalized warped image, soft weights and hard mask
    # - 'valid_mask' is corresponding to optical flow (-FULL NAME: 'valid optical flow mask')
    # - 'content_mask' is corresponding to retained image content in warped image after motion process (-FULL NAME: 'retained image content mask')
    #   - This mask is special in forward warp. In backward warp, the content_mask is same as valid_mask.
    return warped_normed, valid_mask, weight_sum, content_mask

def get_content_mask(weight_sum, threshold=0.25):
    """
    Get content mask from weight sum.
    - The reason I set the default threshold to >= 0.25 is: 
        if a pixel is assigned to the middle of four pixels, 
        then the threshold of the four points is 0.25. 
        If it is biased towards one pixel or two pixels, it or they will be retained. 
        If it is not biased towards one, all of pixels arount it will be retained.
    """
    # Eliminate sparse points to get more stable values
    w_smooth = F.avg_pool2d(weight_sum, kernel_size=3, stride=1, padding=1) 
    return w_smooth >= threshold    # Binarize the w_smooth
    
if __name__ == "__main__":
    from PIL import Image
    import torch
    from torchvision import transforms
    import numpy as np
    from utils.func_decorators.time_decorator import timer_decorator
    from .flow_generator import zoom_flow, rotation_flow, shear_flow, perspective_flow
    from utils.viz_utils.flow_viz import viz_flow_with_rgb, viz_flow_with_arrows
    from .transform_matrix_generator import TRANSFORMATION_MATRICES, get_perspective_view_matrix
    M_top_view, M_top_view_inv = get_perspective_view_matrix(H=1080, W=1920, strength=0.5, eye_pos='bottom')
    
    # Change to your own path
    save_dir = '/public/chenyuzhuo/MODELS/image_watermarking_models/Image_Motion_Pred-dev/outputs/check_flow_utils'
    img_dir = '/public/chenyuzhuo/MODELS/image_watermarking_models/Image_Motion_Pred-dev/outputs/check_flow_utils/frame_left_0002.png'

    img = Image.open()
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    # flow = torch.zeros([1, 2, 1080, 1920])
    # flow[:, 0] += 500
    # flow[:, 1] += 250
    # flow = zoom_flow(B=1, H=1080, W=1920, strength=-0.5, device=img_tensor.device)
    # flow = rotation_flow(B=1, H=1080, W=1920, angle_deg=-45, device=img_tensor.device)
    # flow = shear_flow(B=1, H=1080, W=1920, shear_x=-0, shear_y=-0.1, device=img_tensor.device)
    flow = perspective_flow(B=1, H=1080, W=1920, matrix=M_top_view_inv, device=img_tensor.device)
        
    # backward warp
    warped_image, valid_mask = timer_decorator(backward_warp)(img_tensor, flow)
    backward_warped_image = warped_image

    # save warped image and valid mask
    warped_image = warped_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray((warped_image * 255).astype(np.uint8)).save(save_dir + '/backward_warped_image.png')
    valid_mask = valid_mask.squeeze(0, 1).detach().cpu().numpy()
    Image.fromarray((valid_mask * 255).astype(np.uint8)).convert("L").save(save_dir + '/backward_valid_mask.png')
    print(warped_image.shape, valid_mask.shape)
    # save flow
    flow_rgb_img_pil = viz_flow_with_rgb(flow[0])
    flow_arrow_img_pil = viz_flow_with_arrows(flow[0])
    flow_rgb_img_pil.save(save_dir + '/backward_flow_rgb.png')
    flow_arrow_img_pil.save(save_dir + '/backward_flow_arrow.png')
    
    # forward warp
    # flow = flow * -1
    # flow = rotation_flow(B=1, H=1080, W=1920, angle_deg=45, device=img_tensor.device)
    # flow = shear_flow(B=1, H=1080, W=1920, shear_x=0, shear_y=0.1, device=img_tensor.device)
    flow = perspective_flow(B=1, H=1080, W=1920, matrix=M_top_view, device=img_tensor.device)

    warped_image, valid_mask, weight_sum, content_mask = timer_decorator(forward_warp)(img_tensor, flow)
    forward_warped_image = warped_image

    # save warped image and valid mask
    warped_image = warped_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray((warped_image * 255).astype(np.uint8)).save(save_dir + '/forward_warped_image.png')
    valid_mask = valid_mask.squeeze(0, 1).detach().cpu().numpy()
    Image.fromarray((valid_mask * 255).astype(np.uint8)).convert("L").save(save_dir + '/forward_valid_mask.png')
    weight_sum = weight_sum.squeeze(0, 1).detach().cpu().numpy()
    Image.fromarray((weight_sum * 255).astype(np.uint8)).convert("L").save(save_dir + '/forward_weight_sum.png')
    content_mask = content_mask.squeeze(0, 1).detach().cpu().numpy()
    Image.fromarray((content_mask * 255).astype(np.uint8)).convert("L").save(save_dir + '/forward_content_mask.png')
    print(warped_image.shape, valid_mask.shape, weight_sum.shape, content_mask.shape)
    # save flow
    flow_rgb_img_pil = viz_flow_with_rgb(flow[0])
    flow_arrow_img_pil = viz_flow_with_arrows(flow[0])
    flow_rgb_img_pil.save(save_dir + '/forward_flow_rgb.png')
    flow_arrow_img_pil.save(save_dir + '/forward_flow_arrow.png')
    print("MAE between backward_warped and forward_warped:", (backward_warped_image - forward_warped_image).abs().mean())