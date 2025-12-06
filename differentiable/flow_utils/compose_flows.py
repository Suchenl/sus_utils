import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Literal
from .warp_img_by_flow import forward_warp, backward_warp

def compose_forward_flow_sequence(flows: list, 
                                  valids: list = None, 
                                  mode: Literal['sample', 'scatter'] = 'sample', 
                                  align_corners: bool = False,
                                  dtype=torch.float32):
    """
    Compose a sequence of forward optical flows: f_1->2, f_2->3, ..., f_(n-1)->n.
    The result is the direct flow f_1->n.

    Args:
        flows: list of length (N-1), each tensor shape (B,2,H,W).
               Each flow is forward: pixel (x,y) in frame k moves by flow[k] to frame (k+1).
        align_corners: bool, passed to grid_sample.
               Must be consistent with how you use grid_sample in later warping/resizing.
               
    Returns:
        flow_1n: (B,2,H,W) final forward flow from frame1 -> framen
        valid_mask: (B,1,H,W) binary mask of valid pixels after composition
    """
    if dtype is not None:
        flows = [f.to(dtype) for f in flows]
        if valids is not None:
            valids = [v.to(dtype) for v in valids]
    B, C, H, W = flows[0].shape
    device, dtype = flows[0].device, flows[0].dtype
    assert all(f.shape == (B,2,H,W) for f in flows)

    # Build a base pixel grid of coordinates in frame1
    # base_grid[b,:,y,x] = (x,y) pixel location
    ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base_grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(B,1,1,1)  # (B,2,H,W)

    # Initialize composed flow as zeros (f_1->1)
    flow_total = torch.zeros_like(flows[0])
    # Initialize validity mask as all ones
    valid_total = torch.ones((B,1,H,W), device=device, dtype=dtype)

    # Iterate through the flow sequence
    num_flows = len(flows)
    for i in range(num_flows):  # f is f_k->k+1
        f = flows[i]
        if torch.isnan(f).any() or torch.isinf(f).any():
            print(f"Warning: Input flow at index {i} contains NaN or Inf!")
        valid = valids[i] if valids is not None else None

        if mode == 'sample':
            sampled_f, valid_mask = backward_warp(img=f, flow=flow_total, align_corners=align_corners)
            # Update composed flow: f_1->(k+1)(x) = f_1->k(x) + f_k->(k+1)( x + f_1->k(x) )
            flow_total = flow_total + sampled_f

        # This implemention is wrong, but I'm leaving it here for now
        # elif mode == 'scatter':
        #     warped_flow, valid_mask, _, content_mask = forward_warp(img=flow_total, flow=f)
        #     flow_total = warped_flow + f

        # Update valid mask
        valid_total = valid_total * valid_mask
        if valid is not None:
            valid_total = valid_total * valid

    return flow_total, valid_total


def group_compose_forward_flow_sequence(
    flows: List[torch.Tensor], 
    valids: Optional[List[torch.Tensor]] = None, 
    align_corners: bool = False, 
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Composes a sequence of forward optical flows using a hierarchical grouping (divide and conquer)
    strategy to minimize the chain of interpolation steps and reduce cumulative error.

    This is significantly more accurate for long sequences than a simple iterative method.

    Args:
        flows: List of (N-1) forward optical flows, each tensor shape (B, 2, H, W).
        valids: Optional list of corresponding valid masks, each tensor shape (B, 1, H, W).
        align_corners: Bool, passed to grid_sample.
        dtype: The desired data type for computation.

    Returns:
        flow_1n: The final composed forward flow from frame 1 to frame n.
        valid_mask: The final composed valid mask.
    """
    # --- 1. Initial Setup and Checks ---
    if not flows:
        raise ValueError("Input 'flows' list cannot be empty.")

    if dtype is not None:
        flows = [f.to(dtype) for f in flows]
        if valids is not None:
            valids = [v.to(dtype) for v in valids]

    B, C, H, W = flows[0].shape
    device = flows[0].device
    
    # Create the base coordinate grid once and pass it through the recursion
    ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base_grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

    # --- 2. Start the Recursive Composition ---
    return _recursive_compose(flows, valids, base_grid, align_corners)


def _recursive_compose(
    sub_flows: List[torch.Tensor],
    sub_valids: Optional[List[torch.Tensor]],
    base_grid: torch.Tensor,
    align_corners: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recursive helper function to compose a sub-sequence of flows."""
    n = len(sub_flows)
    B, _, H, W = base_grid.shape
    device, dtype = base_grid.device, base_grid.dtype

    # --- Base Cases for the Recursion ---
    if n == 0:
        # No flows, so it's a zero flow (identity transform) and all pixels are valid.
        return torch.zeros(B, 2, H, W, device=device, dtype=dtype), \
               torch.ones(B, 1, H, W, device=device, dtype=dtype)
    if n == 1:
        # A single flow is its own composition.
        valid_mask = sub_valids[0].clone() if sub_valids is not None else \
                     torch.ones(B, 1, H, W, device=device, dtype=dtype)
        return sub_flows[0].clone(), valid_mask

    # --- Recursive (Divide and Conquer) Step ---
    mid = n // 2
    
    # Recursively compose the first and second halves of the sequence
    flows_first_half, valids_first_half = _recursive_compose(
        sub_flows[:mid], sub_valids[:mid] if sub_valids is not None else None, base_grid, align_corners
    )
    flows_second_half, valids_second_half = _recursive_compose(
        sub_flows[mid:], sub_valids[mid:] if sub_valids is not None else None, base_grid, align_corners
    )

    # --- Combination Step ---
    # Combine the results of the two halves: f_total = f_first + f_second(p_first)
    
    # 1. Calculate landing positions after the first half's flow
    p = base_grid + flows_first_half
    p_x, p_y = p[:, 0], p[:, 1]

    # 2. Normalize coordinates for grid_sample
    if align_corners:
        p_xn = (p_x / (W - 1)) * 2 - 1
        p_yn = (p_y / (H - 1)) * 2 - 1
    else:
        p_xn = ((p_x + 0.5) / W) * 2 - 1
        p_yn = ((p_y + 0.5) / H) * 2 - 1
    sample_grid = torch.stack((p_xn, p_yn), dim=-1)

    # 3. Sample the second half's flow and valid mask at these positions
    sampled_flow_second = F.grid_sample(
        flows_second_half, sample_grid, mode='bilinear',
        padding_mode='zeros', align_corners=align_corners
    )
    sampled_valid_second = F.grid_sample(
        valids_second_half, sample_grid, mode='nearest', # Use 'nearest' for masks
        padding_mode='zeros', align_corners=align_corners
    )

    # 4. Compose the final flow for this level
    final_flow = flows_first_half + sampled_flow_second

    # 5. Compose the final valid mask
    inside_mask = ((p_x >= 0) & (p_x <= W - 1) & (p_y >= 0) & (p_y <= H - 1)).float().unsqueeze(1)
    final_valid = valids_first_half * sampled_valid_second * inside_mask
    
    return final_flow, final_valid