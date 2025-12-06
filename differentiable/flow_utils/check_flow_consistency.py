from .warp_img_by_flow import backward_warp
import torch

def check_flow_consistency(flow_fw: torch.Tensor, 
                           flow_bw: torch.Tensor,
                           return_error: bool = False,
                           alpha1: float = 0.01,
                           alpha2: float = 0.5):
    """
    Optical flow consistency check. (forward check)

    - The forward check corresponds to the 'source' image coordinate, used to detect disappearing regions (occlusions).
    - The backward check corresponds to the 'target' image coordinate, used to detect newly appearing regions (disocclusions).

    Args:
        flow_fw: (B, 2, H, W) forward flow (t -> t+1)
        flow_bw: (B, 2, H, W) backward flow (t+1 -> t)
    Returns:
        occ_mask: (B, 1, H, W) occlusion mask
        error: (B, 1, H, W) consistency error if 'return_error' is True
    """
    _, _, H, W = flow_fw.shape
    # backward warp the backward flow into forward domain
    flow_bw_bwed, _ = backward_warp(img=flow_bw, flow=flow_fw, padding_mode="zeros")
    
    # compute forward–backward error
    fb_sum = flow_fw + flow_bw_bwed
    error = torch.norm(fb_sum, dim=1, keepdim=True) # (B, 1, H, W)

    # compute flow magnitudes
    mag_fw = torch.norm(flow_fw, dim=1, keepdim=True)
    mag_bw_bwed = torch.norm(flow_bw_bwed, dim=1, keepdim=True)

    # RAFT/UnFlow/ARFlow-style occlusion criterion
    occ_mask = (error ** 2 > alpha1 * (mag_fw ** 2 + mag_bw_bwed ** 2) + alpha2).bool()

    if return_error:
        return occ_mask, error
    return occ_mask  # (B, 1, H, W)