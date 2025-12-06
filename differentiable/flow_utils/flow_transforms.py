import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random

PREFERRED_RESOLUTIONS = [ # (W, H)
    (4096, 2160), # 4K (Standard) - 512: 270 ~= 17:9
    (3840, 2160), # 4K (UHDTV) - 16:9
    (2560, 1440), # 2K - 16:9
    (1920, 1080), # 1080p (Full HD) - 16:9
    (1280, 720), # 720p (HD) - 16:9
    (720, 480), # 480p (SD) ~ 3:2
    (854, 480), # 480p (SD) ~ 16:9
    (640, 480), # 480p (SD) ~ 12:9 = 4:3
    (640, 360), # 360p (Fluent) - 16:9
    (320, 240), # 240p (Fluent) - 12:9 = 4:3
    ]

def resize_flow(flow: torch.Tensor, H: int, W: int, valid_mask: torch.Tensor=None) -> torch.Tensor: 
    if not isinstance(flow, torch.Tensor):
        raise TypeError("Input flow must be a torch.Tensor.")
    dim = flow.dim()
    if dim == 3:
        flow = flow.unsqueeze(0)
    ori_H, ori_W = flow.shape[-2:]
    flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    u_scale = W / ori_W  # u scale factor
    v_scale = H / ori_H  # v scale factor
    scale_factors = torch.tensor([u_scale, v_scale], dtype=torch.float32).to(flow.device)
    flow = flow * scale_factors[None, :, None, None]
    if dim == 3:
        flow = flow.squeeze(0)

    if valid_mask is None:
        return flow
    else:
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("Input valid_mask must be a torch.Tensor.")
        ori_H, ori_W = valid_mask.shape[-2:]
        if ori_H == H and ori_W == W:
            pass
        else:
            if valid_mask.dim() == 3:
                valid_mask = F.interpolate(valid_mask.float().unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
            else:
                valid_mask = F.interpolate(valid_mask.float(), size=(H, W), mode='bilinear', align_corners=False)
            valid_mask = (valid_mask > 0.5).float()
        return flow, valid_mask

class ToTensor:
    def __call__(self, flow: np.ndarray, valid_mask: Image.Image=None):
        if isinstance(flow, torch.Tensor):
            return flow
        if not isinstance(flow, np.ndarray):
            raise TypeError("Input flow must be a 'numpy.ndarray' or 'torch.Tensor' (return Identity).")
        flow = np.transpose(flow, [2, 0, 1])
        flow = torch.from_numpy(flow)
        if valid_mask is None:
            return flow
        else:
            if not isinstance(valid_mask, Image.Image):
                raise TypeError("Input valid_mask must be a 'PIL.Image.Image' (return Identity).")
            valid_mask = np.array(valid_mask)
            valid_mask = np.transpose(valid_mask, [2, 0, 1])
            valid_mask = torch.from_numpy(valid_mask)
            return flow, valid_mask

class Resize:
    def __init__(self, out_H, out_W, rescale=True):
        self.out_H = out_H
        self.out_W = out_W
        self.rescale = rescale

    def __call__(self, flow: torch.Tensor, valid_mask: torch.Tensor=None):
        if not isinstance(flow, torch.Tensor):
            raise TypeError("Input flow must be a torch.Tensor.")
        
        ori_H, ori_W = flow.shape[-2:]
        if ori_H == self.out_H and ori_W == self.out_W:
            pass
        else:
            is_3d = flow.dim() == 3
            if is_3d:
                flow = flow.unsqueeze(0)
            flow = F.interpolate(flow.float(), size=(self.out_H, self.out_W), mode='bilinear', align_corners=False)
            if is_3d:
                flow = flow.squeeze(0)
            if self.rescale:
                u_scale = self.out_W / ori_W
                v_scale = self.out_H / ori_H
                scale_factors = torch.tensor([u_scale, v_scale], dtype=flow.dtype, device=flow.device)
                flow.mul_(scale_factors[None, :, None, None])
        
        if valid_mask is None:
            return flow
        
        else:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError("Input valid_mask must be a torch.Tensor.")
            ori_H, ori_W = valid_mask.shape[-2:]
            
            if ori_H == self.out_H and ori_W == self.out_W:
                pass
            else:
                is_3d = valid_mask.dim() == 3
                if is_3d:
                    valid_mask = valid_mask.unsqueeze(0)
                valid_mask = F.interpolate(valid_mask.float(), size=(self.out_H, self.out_W), mode='bilinear', align_corners=False)
                if is_3d:
                    valid_mask = (valid_mask.squeeze(0) > 0.5).float()
            return flow, valid_mask
    
class Normalize:
    def __init__(self, out_H, out_W):
        self.out_H = out_H
        self.out_W = out_W
        self.nor_factor = torch.tensor([self.out_W, self.out_H])
        
    def __call__(self, flow: torch.Tensor, valid_mask: torch.Tensor=None):
        # flow: [2, H, W]
        if not isinstance(flow, torch.Tensor):
            raise TypeError("Input flow must be a torch.Tensor.")
        if valid_mask is None:
            return flow / self.nor_factor.to(flow.device)[:, None, None]
        else:
            return flow / self.nor_factor.to(flow.device)[:, None, None], valid_mask


class Denormalize:
    def __init__(self, out_H, out_W):
        self.nor_factor = torch.tensor([self.out_W, self.out_H])

    def __call__(self, flow: torch.Tensor, valid_mask: torch.Tensor=None):
        # flow: [2, H, W]
        if not isinstance(flow, torch.Tensor):
            raise TypeError("Input flow must be a torch.Tensor.")
        if valid_mask is None:
            return flow * self.nor_factor.to(flow.device)[:, None, None]
        else:
            return flow * self.nor_factor.to(flow.device)[:, None, None], valid_mask

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, flow: torch.Tensor, valid_mask: torch.Tensor=None):
        if random.random() < self.p:
            flow = torch.flip(flow, dims=[-1]) # [2, H, W], flip along W
            if valid_mask is not None:
                valid_mask = torch.flip(valid_mask, dims=[-1]) # [2, H, W], flip along W
        if valid_mask is None:
            return flow
        else:
            return flow, valid_mask
            
class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, flow: torch.Tensor, valid_mask: torch.Tensor=None):
        if random.random() < self.p:
            flow = torch.flip(flow, dims=[-2]) # [2, H, W], flip along H
            if valid_mask is not None:
                valid_mask = torch.flip(valid_mask, dims=[-2]) # [2, H, W], flip along H
        if valid_mask is None:
            return flow
        else:
            return flow, valid_mask

class RamdomSignFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, flow: torch.Tensor, valid_mask: torch.Tensor=None):
        if not isinstance(flow, torch.Tensor):
            raise TypeError("Input flow must be a torch.Tensor.")
        if random.random() < self.p:
            flow.neg_() # sign flip
        if valid_mask is None:
            return flow
        else:
            return flow, valid_mask
                