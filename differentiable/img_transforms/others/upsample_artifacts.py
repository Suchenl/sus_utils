import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
import math

class UpsampleArtifacts(torch.nn.Module):
    """
    Introduces a variety of digital artifacts by upsampling and then downsampling an image.
    This process is highly effective at generating Moiré patterns on images with
    high-frequency, repeating textures (e.g., fabrics, grids).

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 scale_factor_range: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]] = None, 
                 scale_factor_level: str = None,
                 upsample_modes: List[str] = ['bilinear', 'bicubic', 'nearest'],
                 downsample_modes: List[str] = ['area', 'bilinear', 'bicubic', 'nearest'],
                 per_image_randomization: bool = False,
                 randomize_alignment: bool = True):
        super(UpsampleArtifacts, self).__init__()
        # ... (Initialization logic remains the same as yours, it's already excellent)
        self.per_image_randomization = per_image_randomization
        self.upsample_modes = upsample_modes
        self.downsample_modes = downsample_modes
        self.randomize_alignment = randomize_alignment

        factor_level_map = {
            "subtle": ((1.05, 2.0), (1.05, 2.0)), "moderate": ((2.0, 4.0), (2.0, 4.0)),
            "subtle and moderate": ((1.05, 4.0), (1.05, 4.0)), "strong": ((4.0, 8.0), (4.0, 8.0)),
            "subtle to strong": ((1.05, 8.0), (1.05, 8.0))
        }

        if scale_factor_range is None and scale_factor_level is None:
            scale_factor_level = "subtle to strong"
        
        if scale_factor_range is not None:
            self.scale_factor_range = scale_factor_range
        else:
            self.scale_factor_range = factor_level_map[scale_factor_level]
        
        if isinstance(self.scale_factor_range[0], (float, int)):
            self.range_h, self.range_w = self.scale_factor_range, self.scale_factor_range
        else:
            self.range_h, self.range_w = self.scale_factor_range[0], self.scale_factor_range[1]
        
        # ... (Print statements are also fine as they are)

    def _get_random_interp_params(self, mode_list: List[str], generator: Optional[torch.Generator] = None) -> Tuple[str, Optional[bool]]:
        ALIGN_MODES = {'bilinear', 'bicubic'}
        # Choose a random interpolation mode
        mode_idx = torch.randint(0, len(mode_list), (1,), generator=generator, device=generator.device if generator is not None else 'cpu').item()
        interp_mode = mode_list[mode_idx]
        
        align_corners = None
        if interp_mode in ALIGN_MODES:
            # Choose a random boolean for align_corners
            align_corners = torch.rand(1, generator=generator, device=generator.device if generator is not None else 'cpu').item() < 0.5 if self.randomize_alignment else False
        return interp_mode, align_corners

    def _get_random_params(self, generator: Optional[torch.Generator] = None) -> Tuple[float, float, Tuple[str, Optional[bool]], Tuple[str, Optional[bool]]]:
        # Generate random uniform values for scale factors
        rand_h_w = torch.rand(2, generator=generator, device=generator.device if generator is not None else 'cpu')
        rand_h, rand_w = rand_h_w[0].item(), rand_h_w[1].item()
        
        scale_h = self.range_h[0] + rand_h * (self.range_h[1] - self.range_h[0])
        scale_w = self.range_w[0] + rand_w * (self.range_w[1] - self.range_w[0])
        
        # Get random interpolation parameters using the generator
        up_params = self._get_random_interp_params(self.upsample_modes, generator=generator)
        down_params = self._get_random_interp_params(self.downsample_modes, generator=generator)
        return scale_h, scale_w, up_params, down_params

    def forward(self, image: torch.Tensor, 
                scale_factor: Union[float, Tuple[float, float]] = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        is_neg_one_range = image.min() < -0.01
        if scale_factor is not None:
            # Deterministic path
            scale_h, scale_w = (scale_factor, scale_factor) if isinstance(scale_factor, float) else scale_factor
            up_params = (self.upsample_modes[0], False if self.upsample_modes[0] in {'bilinear', 'bicubic'} else None)
            down_params = (self.downsample_modes[0], False if self.downsample_modes[0] in {'bilinear', 'bicubic'} else None)
            output = self._apply_resample(image, scale_h, scale_w, up_params, down_params)
        elif not self.per_image_randomization:
            # "Batch-Uniform" random path
            params = self._get_random_params(generator=generator)
            output = self._apply_resample(image, *params)
        else:
            # "Per-Image" random path
            output = self._apply_resample_per_image(image, generator=generator)
        return torch.clamp(output, -1.0, 1.0) if is_neg_one_range else torch.clamp(output, 0.0, 1.0)

    def _apply_resample(self, image: torch.Tensor, scale_h: float, scale_w: float, 
                        up_params: Tuple[str, Optional[bool]], 
                        down_params: Tuple[str, Optional[bool]]) -> torch.Tensor:
        if scale_h <= 1.0 and scale_w <= 1.0: return image
        _, _, h, w = image.shape
        
        up_mode, up_align = up_params
        down_mode, down_align = down_params
        
        # More robust: Manually calculate target size
        target_h, target_w = int(h * scale_h), int(w * scale_w)
        
        # Avoid upsampling to a size that's smaller than the original
        if target_h <= h and target_w <= w: return image
        
        upsampled = F.interpolate(image, size=(target_h, target_w), mode=up_mode, align_corners=up_align)
        recovered = F.interpolate(upsampled, size=(h, w), mode=down_mode, align_corners=down_align)
        
        return recovered
    
    def _apply_resample_per_image(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        output_images = [
            self._apply_resample(img.unsqueeze(0), *self._get_random_params(generator=generator))
            for img in image
        ]
        return torch.cat(output_images, dim=0)