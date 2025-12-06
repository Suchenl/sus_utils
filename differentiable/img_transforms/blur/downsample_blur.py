import torch
import torch.nn.functional as F
from typing import *

class DownsampleBlur(torch.nn.Module):
    """
    Applies a highly randomized blur by non-uniformly downsampling and then upsampling.
    This ultimate version randomizes scale factors, and independently randomizes the 
    interpolation modes and corner alignment for both the downsampling and upsampling steps.
    """
    def __init__(self, 
                 scale_factor_range: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]] = None, 
                 scale_factor_level: str = None,
                 upsample_modes: List[str] = ['bilinear', 'bicubic', 'nearest'], # 'area' is not used for upsampling. 'nearest' simulates low-quality scaling
                 downsample_modes: List[str] = ['bilinear', 'bicubic', 'nearest', 'area'], # 'area' is often best for anti-aliasing during downsampling
                 per_image_randomization: bool = False,  # In this implementation, 'False' is faster but less random.
                 randomize_alignment: bool = True):
        super(DownsampleBlur, self).__init__()
        self.per_image_randomization = per_image_randomization
        self.downsample_modes = downsample_modes
        self.upsample_modes = upsample_modes
        self.randomize_alignment = randomize_alignment

        factor_level_map = {
            "subtle": ((0.7, 0.95), (0.7, 0.95)), "moderate": ((0.4, 0.7), (0.4, 0.7)),
            "subtle and moderate": ((0.4, 0.95), (0.4, 0.95)), "strong": ((0.15, 0.4), (0.15, 0.4))
        }
        if scale_factor_range is None and scale_factor_level is None:
            scale_factor_level = "subtle and moderate"
        
        if scale_factor_range is not None:
            self.scale_factor_range = scale_factor_range
        else:
            self.scale_factor_range = factor_level_map[scale_factor_level]
        
        if isinstance(self.scale_factor_range[0], float):
            self.range_h, self.range_w = self.scale_factor_range, self.scale_factor_range
        else:
            self.range_h, self.range_w = self.scale_factor_range[0], self.scale_factor_range[1]

        mode = "Per-Image" if self.per_image_randomization else "Batch-Uniform"
        print(f"[DownsampleBlur] H-Scale: {self.range_h}, W-Scale: {self.range_w}, Align Random: {self.randomize_alignment}, Mode: {mode}")

    def _get_random_interp_params(self, mode_list: List[str], generator: Optional[torch.Generator] = None) -> Tuple[str, Optional[bool]]:
        # Choose a random interpolation mode
        mode_idx = torch.randint(0, len(mode_list), (1,), generator=generator, device=generator.device if generator is not None else 'cpu').item()
        interp_mode = mode_list[mode_idx]
        
        align_corners = None
        if interp_mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            if self.randomize_alignment:
                # Choose a random boolean for align_corners
                align_corners = torch.rand(1, generator=generator, device=generator.device if generator is not None else 'cpu').item() < 0.5
            else:
                align_corners = False
        return interp_mode, align_corners

    def _get_random_params(self, generator: Optional[torch.Generator] = None) -> Tuple[float, float, Tuple[str, Optional[bool]], Tuple[str, Optional[bool]]]:
        # Generate random uniform values for scale factors
        rand_h_w = torch.rand(2, generator=generator, device=generator.device if generator is not None else 'cpu')
        rand_h, rand_w = rand_h_w[0].item(), rand_h_w[1].item()
        
        scale_h = self.range_h[0] + rand_h * (self.range_h[1] - self.range_h[0])
        scale_w = self.range_w[0] + rand_w * (self.range_w[1] - self.range_w[0])
        
        # Get random interpolation parameters using the generator
        down_params = self._get_random_interp_params(self.downsample_modes, generator=generator)
        up_params = self._get_random_interp_params(self.upsample_modes, generator=generator)
        
        return scale_h, scale_w, down_params, up_params

    def forward(self, image: torch.Tensor, 
                scale_factor: Union[float, Tuple[float, float]] = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        
        # Deterministic path (no randomness)
        if scale_factor is not None:
            scale_h, scale_w = (scale_factor, scale_factor) if isinstance(scale_factor, float) else scale_factor
            # Use default non-random parameters
            down_params = (self.downsample_modes[0], False if self.downsample_modes[0] in ['bilinear', 'bicubic'] else None)
            up_params = (self.upsample_modes[0], False if self.upsample_modes[0] in ['bilinear', 'bicubic'] else None)
            return self._apply_resize(image, scale_h, scale_w, down_params, up_params)

        # "Batch-Uniform" random path
        if not self.per_image_randomization:
            params = self._get_random_params(generator=generator)
            return self._apply_resize(image, *params)
        
        # "Per-Image" random path
        else:
            return self._apply_resize_per_image(image, generator=generator)

    def _apply_resize(self, image: torch.Tensor, scale_h: float, scale_w: float, 
                      down_params: Tuple[str, Optional[bool]], 
                      up_params: Tuple[str, Optional[bool]]) -> torch.Tensor:
        if scale_h >= 1.0 and scale_w >= 1.0: return image
        _, _, h, w = image.shape
        
        down_mode, down_align = down_params
        up_mode, up_align = up_params
        
        # Use recompute_scale_factor=True for non-integer scale factors with align_corners=False
        downsampled = F.interpolate(image, scale_factor=(scale_h, scale_w), mode=down_mode, 
                                    align_corners=down_align, recompute_scale_factor=True if down_align is False else None)
        
        recovered = F.interpolate(downsampled, size=(h, w), mode=up_mode, align_corners=up_align)
        
        return recovered
    
    def _apply_resize_per_image(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        output_images = [self._apply_resize(image[i:i+1], *self._get_random_params(generator=generator)) for i in range(image.size(0))]
        return torch.cat(output_images, dim=0)
