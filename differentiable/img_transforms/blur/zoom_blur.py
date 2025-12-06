import torch
from typing import *
import kornia

class ZoomBlur(torch.nn.Module):
    """
    Applies a highly configurable Zoom Blur to an image tensor.
    
    Explanation: 
        By averaging a series of "digitally zoomed" images, we simulate the radial blur effect produced by 
    rapidly changing the camera's zoom during exposure.
        Imagine you're taking a photo and quickly zooming your lens from "no zoom" to "2x zoom." 
    In this brief moment, the image sensor (CMOS/CCD) records images at all intermediate zoom levels, 
    from 1.0x, 1.1x, 1.2x, and so on, all the way to 2.0x. When these images are superimposed, 
    a blur radiating outward from the center of the image is formed.
    
    This module provides two computation modes to balance performance vs. memory usage:
    - 'speed': (Default) Fully vectorized. Sacrifices memory (space) for maximum 
               performance (time) by eliminating Python loops. Recommended for systems 
               with ample VRAM.
    - 'memory': Iterative. Sacrifices performance for lower memory usage by processing 
                zoom steps in a loop. Recommended for memory-constrained systems.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 zoom_factor_range: Tuple[float, float] = None, 
                 steps_range: Tuple[int, int] = (3, 7),
                 zoom_factor_level: str = None,
                 per_image_randomization: bool = False, # In this implementation, 'False' or 'True' is equal in speed.
                 computation_mode: Literal['speed', 'memory'] = 'speed',
                 padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros'):
        """
        Args:
            zoom_factor_range (Tuple[float, float], optional): 
                Custom range for the final zoom factor.
            steps_range (Tuple[int, int], optional):
                Custom range for the number of steps used to simulate the blur.
            zoom_factor_level (str, optional): 
                A preset level for zoom factor ('subtle', 'moderate', 'strong').
            per_image_randomization (bool, optional):
                If True, each image in a batch gets its own random parameters.
            computation_mode (str, optional):
                'speed' for the vectorized version, 'memory' for the iterative version.
                Defaults to 'speed'.
            padding_mode (str, optional):
                Padding mode for the zoomed images. Defaults to 'zeros'.
        """
        super(ZoomBlur, self).__init__()
        self.per_image_randomization = per_image_randomization
        self.steps_range = steps_range
        self.padding_mode = padding_mode
        
        if computation_mode not in ['speed', 'memory']:
            raise ValueError("computation_mode must be either 'speed' or 'memory'")
        self.computation_mode = computation_mode

        factor_level_map = {
            "subtle": (0.9, 1.1), "moderate": (0.8, 1.2),
            "subtle and moderate": (0.85, 1.15), "strong": (0.65, 1.35)
        }
        if zoom_factor_range is None and zoom_factor_level is None:
            zoom_factor_level = "subtle and moderate"
        
        if zoom_factor_range is not None:
            self.zoom_factor_range = zoom_factor_range
        else:
            self.zoom_factor_range = factor_level_map[zoom_factor_level]

        rand_mode = "Per-Image" if self.per_image_randomization else "Batch-Uniform"
        print(f"[ZoomBlur] Zoom factor range: {self.zoom_factor_range}\n  - Steps range: {self.steps_range}")
        print(f"  - Randomization Mode: {rand_mode} \n  -Computation Mode: {self.computation_mode.capitalize()}")

    def forward(self, image: torch.Tensor, 
                zoom_factor: float = None, 
                steps: int = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        batch_size = image.size(0)
        
        # --- 1. Determine parameters for the batch (DDP-SAFE) ---
        # Determine `steps` tensor
        if steps is not None:
            steps_tensor = torch.tensor([steps] * batch_size, device=image.device)
        else:
            min_s, max_s = self.steps_range
            if not self.per_image_randomization:
                ## DDP-SAFE CHANGE ##: Use the generator
                rand_s = torch.randint(min_s, max_s + 1, (1,), device=image.device, generator=generator)
                steps_tensor = rand_s.repeat(batch_size)
            else:
                ## DDP-SAFE CHANGE ##: Use the generator
                steps_tensor = torch.randint(min_s, max_s + 1, (batch_size,), device=image.device, generator=generator)

        # Determine `zoom_factor` tensor
        if zoom_factor is not None:
            zoom_factor_tensor = torch.tensor([zoom_factor], device=image.device, dtype=image.dtype).repeat(batch_size)
        else:
            min_val, max_val = self.zoom_factor_range
            if not self.per_image_randomization:
                ## DDP-SAFE CHANGE ##: Use torch.rand with generator instead of empty().uniform_
                rand_float = torch.rand(1, device=image.device, dtype=image.dtype, generator=generator)
                rand_factor = min_val + (max_val - min_val) * rand_float
                zoom_factor_tensor = rand_factor.repeat(batch_size)
            else:
                ## DDP-SAFE CHANGE ##: Use torch.rand with generator instead of empty().uniform_
                rand_floats = torch.rand(batch_size, device=image.device, dtype=image.dtype, generator=generator)
                zoom_factor_tensor = min_val + (max_val - min_val) * rand_floats
        
        # --- 2. Dispatch to the correct implementation based on mode ---
        if self.computation_mode == 'speed':
            return self._apply_blur_vectorized(image, zoom_factor_tensor, steps_tensor)
        else: # 'memory'
            return self._apply_blur_iterative(image, zoom_factor_tensor, steps_tensor)
        
    def _apply_blur_vectorized(self, image: torch.Tensor, zoom_factors: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """Space-for-time implementation. High performance, high memory usage."""
        b, c, h, w = image.shape
        max_steps = int(steps.max().item())
        if max_steps <= 1: return image

        image_expanded = image.unsqueeze(1).repeat(1, max_steps, 1, 1, 1)
        image_super_batch = image_expanded.view(-1, c, h, w)

        interp_base = torch.linspace(0.0, 1.0, max_steps, device=image.device, dtype=image.dtype).view(1, -1)
        scales = 1.0 + interp_base * (zoom_factors.view(-1, 1) - 1.0)
        scales_flat = scales.view(-1)

        center = torch.tensor([[w / 2, h / 2]], device=image.device, dtype=image.dtype).repeat(b * max_steps, 1)
        # FIXED ↓ change [B,1] to [B,2]
        scale_tensor = scales_flat.view(-1, 1).repeat(1, 2)  
        M = kornia.geometry.transform.get_rotation_matrix2d(
            center, torch.zeros_like(center[:, 0]), scale_tensor
        )
        warped_super_batch = kornia.geometry.transform.warp_affine(image_super_batch, M, dsize=(h, w), padding_mode=self.padding_mode)

        warped_expanded = warped_super_batch.view(b, max_steps, c, h, w)
        mask = (torch.arange(max_steps, device=image.device).view(1, -1) < steps.view(-1, 1)).view(b, max_steps, 1, 1, 1)
        masked_sum = torch.sum(warped_expanded * mask, dim=1)

        return masked_sum / steps.view(-1, 1, 1, 1)

    def _apply_blur_iterative(self, image: torch.Tensor, zoom_factors: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """Time-for-space implementation. Low memory usage, lower performance."""
        b, c, h, w = image.shape
        max_steps = int(steps.max().item())
        if max_steps <= 1: return image

        out = torch.zeros_like(image)
        center = torch.tensor([[w / 2, h / 2]], device=image.device, dtype=image.dtype).repeat(b, 1)

        interp_base = torch.linspace(0.0, 1.0, max_steps, device=image.device, dtype=image.dtype).view(1, -1)
        scales_all_steps = 1.0 + interp_base * (zoom_factors.view(-1, 1) - 1.0)

        for i in range(max_steps):
            active_mask = (steps > i)
            if not active_mask.any():
                break

            active_image = image[active_mask]
            active_scales = scales_all_steps[active_mask, i].view(-1, 1)
            active_center = center[active_mask]

            active_scales = active_scales.repeat(1, 2)  
            M = kornia.geometry.transform.get_rotation_matrix2d(
                active_center, torch.zeros_like(active_center[:, 0]), active_scales
            )
            warped = kornia.geometry.transform.warp_affine(active_image, M, dsize=(h, w), padding_mode=self.padding_mode)

            out[active_mask] += warped

        return out / steps.view(-1, 1, 1, 1)