import torch
from typing import *
import kornia
import random

class ElasticTransform(torch.nn.Module):
    """
    Applies a highly configurable elastic transformation to an image tensor.

    This module provides two computation modes to balance performance vs. memory usage:
    - 'speed': (Default) Fully vectorized. High performance, but also high memory usage
               due to large intermediate tensors. Recommended for systems with ample VRAM.
    - 'memory': Iterative. Lower memory usage by processing each image in the batch
                in a loop, at the cost of performance. Recommended for memory-constrained
                systems, especially when `per_image_randomization` is True.
    """
    def __init__(self, 
                 alpha_range: Tuple[float, float] = None, 
                 sigma_range: Tuple[float, float] = None,
                 kernel_size_range: Tuple[int, int] = (9, 13),
                 strength_level: str = None,
                 per_image_randomization: bool = False,
                 computation_mode: Literal['speed', 'memory'] = 'speed'):
        super(ElasticTransform, self).__init__()
        self.per_image_randomization = per_image_randomization
        self.kernel_size_range = kernel_size_range
        
        if computation_mode not in ['speed', 'memory']:
            raise ValueError("computation_mode must be either 'speed' or 'memory'")
        self.computation_mode = computation_mode

        strength_level_map = {
            "subtle": ((0.5, 1.0), (0.1, 0.2)), "moderate": ((1.0, 2.5), (0.2, 0.4)),
            "subtle and moderate": ((0.5, 2.0), (0.1, 0.4)), "strong": ((2.5, 5.0), (0.4, 0.8))
        }
        if strength_level is None and (alpha_range is None or sigma_range is None):
            strength_level = "subtle and moderate"
        
        if alpha_range is not None and sigma_range is not None:
            self.alpha_range = alpha_range
            self.sigma_range = sigma_range
        else:
            self.alpha_range, self.sigma_range = strength_level_map[strength_level]

        rand_mode = "Per-Image" if self.per_image_randomization else "Batch-Uniform"
        print(f"--- [ElasticTransform Initialized] ---")
        print(f"  - Alpha (Intensity) Range: {self.alpha_range}")
        print(f"  - Sigma (Smoothness) Range: {self.sigma_range}")
        print(f"  - Kernel Size Range: {self.kernel_size_range}")
        print(f"  - Randomization Mode: {rand_mode}, Computation Mode: {self.computation_mode.capitalize()}")
        print(f"------------------------------------")

    def _get_random_params(self) -> Tuple[float, float, int]:
        alpha = random.uniform(*self.alpha_range)
        sigma = random.uniform(*self.sigma_range)
        min_k, max_k = self.kernel_size_range
        kernel_size = random.randint(min_k // 2, max_k // 2) * 2 + 1
        return alpha, sigma, kernel_size

    def forward(self, image: torch.Tensor, alpha: float = None, sigma: float = None, kernel_size: int = None) -> torch.Tensor:
        # --- 1. Dispatch based on user override ---
        if all(p is not None for p in [alpha, sigma, kernel_size]):
            # Deterministic path, always use vectorized as parameters are uniform
            return self._apply_transform_vectorized(image, alpha, sigma, kernel_size)

        # --- 2. Dispatch based on computation mode and randomization ---
        # For batch-uniform, the vectorized approach is always better and memory is not an issue
        if not self.per_image_randomization:
            params = self._get_random_params()
            return self._apply_transform_vectorized(image, *params)
        else: # per_image_randomization is True
            if self.computation_mode == 'speed':
                return self._apply_transform_vectorized_per_image(image)
            else: # 'memory'
                return self._apply_transform_iterative(image)

    def _apply_transform_vectorized(self, image: torch.Tensor, alpha: float, sigma: float, kernel_size: int) -> torch.Tensor:
        """Helper for batch-uniform or deterministic application."""
        b, c, h, w = image.shape
        noise = torch.rand(b, h, w, 2, device=image.device, dtype=image.dtype) * 2 - 1
        
        sigma_tensor = torch.tensor([sigma, sigma], device=image.device, dtype=image.dtype).view(1, 2)
        alpha_tensor = torch.tensor([alpha, alpha], device=image.device, dtype=image.dtype).view(1, 1, 1, 2)

        displacement_field = kornia.filters.gaussian_blur2d(
            noise.permute(0, 3, 1, 2), (kernel_size, kernel_size), sigma_tensor
        ).permute(0, 2, 3, 1) * alpha_tensor
        
        return kornia.geometry.transform.remap(image, displacement_field)

    def _apply_transform_vectorized_per_image(self, image: torch.Tensor) -> torch.Tensor:
        """Helper for speed-priority, per-image randomization."""
        b, c, h, w = image.shape
        noise = torch.rand(b, h, w, 2, device=image.device, dtype=image.dtype) * 2 - 1

        # Kornia blur doesn't support batched kernel_size, so we pick one for the batch
        min_k, max_k = self.kernel_size_range
        kernel_size = random.randint(min_k // 2, max_k // 2) * 2 + 1
        
        # Generate per-image alpha and sigma
        min_a, max_a = self.alpha_range
        alpha_tensor = torch.empty(b, 1, 1, 2, device=image.device, dtype=image.dtype).uniform_(min_a, max_a)
        alpha_tensor[..., 1] = alpha_tensor[..., 0] # Isotropic alpha

        min_s, max_s = self.sigma_range
        sigma_tensor = torch.empty(b, 2, device=image.device, dtype=image.dtype).uniform_(min_s, max_s)
        sigma_tensor[:, 1] = sigma_tensor[:, 0] # Isotropic sigma

        displacement_field = kornia.filters.gaussian_blur2d(
            noise.permute(0, 3, 1, 2), (kernel_size, kernel_size), sigma_tensor
        ).permute(0, 2, 3, 1) * alpha_tensor

        return kornia.geometry.transform.remap(image, displacement_field)

    def _apply_transform_iterative(self, image: torch.Tensor) -> torch.Tensor:
        """Helper for memory-priority, per-image randomization."""
        # This is the original safe but slow implementation
        output_images = [
            self._apply_transform_vectorized(image[i:i+1], *self._get_random_params()) 
            for i in range(image.size(0))
        ]
        return torch.cat(output_images, dim=0)