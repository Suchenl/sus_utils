import torch
import torch.nn.functional as F
import kornia.filters as KF
from typing import *

class MedianFilter(torch.nn.Module):
    """
    Applies a Median Filter to an image tensor.
    Note: While part of the torch graph, the median operation has non-smooth gradients.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 kernel_size_range: Tuple[int, int] = None, 
                 kernel_size_level: str = None,
                 per_image_randomization: bool = False):  # In this implementation, 'False' is faster but less random.
        super(MedianFilter, self).__init__()
        self.per_image_randomization = per_image_randomization
        
        kernel_size_level_map = {"subtle": (3, 3), "moderate": (5, 7), "subtle and moderate": (3, 7), "strong": (9, 11)}
        if kernel_size_range is None and kernel_size_level is None:
            kernel_size_level = "subtle and moderate"

        if kernel_size_range is not None:
            self.kernel_size_range = kernel_size_range
        else:
            self.kernel_size_range = kernel_size_level_map[kernel_size_level]

        mode = "Per-Image Randomization" if self.per_image_randomization else "Batch-Uniform (High Performance)"
        print(f"[MedianFilter] Initialized with kernel size range: {self.kernel_size_range}, Mode: {mode}")

    def forward(self, image: torch.Tensor, 
                kernel_size: int = None, 
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        
        # Deterministic path: if kernel_size is provided, no randomness is involved.
        if kernel_size is not None:
            return self._apply_filter(image, kernel_size)

        # "Batch-Uniform" random path
        if not self.per_image_randomization:
            min_k, max_k = self.kernel_size_range
            ## DDP-SAFE CHANGE ##: Pass the generator to torch.randint
            rand_k = torch.randint(min_k // 2, (max_k // 2) + 1, (1,), generator=generator, device=image.device).item() * 2 + 1
            return self._apply_filter(image, rand_k)
        
        # "Per-Image" random path
        else:
            return self._apply_filter_per_image(image, generator=generator)

    def _apply_filter(self, image: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size <= 1: return image
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        return KF.median_blur(image, (kernel_size, kernel_size))

    def _apply_filter_per_image(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        batch_size = image.size(0)
        min_k, max_k = self.kernel_size_range
        
        ## DDP-SAFE CHANGE ##: Pass the generator to torch.randint
        kernel_sizes = torch.randint(min_k // 2, (max_k // 2) + 1, (batch_size,), 
                                     device=image.device, generator=generator) * 2 + 1
                                     
        output_images = [self._apply_filter(image[i:i+1], k.item()) for i, k in enumerate(kernel_sizes)]
        return torch.cat(output_images, dim=0)