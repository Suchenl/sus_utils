import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from typing import *

class GaussianBlur(torch.nn.Module):
    """
    Applies Gaussian Blur to an image tensor.
    Supports both a high-performance batch-uniform mode and a
    high-randomness per-image randomization mode.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 kernel_size_range: Tuple[int, int] = None, 
                 kernel_size_level: str = None,
                 per_image_randomization: bool = False):  # In this implementation, 'False' is faster but less random.
        super(GaussianBlur, self).__init__()
        self.per_image_randomization = per_image_randomization
        
        kernel_size_level_map = {"subtle": (3, 5), "moderate": (7, 11), "subtle and moderate": (3, 11), "strong": (13, 21)}
        # Default range, ensuring it covers "subtle and moderate"
        if kernel_size_range is None and kernel_size_level is None:
            kernel_size_level = "subtle and moderate"

        if kernel_size_range is not None:
            self.kernel_size_range = kernel_size_range
        else:
            self.kernel_size_range = kernel_size_level_map[kernel_size_level]

        mode = "Per-Image Randomization" if self.per_image_randomization else "Batch-Uniform (High Performance)"
        print(f"[GaussianBlur] Initialized with kernel size range: {self.kernel_size_range}, Mode: {mode}")

    def forward(self, image: torch.Tensor, 
                kernel_size: int = None, 
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        
        # Deterministic path: if kernel_size is provided, no randomness is involved.
        if kernel_size is not None:
            return self._apply_blur(image, kernel_size)

        # "Batch-Uniform" random path
        if not self.per_image_randomization:
            min_k, max_k = self.kernel_size_range
            ## DDP-SAFE CHANGE ##: Pass the generator to torch.randint
            rand_k = torch.randint(min_k // 2, (max_k // 2) + 1, (1,), generator=generator, device=image.device).item() * 2 + 1
            return self._apply_blur(image, rand_k)
        
        # "Per-Image" random path
        else:
            return self._apply_blur_per_image(image, generator=generator)

    def _apply_blur(self, image: torch.Tensor, kernel_size: int) -> torch.Tensor:
        # kernel_size must be a positive and odd integer
        if kernel_size <= 1: return image
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        return F_vision.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])

    def _apply_blur_per_image(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        batch_size = image.size(0)
        min_k, max_k = self.kernel_size_range
        
        ## DDP-SAFE CHANGE ##: Pass the generator to torch.randint
        kernel_sizes = torch.randint(min_k // 2, (max_k // 2) + 1, (batch_size,), device=image.device, generator=generator) * 2 + 1
        
        output_images = [self._apply_blur(image[i:i+1], k.item()) for i, k in enumerate(kernel_sizes)]
        return torch.cat(output_images, dim=0)

