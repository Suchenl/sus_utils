import torch
import torch.nn.functional as F
from typing import *

class DefocusBlur(torch.nn.Module):
    """
    Applies Defocus Blur (Disk Blur) to an image tensor using a custom disk kernel.

    Core Concept: Simulating camera lens "Out of Focus" effect.

    Ideal Camera:
    - Under perfect focus, a distant point light source should converge into an equally infinitesimal point on the sensor.

    Out-of-Focus Camera:
    - When the lens is defocused, light rays from that point source no longer converge to a single point.
    - Instead, they spread out to form a blurred circle on the sensor.

    Circle of Confusion (CoC):
    - In a simplified, idealized optical model, this blurred spot takes the shape of a uniform, solid circular disk.
    - This circular disk is called the "Circle of Confusion".
    - The more severe the defocus, the larger the radius of the Circle of Confusion.
    """
    def __init__(self, 
                 kernel_size_range: Tuple[int, int] = None, 
                 kernel_size_level: str = None,
                 per_image_randomization: bool = False):  # In this implementation, 'False' is faster but less random.
        super(DefocusBlur, self).__init__()
        self.per_image_randomization = per_image_randomization
        kernel_size_level_map = {"subtle": (3, 5), "moderate": (7, 11), "subtle and moderate": (3, 11), "strong": (13, 21)}
        if kernel_size_range is None and kernel_size_level is None:
            kernel_size_level = "subtle and moderate"
        if kernel_size_range is not None: self.kernel_size_range = kernel_size_range
        else: self.kernel_size_range = kernel_size_level_map[kernel_size_level]
        mode = "Per-Image" if self.per_image_randomization else "Batch-Uniform"
        print(f"[DefocusBlur] Range: {self.kernel_size_range}, Mode: {mode}")

    def forward(self, image: torch.Tensor, kernel_size: int = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if kernel_size is not None:
            return self._apply_blur(image, kernel_size)
        if not self.per_image_randomization:
            min_k, max_k = self.kernel_size_range
            rand_k = torch.randint(min_k // 2, (max_k // 2) + 1, (1,), device=image.device, generator=generator).item() * 2 + 1
            return self._apply_blur(image, rand_k)
        else:
            return self._apply_blur_per_image(image, generator=generator)

    def _create_disk_kernel(self, kernel_size: int, device, dtype) -> torch.Tensor:
        # Create a 2D grid of coordinates
        ax = torch.arange(kernel_size, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing='xy')
        center = (kernel_size - 1) / 2.0
        # Calculate distance from center
        dist = torch.sqrt((xx - center)**2 + (yy - center)**2)
        # Create the binary disk
        kernel = (dist <= center).float()
        # Normalize to sum to 1
        return kernel / torch.sum(kernel)

    def _apply_blur(self, image: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size <= 1: return image
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        
        b, c, h, w = image.shape
        kernel = self._create_disk_kernel(kernel_size, image.device, image.dtype)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
        kernel = kernel.to(dtype=image.dtype)
        
        return F.conv2d(image, kernel, padding='same', groups=c)
        
    def _apply_blur_per_image(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        batch_size = image.size(0)
        min_k, max_k = self.kernel_size_range
        kernel_sizes = torch.randint(min_k // 2, (max_k // 2) + 1, (batch_size,), device=image.device, generator=generator) * 2 + 1
        output_images = [self._apply_blur(image[i:i+1], k.item()) for i, k in enumerate(kernel_sizes)]
        return torch.cat(output_images, dim=0)