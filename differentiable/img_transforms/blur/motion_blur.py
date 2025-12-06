import torch
from typing import *
import kornia

class MotionBlur(torch.nn.Module):
    """
    Applies a highly randomized Motion Blur to an image tensor using kornia.
    This version randomizes kernel size (length), angle (orientation), and
    direction (symmetry) for maximum data augmentation effect.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 kernel_size_range: Tuple[int, int] = None, 
                 angle_range: Tuple[float, float] = (0.0, 360.0),
                 direction_range: Tuple[float, float] = (-1.0, 1.0),
                 kernel_size_level: str = None,
                 per_image_randomization: bool = False):  # In this implementation, 'False' is faster but less random.
        """
        Args:
            kernel_size_range (Tuple[int, int], optional): 
                Custom range for the blur kernel size (length of motion).
            angle_range (Tuple[float, float], optional): 
                Custom range for the motion angle in degrees.
            direction_range (Tuple[float, float], optional): 
                Custom range for the motion direction (symmetry). 
                -1.0 to 1.0. 
                0.0 is centered.
            kernel_size_level (str, optional): 
                A preset level for kernel size ('subtle', 'moderate', 'strong').
            per_image_randomization (bool, optional):
                If False (default), a single set of random parameters is applied
                to the whole batch. If True, each image gets its own random set.
        """
        super(MotionBlur, self).__init__()
        self.per_image_randomization = per_image_randomization
        self.angle_range = angle_range
        self.direction_range = direction_range
        
        kernel_size_level_map = {"subtle": (3, 7), "moderate": (9, 15), "strong": (17, 25)}
        if kernel_size_range is None and kernel_size_level is None:
            kernel_size_level = "moderate"
        if kernel_size_range is not None: 
            self.kernel_size_range = kernel_size_range
        else: 
            self.kernel_size_range = kernel_size_level_map[kernel_size_level]
            
        mode = "Per-Image Randomization" if self.per_image_randomization else "Batch-Uniform (High Performance)"
        print(f"[MotionBlur] Kernel Range: {self.kernel_size_range}, Angle Range: {self.angle_range}, Direction Range: {self.direction_range}, Mode: {mode}")

    def forward(self, image: torch.Tensor, 
                kernel_size: int = None, 
                angle: float = None, 
                direction: float = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        
        # Deterministic path: if all parameters are provided, no randomness is involved.
        if all(p is not None for p in [kernel_size, angle, direction]):
            return kornia.filters.motion_blur(image, kernel_size, angle, direction)

        # "Batch-Uniform" random path
        if not self.per_image_randomization:
            rand_k, rand_a, rand_d = self._get_random_params(generator=generator)
            return kornia.filters.motion_blur(image, rand_k, rand_a, rand_d)
        
        # "Per-Image" random path
        else:
            return self._apply_blur_per_image(image, generator=generator)

    ## DDP-SAFE CHANGE ##: Added a helper to centralize synchronized random parameter generation
    def _get_random_params(self, generator: Optional[torch.Generator] = None) -> Tuple[int, float, float]:
        min_k, max_k = self.kernel_size_range
        min_a, max_a = self.angle_range
        min_d, max_d = self.direction_range

        # Generate random kernel size (integer)
        rand_k = torch.randint(min_k // 2, (max_k // 2) + 1, (1,), generator=generator, device=generator.device if generator is not None else 'cpu').item() * 2 + 1
        
        # Generate random angle and direction (floats)
        # Use torch.rand with the generator for DDP-safe uniform random floats
        rand_vals = torch.rand(2, generator=generator, device=generator.device if generator is not None else 'cpu')
        rand_a = min_a + (max_a - min_a) * rand_vals[0].item()
        rand_d = min_d + (max_d - min_d) * rand_vals[1].item()
        
        return rand_k, rand_a, rand_d
    
    def _apply_blur_per_image(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        batch_size = image.size(0)
        min_k, max_k = self.kernel_size_range
        min_a, max_a = self.angle_range
        min_d, max_d = self.direction_range
        
        # Generate synchronized random parameters for each image in the batch
        ## DDP-SAFE CHANGE ##: Use the generator for all random tensor creations
        kernel_sizes = torch.randint(min_k // 2, (max_k // 2) + 1, (batch_size,), 
                                     device=image.device, generator=generator) * 2 + 1
                                     
        # Use torch.rand with the generator for DDP-safe uniform random floats
        rand_floats_a = torch.rand(batch_size, device=image.device, generator=generator)
        rand_floats_d = torch.rand(batch_size, device=image.device, generator=generator)
        angles = min_a + (max_a - min_a) * rand_floats_a
        directions = min_d + (max_d - min_d) * rand_floats_d

        output_images = [
            kornia.filters.motion_blur(image[i:i+1], k.item(), a.item(), d.item()) 
            for i, (k, a, d) in enumerate(zip(kernel_sizes, angles, directions))
        ]
        
        return torch.cat(output_images, dim=0)