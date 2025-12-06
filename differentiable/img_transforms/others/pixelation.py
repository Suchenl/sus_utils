import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

class Pixelation(torch.nn.Module):
    """
    Applies a pixelation effect to an image tensor.
    
    This robust version automatically handles images in the [0, 1] or [-1, 1]
    range, ensuring consistent behavior within a data augmentation pipeline.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 block_size_range: Tuple[int, int] = None, 
                 block_size_level: str = None,
                 per_image_randomization: bool = False):  # In this implementation, 'False' is faster but less random.
        super(Pixelation, self).__init__()

        self.per_image_randomization = per_image_randomization

        block_size_level_map = {
            "subtle": (2, 4), "moderate": (5, 9),
            "subtle and moderate": (3, 8), "strong": (10, 16)
        }
        
        if block_size_range is not None:
            self.block_size_range = block_size_range
        elif block_size_level is not None:
            if block_size_level not in block_size_level_map:
                raise ValueError(f"Unknown block size level: {block_size_level}")
            self.block_size_range = block_size_level_map[block_size_level]
        else:
            self.block_size_range = block_size_level_map["subtle and moderate"]
            block_size_level = "subtle and moderate"
        
        mode = "Per-Image Randomization" if self.per_image_randomization else "Batch-Uniform (High Performance)"
        level_info = f"level='{block_size_level}'" if 'block_size_level' in locals() else "custom range"
        print(f"[Pixelation (Robust)] Initialized with {level_info}, range={self.block_size_range}, Mode: {mode}")

    def forward(self, image: torch.Tensor, block_size: int = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): The input image tensor (B, C, H, W).
            block_size (int, optional): Overrides random generation.

        Returns:
            torch.Tensor: The pixelated image tensor in the same range as the input.
        """
        # --- Step 1: Standardize input range for processing (Best Practice) ---
        # Although not strictly necessary for pixelation, this makes the module
        # consistent with other augmentation modules.
        is_neg_one_range = image.min() < -0.01
        
        # Process the core logic on the original image since interpolate is range-preserving.
        # This check is primarily to ensure we return the correct range.
        
        # --- Step 2: Apply Pixelation Logic ---
        if block_size is not None:
            # Deterministic path
            pixelated_image = self._apply_pixelation(image, int(block_size))
        elif not self.per_image_randomization:
            # "Batch-Uniform" random path
            min_val, max_val = self.block_size_range
            ## DDP-SAFE CHANGE ##: Use the generator
            random_block_size = torch.randint(min_val, max_val + 1, (1,), generator=generator, device=image.device).item()
            pixelated_image = self._apply_pixelation(image, random_block_size)
        else:
            # "Per-Image" random path
            ## DDP-SAFE CHANGE ##: Pass the generator to the helper
            pixelated_image = self._apply_pixelation_per_image(image, generator=generator)

        # Output range handling (unchanged)
        if is_neg_one_range:
            return torch.clamp(pixelated_image, -1.0, 1.0)
        else:
            return torch.clamp(pixelated_image, 0.0, 1.0)

    def _apply_pixelation(self, image: torch.Tensor, block_size: int) -> torch.Tensor:
        """Helper function for the fast, vectorized pixelation."""
        if block_size <= 1:
            return image
        _, _, h, w = image.shape
        down_h, down_w = max(1, h // block_size), max(1, w // block_size)
        downsampled = F.interpolate(image, size=(down_h, down_w), mode='bilinear', align_corners=False)
        pixelated = F.interpolate(downsampled, size=(h, w), mode='nearest')
        return pixelated
    
    def _apply_pixelation_per_image(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Helper function for the iterative, per-image pixelation."""
        batch_size = image.size(0)
        min_val, max_val = self.block_size_range
        block_sizes_tensor = torch.randint(min_val, max_val + 1, (batch_size,), 
                                           device=image.device, generator=generator)
        output_images = [
            self._apply_pixelation(img.unsqueeze(0), size.item())
            for img, size in zip(image, block_sizes_tensor)
        ]
        return torch.cat(output_images, dim=0)