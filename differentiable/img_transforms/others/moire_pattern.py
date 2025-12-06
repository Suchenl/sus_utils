import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

# This is the new, high-performance version.
class MoirePattern(torch.nn.Module):
    """
    (High-Performance Vectorized Version)
    Introduces Moire patterns by overlaying a procedurally generated sine wave grid.
    
    This version is fully vectorized to eliminate Python loops, providing a
    significant performance increase for batch processing, especially on GPUs.

    This implementation is DDP-safe.
    """
    def __init__(self,
                 frequency_range: Tuple[float, float] = (10.0, 40.0),
                 intensity_range: Tuple[float, float] = (0.05, 0.2),
                 angle_range: Tuple[int, int] = (0, 180),
                 per_image_randomization: bool = True):
        super().__init__()
        self.frequency_range = frequency_range
        self.intensity_range = intensity_range
        self.angle_range = angle_range
        self.per_image_randomization = per_image_randomization
        
        mode = "Vectorized Per-Image" if self.per_image_randomization else "Batch-Uniform"
        print(f"[MoirePattern Initialized] Mode: {mode}")

    def forward(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # Pre-processing
        is_neg_one_range = image.min() < -0.01
        image_0_1 = (image + 1.0) / 2.0 if is_neg_one_range else image
        b, _, h, w = image.shape
        device = image.device

        batch_size = b if self.per_image_randomization else 1
        
        # --- Step 1: DDP-SAFE Vectorized Random Parameter Generation ---
        ## DDP-SAFE CHANGE ##: Replace all empty().uniform_ calls
        
        # Generate random floats in [0, 1) for each parameter
        rand_floats_freq = torch.rand(batch_size, device=device, generator=generator)
        rand_floats_intensity = torch.rand(batch_size, device=device, generator=generator)
        rand_floats_angle = torch.rand(batch_size, device=device, generator=generator)

        # Scale the random floats to the desired ranges
        min_f, max_f = self.frequency_range
        min_i, max_i = self.intensity_range
        min_a, max_a = self.angle_range

        freq = min_f + (max_f - min_f) * rand_floats_freq
        intensity = min_i + (max_i - min_i) * rand_floats_intensity
        angle_deg = min_a + (max_a - min_a) * rand_floats_angle

        # --- Step 2: Vectorized Trigonometry (unchanged) ---
        angle_rad = torch.deg2rad(angle_deg)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # --- Step 3: Create Coordinate Grid (unchanged) ---
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        # --- Step 4: Broadcasting (unchanged) ---
        cos_a = cos_a.view(-1, 1, 1)
        sin_a = sin_a.view(-1, 1, 1)
        freq = freq.view(-1, 1, 1)
        
        x_rot = x * cos_a - y * sin_a
        pattern = torch.sin(x_rot * freq * math.pi)
        pattern = (pattern + 1) / 2
        pattern = pattern.unsqueeze(1)
        
        # --- Step 5: Apply Pattern to Image Batch (unchanged) ---
        intensity = intensity.view(-1, 1, 1, 1)
        
        # If not per-image, the batch_size for params was 1, so this broadcasting still works
        noisy_image_0_1 = image_0_1 * (1 - pattern * intensity)
        
        noisy_image_0_1 = torch.clamp(noisy_image_0_1, 0.0, 1.0)
        
        return (noisy_image_0_1 * 2.0 - 1.0) if is_neg_one_range else noisy_image_0_1