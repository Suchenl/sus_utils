import torch
import torch.nn as nn
from typing import Tuple, Optional
import kornia

class SaturationAdjustment(torch.nn.Module):
    """
    Adjusts the saturation of an image tensor using the high-performance kornia library.
    
    This implementation provides separate controls for decreasing and increasing saturation,
    with a probabilistic choice between the two operations. It is fully vectorized and
    handles both batch-uniform and per-image randomization efficiently without loops.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 down_factor_range: Optional[Tuple[float, float]] = (0.2, 1.0),
                 up_factor_range: Optional[Tuple[float, float]] = (1.0, 2.0),
                 down_prob: float = 0.5,
                 per_image_randomization: bool = False):  # In this implementation, 'False' or 'True' is equal in speed.
        """
        Args:
            down_factor_range (Tuple[float, float], optional): 
                Range for decreasing saturation (desaturating), e.g., (0.2, 1.0). Must be <= 1.
            up_factor_range (Tuple[float, float], optional): 
                Range for increasing saturation (oversaturating), e.g., (1.0, 2.0). Must be >= 1.
            down_prob (float): 
                Probability of applying desaturation. Oversaturation probability is (1 - down_prob).
            per_image_randomization (bool): 
                If True, applies a different random factor to each image in the batch.
        """
        super(SaturationAdjustment, self).__init__()
        
        # --- Parameter Validation ---
        if down_factor_range is None and up_factor_range is None:
            raise ValueError("At least one of down_factor_range or up_factor_range must be provided.")
        if down_factor_range is not None and (down_factor_range[0] > down_factor_range[1] or down_factor_range[1] > 1.0):
            raise ValueError(f"down_factor_range {down_factor_range} is invalid. Must be [min, max] with max <= 1.0.")
        if up_factor_range is not None and (up_factor_range[0] > up_factor_range[1] or up_factor_range[0] < 1.0):
            raise ValueError(f"up_factor_range {up_factor_range} is invalid. Must be [min, max] with min >= 1.0.")
        if not (0.0 <= down_prob <= 1.0):
            raise ValueError(f"down_prob must be between 0.0 and 1.0, but got {down_prob}.")

        self.down_factor_range = down_factor_range
        self.up_factor_range = up_factor_range
        self.down_prob = down_prob
        self.per_image_randomization = per_image_randomization

        # Adjust effective probability if one range is disabled
        if self.down_factor_range is None: self.down_prob = 0.0
        if self.up_factor_range is None: self.down_prob = 1.0

        print(f"[SaturationAdjustment (Probabilistic Kornia)] Initialized.")
        if self.down_factor_range: print(f"  - Desaturate Range: {self.down_factor_range}, Prob: {self.down_prob if self.up_factor_range else 1.0}")
        if self.up_factor_range: print(f"  - Oversaturate Range: {self.up_factor_range}, Prob: {1.0 - self.down_prob if self.down_factor_range else 1.0}")

    def forward(self, image: torch.Tensor, 
                factor: float = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        
        is_neg_one_range = image.min() < -0.01
        image_0_1 = (image + 1.0) / 2.0 if is_neg_one_range else image
        batch_size = image.size(0)
        
        # --- DDP-SAFE Factor Generation ---
        if factor is not None:
            # Deterministic path
            if factor < 0: raise ValueError("Factor must be a non-negative number.")
            factor_tensor = torch.full((batch_size,), factor, device=image.device, dtype=image.dtype)
        else:
            if not self.per_image_randomization:
                # "Batch-Uniform" random path
                ## DDP-SAFE CHANGE ##: Replace random.random()
                choice_rand = torch.rand(1, generator=generator).item()
                factor_range_to_use = self.down_factor_range if choice_rand < self.down_prob else self.up_factor_range
                
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                min_f, max_f = factor_range_to_use
                rand_float = torch.rand(1, generator=generator).item()
                rand_factor = min_f + (max_f - min_f) * rand_float
                
                factor_tensor = torch.full((batch_size,), rand_factor, device=image.device, dtype=image.dtype)
            else:
                # "Per-Image" random path
                ## DDP-SAFE CHANGE ##: Replace rand() and empty().uniform_
                choices = torch.rand(batch_size, device=image.device, generator=generator) < self.down_prob
                
                min_down, max_down = self.down_factor_range
                min_up, max_up = self.up_factor_range
                
                rand_floats_down = torch.rand(batch_size, device=image.device, generator=generator)
                rand_floats_up = torch.rand(batch_size, device=image.device, generator=generator)

                down_factors = min_down + (max_down - min_down) * rand_floats_down
                up_factors = min_up + (max_up - min_up) * rand_floats_up
                
                factor_tensor = torch.where(choices, down_factors, up_factors)

        # Apply kornia's function (unchanged)
        adjusted_image_0_1 = kornia.enhance.adjust_saturation(image_0_1, factor_tensor)
        
        # Convert back to original range (unchanged)
        output_final = adjusted_image_0_1 * 2.0 - 1.0 if is_neg_one_range else adjusted_image_0_1
        return torch.clamp(output_final, -1.0 if is_neg_one_range else 0.0, 1.0)


# --- Example Usage ---
if __name__ == '__main__':
    # ... (Setup: create a dummy batch tensor) ...
    input_batch = torch.rand(10, 3, 64, 64) * 2.0 - 1.0
    
    print("\n--- Testing Probabilistic Saturation Adjustment (per_image_randomization=True) ---")
    
    # Initialize with separate up/down ranges and 50% probability
    saturation_adjuster = SaturationAdjustment(
        down_factor_range=(0.1, 0.7),   # Desaturate
        up_factor_range=(1.2, 2.5),     # Oversaturate
        down_prob=0.5,
        per_image_randomization=True
    )
    
    # Apply the transform. The output is a single tensor.
    output_batch = saturation_adjuster(input_batch)
    
    print(f"Successfully processed a batch of {output_batch.size(0)} images with per-image saturation randomization.")
    # You could save and inspect individual images from the batch to verify that
    # some are less saturated and others are more saturated than the original.```