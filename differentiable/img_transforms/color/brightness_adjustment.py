import torch
import torch.nn as nn
from typing import Tuple, Optional
import kornia # <-- Use kornia

class BrightnessAdjustment(nn.Module):
    """
    (Corrected Differentiable Version)
    Adjusts the brightness of an image tensor by manipulating the V (Value) channel
    in the HSV color space using the differentiable kornia library.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 down_factor_range: Optional[Tuple[float, float]] = (0.5, 0.99),
                 up_factor_range: Optional[Tuple[float, float]] = (1.1, 2.0),
                 down_prob: float = 0.5,
                 per_image_randomization: bool = False):
        super(BrightnessAdjustment, self).__init__()
        
        # --- (Parameter validation remains the same, it's correct) ---
        if down_factor_range is None and up_factor_range is None:
            raise ValueError("At least one of down_factor_range or up_factor_range must be provided.")
        # ... (rest of the __init__ is fine) ...
        self.down_factor_range = down_factor_range
        self.up_factor_range = up_factor_range
        self.down_prob = down_prob
        self.per_image_randomization = per_image_randomization
        if self.down_factor_range is None: self.down_prob = 0.0
        if self.up_factor_range is None: self.down_prob = 1.0

    def forward(self, image: torch.Tensor, 
                factor: float = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        
        is_neg_one_range = image.min() < -0.01 
        image_0_1 = (image + 1.0) / 2.0 if is_neg_one_range else image
        batch_size = image.size(0)

        # --- DDP-SAFE Random factor generation ---
        if factor is not None:
            # Deterministic path
            if factor < 0: raise ValueError("Factor must be a non-negative number.")
            factor_tensor = torch.full((batch_size, 1, 1, 1), factor, device=image.device, dtype=image.dtype)
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
                
                factor_tensor = torch.full((batch_size, 1, 1, 1), rand_factor, device=image.device, dtype=image.dtype)
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
                
                final_factors = torch.where(choices, down_factors, up_factors).view(batch_size, 1, 1, 1)
                factor_tensor = final_factors
        
        # --- HSV Adjustment (unchanged) ---
        hsv_image = kornia.color.rgb_to_hsv(image_0_1)
        h, s, v = hsv_image.chunk(3, dim=1)
        
        v_adjusted = v * factor_tensor
        
        saturation_reduction = torch.relu(v_adjusted - 1.0)
        s_adjusted = torch.clamp(s - saturation_reduction, 0.0, 1.0)
        v_final = torch.clamp(v_adjusted, 0.0, 1.0)
        
        adjusted_hsv = torch.cat([h, s_adjusted, v_final], dim=1)
        output_0_1 = kornia.color.hsv_to_rgb(adjusted_hsv).clamp(0.0, 1.0)
        output_final = output_0_1 * 2.0 - 1.0 if is_neg_one_range else output_0_1
        
        return output_final