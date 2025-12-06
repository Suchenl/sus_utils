import torch
from typing import *

class SaltAndPepperNoise(torch.nn.Module):
    """
    Applies Salt and Pepper Noise to an image tensor.

    This robust implementation correctly handles different input ranges (e.g., [0, 1] or [-1, 1]).
    It normalizes the image to a [0, 1] space, where "salt" is the maximum value (1.0)
    and "pepper" is the minimum value (0.0). The image is then converted back to
    its original range.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 amount_range: Tuple[float, float] = None, 
                 amount_level: str = None,
                 salt_ratio: float = 0.5,
                 per_image_randomization: bool = False): # In this implementation, 'False' or 'True' is equal in speed.
        super(SaltAndPepperNoise, self).__init__()
        self.per_image_randomization = per_image_randomization
        if not (0.0 <= salt_ratio <= 1.0):
            raise ValueError("salt_ratio must be between 0.0 and 1.0")
        self.salt_ratio = salt_ratio

        # --- Amount level mapping ---
        amount_level_map = {
            "subtle": (0.01, 0.04), "moderate": (0.04, 0.1),
            "subtle and moderate": (0.01, 0.1), "strong": (0.1, 0.2)
        }
        if amount_range is None and amount_level is None:
            amount_level = "subtle and moderate"
        
        if amount_range is not None:
            self.amount_range = amount_range
        else:
            self.amount_range = amount_level_map[amount_level]

        mode = "Per-Image Randomization" if self.per_image_randomization else "Batch-Uniform (High Performance)"
        print(f"[SaltAndPepperNoise (Robust)] Initialized with amount range: {self.amount_range}, Mode: {mode}")
            
    def forward(self, image: torch.Tensor, 
                amount: float = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        
        # Step 1: Normalize to [0, 1] (unchanged)
        is_neg_one_range = image.min() < -0.01
        image_0_1 = (image + 1.0) / 2.0 if is_neg_one_range else image
        batch_size = image.size(0)

        # --- Step 2: DDP-SAFE amount of noise generation ---
        if amount is not None:
            # Deterministic path
            if not (0.0 <= amount <= 1.0): raise ValueError("Amount must be between 0.0 and 1.0.")
            amount_tensor = torch.full((1, 1, 1, 1), amount, device=image.device, dtype=image.dtype)
        else:
            min_val, max_val = self.amount_range
            if not self.per_image_randomization:
                # "Batch-Uniform" random path
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                rand_float = torch.rand(1, generator=generator, device=image.device).item()
                rand_amount = min_val + (max_val - min_val) * rand_float
                amount_tensor = torch.full((1, 1, 1, 1), rand_amount, device=image.device, dtype=image.dtype)
            else:
                # "Per-Image" random path
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                rand_floats = torch.rand(batch_size, device=image.device, generator=generator)
                amount_values = min_val + (max_val - min_val) * rand_floats
                amount_tensor = amount_values.view(batch_size, 1, 1, 1)

        # --- Step 3: DDP-SAFE noise application ---
        ## DDP-SAFE CHANGE ##: Pass the generator to the helper function
        noisy_image_0_1 = self._apply_noise(image_0_1, amount_tensor, generator=generator)

        # --- Step 4: Convert back to original range (unchanged) ---
        output_final = noisy_image_0_1 * 2.0 - 1.0 if is_neg_one_range else noisy_image_0_1
        
        return output_final

    def _apply_noise(self, image: torch.Tensor, amount_tensor: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Helper function to apply noise. Assumes input image is in [0, 1] range.
        """
        noisy_image = image.clone()
        
        ## DDP-SAFE CHANGE ##: Use the generator for creating the random map
        # Create a new tensor with the specified generator, not rand_like
        random_map = torch.rand(image.shape, device=image.device, dtype=image.dtype, generator=generator)
        
        # Determine thresholds (unchanged)
        salt_threshold = amount_tensor * self.salt_ratio
        pepper_threshold = amount_tensor
        
        # Create masks (unchanged)
        salt_mask = random_map < salt_threshold
        pepper_mask = (random_map >= salt_threshold) & (random_map < pepper_threshold)
        
        # Apply salt and pepper (unchanged)
        noisy_image[salt_mask] = 1.0
        noisy_image[pepper_mask] = 0.0
        
        return noisy_image