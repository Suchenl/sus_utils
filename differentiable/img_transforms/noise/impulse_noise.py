import torch
from typing import Tuple, Optional

class ImpulseNoise(torch.nn.Module):
    """
    Applies general Impulse Noise to an image tensor.
    
    This implementation is robust to different input ranges (e.g., [0, 1] or [-1, 1]).
    It first normalizes the image to a standard [0, 1] space, replaces a random 
    portion of pixels with random intensity values within that space, and then
    converts the image back to its original range.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 amount_range: Tuple[float, float] = None, 
                 amount_level: str = None,
                 per_image_randomization: bool = False): # In this implementation, 'False' or 'True' is equal in speed.
        super(ImpulseNoise, self).__init__()
        self.per_image_randomization = per_image_randomization

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
        print(f"[ImpulseNoise (Robust)] Initialized with amount range: {self.amount_range}, Mode: {mode}")
            
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
        
        ## DDP-SAFE CHANGE ##: Use the generator for creating the corruption mask
        # Create a new tensor with the specified generator, not rand_like
        corruption_mask_rand = torch.rand(image.shape, device=image.device, dtype=image.dtype, generator=generator)
        corruption_mask = corruption_mask_rand < amount_tensor
        
        ## DDP-SAFE CHANGE ##: Use the generator for creating the replacement values
        # Create a new tensor with the specified generator, not rand_like
        random_values = torch.rand(image.shape, device=image.device, dtype=image.dtype, generator=generator)
        
        # Apply the random values using the mask (unchanged)
        noisy_image[corruption_mask] = random_values[corruption_mask]
        
        return noisy_image
    
# --- Example of how to use and test the corrected version ---
if __name__ == '__main__':
    impulse_augmenter = ImpulseNoise(amount_level="strong", per_image_randomization=True)

    # --- Test with an image in [0, 1] range ---
    print("\n--- Testing with input range [0, 1] ---")
    zero_one_tensor = torch.zeros(2, 3, 32, 32) # Use zeros to easily see the noise
    print(f"Input min: {zero_one_tensor.min():.2f}, Input max: {zero_one_tensor.max():.2f}")
    output1 = impulse_augmenter(zero_one_tensor)
    print(f"Output min: {output1.min():.2f}, Output max: {output1.max():.2f}")
    # The output should remain within the [0, 1] range
    assert output1.min() >= 0.0 and output1.max() <= 1.0
    print("Test Passed! Output is correctly in [0, 1].")

    # --- Test with an image in [-1, 1] range ---
    print("\n--- Testing with input range [-1, 1] ---")
    neg_one_tensor = torch.ones(2, 3, 32, 32) * -1.0 # Use -1s to easily see the noise
    print(f"Input min: {neg_one_tensor.min():.2f}, Input max: {neg_one_tensor.max():.2f}")
    output2 = impulse_augmenter(neg_one_tensor)
    print(f"Output min: {output2.min():.2f}, Output max: {output2.max():.2f}")
    # The output should remain within the [-1, 1] range
    assert output2.min() >= -1.0 and output2.max() <= 1.0
    print("Test Passed! Output is correctly in [-1, 1].")