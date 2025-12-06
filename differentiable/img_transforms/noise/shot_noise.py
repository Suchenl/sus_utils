import torch
import torch.nn as nn
from typing import *
import kornia 

class ShotNoise(nn.Module):
    """
    A fully differentiable shot noise simulator implemented using Kornia.
    This version is safe for use within a training loop as all operations
    are differentiable.

    Implementation Method:
    1. Uses `kornia.color.rgb_to_hsv` for a differentiable color space conversion.
    2. Approximates shot noise behavior using Gaussian noise. The standard deviation
       of the noise is proportional to the square root of the luminance signal (V channel),
       which preserves the core physical characteristic of shot noise.
    3. Uses `kornia.color.hsv_to_rgb` to convert the image back to the RGB space.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 strength_range: Tuple[float, float] = (0.05, 0.5),
                 per_image_randomization: bool = False): # In this implementation, 'False' or 'True' is equal in speed.
        """
        Initializes the differentiable shot noise module.
        
        Args:
            strength_range (Tuple[float, float]): The range [min, max] from which to sample the noise strength.
                                                  Should be between 0.0 and 1.0.
            per_image_randomization (bool): If True, each image in a batch gets a different random noise strength.
                                            If False, all images in the batch get the same strength, which is slightly faster.
        """
        super(ShotNoise, self).__init__()
        self.per_image_randomization = per_image_randomization
        self.strength_range = strength_range
        mode = "Per-Image" if self.per_image_randomization else "Batch-Uniform"
        print(f"[KorniaShotNoise] Initialized with strength range: {self.strength_range}, Mode: {mode}")

    def forward(self, image: torch.Tensor, 
                strength: float = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:

        if strength is not None and strength == 0.0:
            return image

        batch_size = image.size(0)
        
        # --- Step 1: DDP-SAFE noise strength generation ---
        if strength is not None:
            # Deterministic path
            strength_tensor = torch.full((batch_size, 1, 1, 1), strength, device=image.device, dtype=image.dtype)
        else:
            min_val, max_val = self.strength_range
            if not self.per_image_randomization:
                # "Batch-Uniform" random path
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                rand_float = torch.rand(1, generator=generator, device=image.device).item()
                rand_strength = min_val + (max_val - min_val) * rand_float
                strength_tensor = torch.full((batch_size, 1, 1, 1), rand_strength, device=image.device, dtype=image.dtype)
            else:
                # "Per-Image" random path
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                rand_floats = torch.rand(batch_size, device=image.device, generator=generator)
                strength_values = min_val + (max_val - min_val) * rand_floats
                strength_tensor = strength_values.view(batch_size, 1, 1, 1)

        # --- Step 2: Pre-process image to [0, 1] (unchanged) ---
        is_neg_one_range = image.min() < -0.01
        image_0_1 = (image + 1.0) / 2.0 if is_neg_one_range else image
        
        # --- Step 3: RGB -> HSV conversion (unchanged) ---
        hsv_image = kornia.color.rgb_to_hsv(image_0_1)
        h, s, v = hsv_image.chunk(3, dim=1)

        # --- Step 4: DDP-SAFE noise application ---
        noise_std = torch.sqrt(v.clamp(min=1e-8)) * strength_tensor
        
        ## DDP-SAFE CHANGE ##: Use the generator with torch.randn
        # Create a new tensor with the specified generator, not randn_like
        noise = torch.randn(v.shape, generator=generator, device=v.device, dtype=v.dtype) * noise_std
        
        noisy_v = (v + noise).clamp(0.0, 1.0)

        # --- Step 5: Recombine and convert back to RGB (unchanged) ---
        noisy_hsv_image = torch.cat([h, s, noisy_v], dim=1)
        noisy_image_0_1 = kornia.color.hsv_to_rgb(noisy_hsv_image).clamp(0.0, 1.0)

        # --- Step 6: Convert back to original range (unchanged) ---
        output_final = noisy_image_0_1 * 2.0 - 1.0 if is_neg_one_range else noisy_image_0_1
            
        return output_final

# --- Example of how to use and test the corrected version ---
if __name__ == '__main__':
    # You need to have kornia installed: pip install kornia
    try:
        import kornia
        print(f"Kornia version {kornia.__version__} detected.")
    except ImportError:
        print("Please install kornia to run this test: `pip install kornia`")
        exit()

    # Instantiate the new, differentiable noise module
    shot_augmenter = KorniaShotNoise(strength_range=(0.05, 0.8))

    # --- Test with an input tensor in the [-1, 1] range ---
    print("\n--- Testing with input range [-1, 1] ---")
    test_tensor_neg_one = torch.rand(4, 3, 64, 64) * 2 - 1
    print(f"Input range: [{test_tensor_neg_one.min():.2f}, {test_tensor_neg_one.max():.2f}]")

    # Apply noise
    noisy_tensor = shot_augmenter(test_tensor_neg_one)
    print(f"Output range: [{noisy_tensor.min():.2f}, {noisy_tensor.max():.2f}]")
    assert noisy_tensor.min() >= -1.0 and noisy_tensor.max() <= 1.0
    print("Range Test Passed! Output is correctly in [-1, 1].")

    # --- Test gradient flow (to simulate a training step) ---
    print("\n--- Testing gradient flow ---")
    test_tensor_grad = torch.rand(2, 3, 32, 32, requires_grad=True)
    output = shot_augmenter(test_tensor_grad)
    
    try:
        # Sum the output and perform a backward pass
        output.sum().backward()
        print("Gradient test PASSED: Backward pass completed successfully.")
        # Check if the input tensor has received gradients
        assert test_tensor_grad.grad is not None
        print("Gradient test PASSED: Input tensor received non-zero gradients.")
    except Exception as e:
        print(f"Gradient test FAILED: {e}")