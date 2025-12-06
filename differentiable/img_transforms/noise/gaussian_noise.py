import torch
from typing import Tuple, Optional

class GaussianNoise(torch.nn.Module):
    """
    Adds Gaussian Noise to an image tensor.
    
    This robust implementation first normalizes the input image to a standard [0, 1] range,
    applies noise with a consistent intensity meaning, and then converts the image back
    to its original range ([0, 1] or [-1, 1]). This ensures predictable behavior
    regardless of the input data's format.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 intensity_range: Optional[Tuple[float, float]] = None, 
                 intensity_level: str = None,
                 per_image_randomization: bool = False): # In this implementation, 'False' or 'True' is equal in speed.
        super(GaussianNoise, self).__init__()
        self.per_image_randomization = per_image_randomization

        # --- Intensity level mapping ---
        intensity_level_map = { 
            "subtle": (0.01, 0.05), 
            "moderate": (0.05, 0.1), 
            "subtle and moderate": (0.01, 0.1), 
            "strong": (0.1, 0.2) 
        }
        
        if intensity_range is None and intensity_level is None:
            intensity_level = "subtle and moderate"
        
        if intensity_range is not None:
            self.intensity_range = intensity_range
        else:
            self.intensity_range = intensity_level_map[intensity_level]

        mode = "Per-Image Randomization" if self.per_image_randomization else "Batch-Uniform (High Performance)"
        print(f"[GaussianNoise (Robust)] Initialized with intensity range: {self.intensity_range}, Mode: {mode}")
            
    def forward(self, image: torch.Tensor, 
                intensity: float = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        
        # Step 1: Normalize to [0, 1] (unchanged)
        is_neg_one_range = image.min() < -0.01
        image_0_1 = (image + 1.0) / 2.0 if is_neg_one_range else image
        batch_size = image.size(0)

        # --- Step 2: DDP-SAFE noise intensity generation ---
        if intensity is not None:
            # Deterministic path
            if intensity < 0: raise ValueError("Intensity must be a non-negative number.")
            intensity_tensor = torch.full((1, 1, 1, 1), intensity, device=image.device, dtype=image.dtype)
        else:
            min_val, max_val = self.intensity_range
            if not self.per_image_randomization:
                # "Batch-Uniform" random path
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                rand_float = torch.rand(1, generator=generator, device=image.device).item()
                rand_intensity = min_val + (max_val - min_val) * rand_float
                intensity_tensor = torch.full((1, 1, 1, 1), rand_intensity, device=image.device, dtype=image.dtype)
            else:
                # "Per-Image" random path
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                rand_floats = torch.rand(batch_size, device=image.device, generator=generator)
                intensity_values = min_val + (max_val - min_val) * rand_floats
                intensity_tensor = intensity_values.view(batch_size, 1, 1, 1)

        # --- Step 3: DDP-SAFE noise generation and application ---
        ## DDP-SAFE CHANGE ##: Use the generator with torch.randn
        # Create a new tensor with the same shape and specified generator
        noise = torch.randn(image_0_1.shape, generator=generator, device=image.device, dtype=image.dtype)
        
        adjusted_noise = noise * intensity_tensor
        noisy_image_0_1 = image_0_1 + adjusted_noise
        noisy_image_0_1 = torch.clamp(noisy_image_0_1, 0.0, 1.0)

        # --- Step 4: Convert back to original range (unchanged) ---
        output_final = noisy_image_0_1 * 2.0 - 1.0 if is_neg_one_range else noisy_image_0_1
        
        return output_final

# --- Example of how to use the robust version ---
if __name__ == '__main__':
    noise_augmenter = GaussianNoise(intensity_level="strong", per_image_randomization=True)

    # --- Test with an image in [0, 1] range ---
    print("\n--- Testing with input range [0, 1] ---")
    zero_one_tensor = torch.rand(2, 3, 32, 32)
    print(f"Input min: {zero_one_tensor.min():.2f}, Input max: {zero_one_tensor.max():.2f}")
    output1 = noise_augmenter(zero_one_tensor)
    print(f"Output min: {output1.min():.2f}, Output max: {output1.max():.2f}")
    assert output1.min() >= 0.0 and output1.max() <= 1.0
    print("Test Passed! Output is correctly in [0, 1].")

    # --- Test with an image in [-1, 1] range ---
    print("\n--- Testing with input range [-1, 1] ---")
    neg_one_tensor = torch.rand(2, 3, 32, 32) * 2.0 - 1.0
    print(f"Input min: {neg_one_tensor.min():.2f}, Input max: {neg_one_tensor.max():.2f}")
    output2 = noise_augmenter(neg_one_tensor)
    print(f"Output min: {output2.min():.2f}, Output max: {output2.max():.2f}")
    assert output2.min() >= -1.0 and output2.max() <= 1.0
    print("Test Passed! Output is correctly in [-1, 1].")