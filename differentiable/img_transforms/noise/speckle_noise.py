import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
import math

class SpeckleNoise(nn.Module):
    """
    Adds Speckle Noise with a user-friendly, physically-motivated model.
    
    The user specifies the desired `speckle_size` in pixels, which is then
    internally converted to a Gaussian `sigma` to generate isotropic (circular)
    speckles. The implementation uses a highly-efficient separable blur.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 intensity_range: Tuple[float, float] = None, 
                 intensity_level: str = None,
                 speckle_size_range: Tuple[int, int] = (3, 13), # User-friendly kernel size in pixels
                 per_image_randomization: bool = True):
        super(SpeckleNoise, self).__init__()
        self.per_image_randomization = per_image_randomization

        # --- Intensity Configuration ---
        intensity_level_map = {
            "subtle": (0.05, 0.15), "moderate": (0.15, 0.3),
            "subtle and moderate": (0.05, 0.25), "strong": (0.3, 0.5)
        }
        if intensity_range is None and intensity_level is None:
            intensity_level = "subtle and moderate"
        
        if intensity_range is not None:
            self.intensity_range = intensity_range
        else:
            self.intensity_range = intensity_level_map[intensity_level]
        
        # --- Speckle Size Configuration (user-facing) ---
        if not (isinstance(speckle_size_range, tuple) and len(speckle_size_range) == 2 and
                speckle_size_range[0] <= speckle_size_range[1] and speckle_size_range[0] >= 3):
            raise ValueError("speckle_size_range must be a tuple of two ints (min, max) with min >= 3.")
        # Ensure kernel sizes are odd
        self.speckle_size_range = (
            speckle_size_range[0] | 1, # Bitwise OR 1 makes any int odd
            speckle_size_range[1] | 1
        )

        mode = "Per-Image Randomization" if self.per_image_randomization else "Batch-Uniform"
        print(f"[SpeckleNoise (Corrected Model v3)] Initialized with Intensity Range: {self.intensity_range}, "
              f"Speckle Size (Kernel) Range: {self.speckle_size_range}px, Mode: {mode}")

    def forward(self, image: torch.Tensor, intensity: float = None, kernel_size: int = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Applies speckle noise to the image.

        Args:
            image (torch.Tensor): The input image tensor.
            intensity (float, optional): If provided, overrides the random intensity range. Defaults to None.
            kernel_size (int, optional): If provided, overrides the random speckle size range. Defaults to None.
        
        Returns:
            torch.Tensor: The image with speckle noise applied.
        """
        # --- Step 1: Normalize image to [0, 1] for processing ---
        is_neg_one_range = image.min() < -0.01
        image_0_1 = (image + 1.0) / 2.0 if is_neg_one_range else image
        image_0_1 = torch.clamp(image_0_1, 0.0, 1.0)
        
        b, c, h, w = image.shape
        if c == 0: return image # Handle empty tensors

        # --- Step 2: DDP-SAFE parameter generation ---
        min_i, max_i = self.intensity_range
        
        # Determine intensity tensor
        if intensity is not None:
            intensity_tensor = torch.full((b, 1, 1, 1), intensity, device=image.device, dtype=image.dtype)
        else:
            if self.per_image_randomization and b > 1:
                # "Per-Image" random intensity
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                rand_floats = torch.rand(b, device=image.device, generator=generator)
                intensity_values = min_i + (max_i - min_i) * rand_floats
                intensity_tensor = intensity_values.view(b, 1, 1, 1)
            else:
                # "Batch-Uniform" random intensity
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                rand_float = torch.rand(1, device=image.device, generator=generator).item()
                final_intensity = min_i + (max_i - min_i) * rand_float
                intensity_tensor = torch.full((b, 1, 1, 1), final_intensity, device=image.device, dtype=image.dtype)
        
        # Determine kernel size (one for the whole batch for performance)
        if kernel_size is not None:
            if not isinstance(kernel_size, int) or kernel_size < 3:
                raise ValueError(f"Provided kernel_size must be an int >= 3, but got {kernel_size}.")
            final_kernel_size = kernel_size | 1
        else:
            min_k, max_k = self.speckle_size_range
            ## DDP-SAFE CHANGE ##: Replace torch.randint
            final_kernel_size = torch.randint(min_k // 2, max_k // 2 + 1, (1,), generator=generator, device=image.device).item() * 2 + 1
        
        sigma = final_kernel_size / 6.0

        # --- Step 3: DDP-SAFE noise pattern generation ---
        ## DDP-SAFE CHANGE ##: Replace randn_like
        noise = torch.randn(image_0_1.shape, device=image.device, dtype=image.dtype, generator=generator)
        
        # Separable Gaussian Blur (unchanged, as it's deterministic)
        gaussian_kernel_1d = self._create_1d_gaussian_kernel(final_kernel_size, sigma, image.device, image.dtype)
        horizontal_kernel = gaussian_kernel_1d.view(1, 1, 1, final_kernel_size).repeat(c, 1, 1, 1)
        vertical_kernel = gaussian_kernel_1d.view(1, 1, final_kernel_size, 1).repeat(c, 1, 1, 1)
        pad_h = (final_kernel_size - 1) // 2
        noise_padded_h = F.pad(noise, (pad_h, pad_h, 0, 0))
        horizontal_blur = F.conv2d(noise_padded_h, horizontal_kernel, padding='valid', groups=c)
        blur_padded_v = F.pad(horizontal_blur, (0, 0, pad_h, pad_h))
        blurred_noise = F.conv2d(blur_padded_v, vertical_kernel, padding='valid', groups=c)
        
        std_per_image = torch.std(blurred_noise, dim=(-3, -2, -1), keepdim=True)
        normalized_noise = torch.where(std_per_image > 1e-6, blurred_noise / std_per_image, blurred_noise)

        # --- Step 4: Apply multiplicative noise (unchanged) ---
        noisy_image_0_1 = image_0_1 * (1 + normalized_noise * intensity_tensor)
        noisy_image_0_1 = torch.clamp(noisy_image_0_1, 0.0, 1.0)

        # --- Step 5: Convert back to original range (unchanged) ---
        output_final = noisy_image_0_1 * 2.0 - 1.0 if is_neg_one_range else noisy_image_0_1
        
        return output_final

    @staticmethod
    def _create_1d_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.float32) -> torch.Tensor:
        """Creates a 1D Gaussian kernel."""
        ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=dtype)
        kernel = torch.exp(-ax.pow(2) / (2 * sigma**2))
        return kernel / kernel.sum()

# --- Example of how to use and test the corrected version ---
if __name__ == '__main__':
    print("\n--- Running Final Corrected SpeckleNoise Test ---")
    
    # Use a batch size > 1 and channels = 3 to properly test the grouped convolution
    batch_size = 4
    channels = 3
    height, width = 256, 256
    
    # Initialize the module
    speckle_augmenter = SpeckleNoise(
        intensity_level="strong", 
        speckle_size_range=(7, 15), 
        per_image_randomization=True
    )
    
    # Create a dummy input tensor
    input_tensor = torch.ones(batch_size, channels, height, width) * 0.5 # A gray image
    
    # Apply the transformation
    try:
        output_tensor = speckle_augmenter(input_tensor)
        print("Test Passed! SpeckleNoise executed successfully with a batch of images.")
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Output tensor shape: {output_tensor.shape}")
        
        # Verify that the output is different from the input
        assert not torch.allclose(input_tensor, output_tensor)
        print("Output is different from input.")
        # Verify the output range is preserved
        assert output_tensor.min() >= 0.0 and output_tensor.max() <= 1.0
        print("Output range is preserved [0, 1].")

    except Exception as e:
        print(f"Test Failed! An error occurred: {e}")
        # Add traceback for easier debugging
        import traceback
        traceback.print_exc()