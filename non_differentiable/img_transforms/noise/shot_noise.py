import torch
from typing import *

import torch
from typing import *

import torch
from typing import *

class ShotNoise(torch.nn.Module):
    """
    Applies Shot Noise using a dynamically controlled physical simulation model.
    The 'strength' parameter directly modulates a physical property (effective photons),
    linking the data augmentation framework to a physical noise generation process.
    """
    def __init__(self, 
                 strength_range: Tuple[float, float] = None, # 
                 strength_level: str = None,
                 per_image_randomization: bool = False):  # In this implementation, 'False' or 'True' is equal in speed.
        super(ShotNoise, self).__init__()
        self.per_image_randomization = per_image_randomization

        # Strength will be used to dynamically calculate the effective number of photons.
        strength_level_map = {
            "subtle": (0.05, 0.2), "moderate": (0.2, 0.5),
            "subtle and moderate": (0.05, 0.5), "strong": (0.5, 1.0)
        }
        if strength_range is None and strength_level is None:
            strength_level = "subtle and moderate"
        
        if strength_range is not None:
            # Strength cannot be zero in this model, so we clip the lower bound.
            self.strength_range = (max(strength_range[0], 1e-6), strength_range[1])
        else:
            self.strength_range = strength_level_map[strength_level]

        # This represents the photon count for a hypothetical "strength of 1.0".
        # It serves as a base scale for the inverse relationship.
        self.base_photons_at_strength_one = 50.0

        mode = "Per-Image Randomization" if self.per_image_randomization else "Batch-Uniform (High Performance)"
        print(f"[ShotNoise (Dynamic Physical Model)] Initialized with strength range: {self.strength_range}, "
              f"Base Photons: {self.base_photons_at_strength_one}, Mode: {mode}")

    def forward(self, image: torch.Tensor, strength: float = None) -> torch.Tensor:
        # --- Step 1: Determine the control strength, handling strength=0 case ---
        if strength is not None and strength == 0.0:
            return image

        batch_size = image.size(0)
        if strength is not None:
            if not (0.0 <= strength <= 1.0): raise ValueError("Strength must be between 0.0 and 1.0.")
            strength_tensor = torch.full((1, 1, 1, 1), strength, device=image.device, dtype=image.dtype)
        else:
            min_val, max_val = self.strength_range
            if not self.per_image_randomization:
                rand_strength = torch.empty(1).uniform_(min_val, max_val).item()
                strength_tensor = torch.full((1, 1, 1, 1), rand_strength, device=image.device, dtype=image.dtype)
            else:
                strength_values = torch.empty(batch_size, device=image.device).uniform_(min_val, max_val)
                strength_tensor = strength_values.view(batch_size, 1, 1, 1)

        # --- Step 2: Normalize image to [0, 1] and convert to HSV space ---
        is_neg_one_range = image.min() < -0.01
        image_0_1 = (image + 1.0) / 2.0 if is_neg_one_range else image
        image_0_1 = torch.clamp(image_0_1, 0.0, 1.0)

        hsv_image = self._rgb_to_hsv_batched(image_0_1)
        h, s, v = hsv_image.chunk(3, dim=1)

        # --- Step 3: Dynamically calculate effective photons from strength ---
        # Using an epsilon for numerical stability is a good practice.
        effective_photons = self.base_photons_at_strength_one / (strength_tensor + 1e-8)
        
        # --- Step 4: Apply the physical simulation to the V (Value) channel ---
        mean_photons = v * effective_photons
        # detected_photons = torch.poisson(mean_photons)
        safe_mean_photons = mean_photons.clamp(min=0.0)
        detected_photons = torch.poisson(safe_mean_photons)
        noisy_v = detected_photons / (effective_photons + 1e-8)
        noisy_v = torch.clamp(noisy_v, 0.0, 1.0)

        # --- Step 5: Recombine channels and convert back to RGB ---
        noisy_hsv_image = torch.cat([h, s, noisy_v], dim=1)
        noisy_image_0_1 = self._hsv_to_rgb_batched(noisy_hsv_image)

        # --- Step 6: Convert back to the original range ---
        output_final = noisy_image_0_1 * 2.0 - 1.0 if is_neg_one_range else noisy_image_0_1
        
        return output_final
    
    @staticmethod
    def _rgb_to_hsv_batched(rgb: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        r, g, b = rgb.chunk(3, dim=1)
        max_c, _ = torch.max(rgb, dim=1, keepdim=True)
        min_c, _ = torch.min(rgb, dim=1, keepdim=True)
        delta = max_c - min_c
        v = max_c
        s = torch.where(max_c > 0, delta / max_c, torch.zeros_like(max_c))
        delta_safe = delta + eps
        rc = (max_c - r) / delta_safe
        gc = (max_c - g) / delta_safe
        bc = (max_c - b) / delta_safe
        h = torch.zeros_like(max_c)
        is_r_max = r == max_c
        is_g_max = g == max_c
        h[is_r_max] = (bc - gc)[is_r_max]
        h[is_g_max] = (2.0 + rc - gc)[is_g_max]
        h[~(is_r_max | is_g_max)] = (4.0 + gc - rc)[~(is_r_max | is_g_max)]
        h = (h / 6.0) % 1.0
        h = torch.where(delta == 0, 0.0, h)
        return torch.cat([h, s, v], dim=1)

    @staticmethod
    def _hsv_to_rgb_batched(hsv: torch.Tensor) -> torch.Tensor:
        h, s, v = hsv.chunk(3, dim=1)
        i = (h * 6.0).floor()
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i.long() % 6
        rgb = torch.zeros_like(hsv)
        rgb = torch.where(i == 0, torch.cat([v, t, p], dim=1), rgb)
        rgb = torch.where(i == 1, torch.cat([q, v, p], dim=1), rgb)
        rgb = torch.where(i == 2, torch.cat([p, v, t], dim=1), rgb)
        rgb = torch.where(i == 3, torch.cat([p, q, v], dim=1), rgb)
        rgb = torch.where(i == 4, torch.cat([t, p, v], dim=1), rgb)
        rgb = torch.where(i == 5, torch.cat([v, p, q], dim=1), rgb)
        rgb = torch.where(s == 0, torch.cat([v, v, v], dim=1), rgb)
        return rgb


# --- Example of how to use and test the corrected version ---
if __name__ == '__main__':
    shot_augmenter = ShotNoise(strength_level="strong")

    # --- Test with strength=0 (should return original image) ---
    print("\n--- Testing with strength=0 ---")
    zero_one_tensor = torch.rand(2, 3, 32, 32)
    output_no_noise = shot_augmenter(zero_one_tensor, strength=0.0)
    assert torch.allclose(zero_one_tensor, output_no_noise)
    print("Test Passed! strength=0 returns the original image.")

    # --- Test with strength=1 (should return fully noisy image) ---
    print("\n--- Testing with strength=1 ---")
    output_max_noise = shot_augmenter(zero_one_tensor, strength=1.0)
    assert not torch.allclose(zero_one_tensor, output_max_noise)
    print("Test Passed! strength=1 returns a modified image.")
    
    # --- Test range preservation with random strength ---
    print("\n--- Testing with input range [0, 1] ---")
    print(f"Input min: {zero_one_tensor.min():.2f}, Input max: {zero_one_tensor.max():.2f}")
    output1 = shot_augmenter(zero_one_tensor)
    print(f"Output min: {output1.min():.2f}, Output max: {output1.max():.2f}")
    assert output1.min() >= 0.0 and output1.max() <= 1.0
    print("Range Test Passed! Output is correctly in [0, 1].")