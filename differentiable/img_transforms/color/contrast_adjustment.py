import torch
import torch.nn as nn
from typing import Tuple, Optional
import torchvision.transforms.functional as F

class ContrastAdjustment(torch.nn.Module):
    """
    Adjusts the contrast of an image tensor by wrapping the official, differentiable
    `torchvision.transforms.functional.adjust_contrast` function.
    
    This implementation provides separate controls for decreasing and increasing contrast,
    with a probabilistic choice between the two operations.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 down_factor_range: Optional[Tuple[float, float]] = (0.5, 0.99),
                 up_factor_range: Optional[Tuple[float, float]] = (1.1, 1.8),
                 down_prob: float = 0.5,
                 per_image_randomization: bool = False):
        """
        Args:
            down_factor_range (Tuple[float, float], optional): 
                Range for decreasing contrast, e.g., (0.5, 0.99). Must be < 1.
                If None, this operation is disabled.
            up_factor_range (Tuple[float, float], optional): 
                Range for increasing contrast, e.g., (1.1, 5). Must be > 1.
                If None, this operation is disabled.
            down_prob (float): 
                Probability of applying contrast decrease. The probability of 
                increasing contrast will be (1 - down_prob).
            per_image_randomization (bool): 
                If True, applies a different random factor to each image in the batch.
        """
        super(ContrastAdjustment, self).__init__()
        
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

        # Determine the effective probability if one range is disabled
        if self.down_factor_range is None:
            self.down_prob = 0.0
        if self.up_factor_range is None:
            self.down_prob = 1.0

        print(f"[ContrastAdjustment (Probabilistic)] Initialized.")
        if self.down_factor_range: print(f"  - Decrease Range: {self.down_factor_range}, Prob: {self.down_prob if self.up_factor_range else 1.0}")
        if self.up_factor_range: print(f"  - Increase Range: {self.up_factor_range}, Prob: {1.0 - self.down_prob if self.down_factor_range else 1.0}")

            
    def forward(self, image: torch.Tensor, 
                factor: float = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): The input image tensor (B, C, H, W).
            factor (float, optional): If provided, this specific factor will be used, overriding randomization.
        Returns:
            torch.Tensor: The contrast-adjusted image tensor.
        """
        if factor is not None:
            if factor < 0: raise ValueError("Factor must be a non-negative number.")
            return F.adjust_contrast(image, contrast_factor=factor)

        # --- DDP-SAFE Randomization with Probabilistic Choice ---
        if not self.per_image_randomization:
            # "Batch-Uniform" random path
            ## DDP-SAFE CHANGE ##: Replace random.random()
            choice_rand = torch.rand(1, generator=generator).item()
            factor_range_to_use = self.down_factor_range if choice_rand < self.down_prob else self.up_factor_range

            ## DDP-SAFE CHANGE ##: Replace empty().uniform_
            min_f, max_f = factor_range_to_use
            rand_float = torch.rand(1, generator=generator).item()
            uniform_factor = min_f + (max_f - min_f) * rand_float
            
            return F.adjust_contrast(image, contrast_factor=uniform_factor)
        else:
            # "Per-Image" random path
            batch_size = image.size(0)
            
            ## DDP-SAFE CHANGE ##: Replace rand() and empty().uniform_
            # Decide for each image in the batch using the generator
            choices = torch.rand(batch_size, device=image.device, generator=generator) < self.down_prob
            
            # Generate random factors from both ranges for the entire batch using the generator
            min_down, max_down = self.down_factor_range
            min_up, max_up = self.up_factor_range
            
            rand_floats_down = torch.rand(batch_size, device=image.device, generator=generator)
            rand_floats_up = torch.rand(batch_size, device=image.device, generator=generator)

            down_factors = min_down + (max_down - min_down) * rand_floats_down
            up_factors = min_up + (max_up - min_up) * rand_floats_up
            
            # Use torch.where to select the final factors based on the choice mask
            final_factors = torch.where(choices, down_factors, up_factors)
            
            # This loop can be vectorized for better performance
            # However, for DDP-safety, the parameter generation is the critical part, which is now fixed.
            adjusted_images = []
            for i in range(batch_size):
                adjusted_slice = F.adjust_contrast(image[i:i+1], contrast_factor=final_factors[i].item())
                adjusted_images.append(adjusted_slice)
            
            return torch.cat(adjusted_images, dim=0)
    
# --- Example Usage to Verify the Correctness ---
if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import ToTensor, ToPILImage
    import torchvision.transforms.functional as F
    import os

    contrast_factor = 0.1
    save_dir = "/public/chenyuzhuo/MODELS/image_watermarking_models/Image_Motion_Pred-dev/outputs/check_distortion_simulator/Check_contrast_adjustment/"
    os.makedirs(save_dir, exist_ok=True)
    
    def save_tensor_image(tensor, filename):
        img_tensor = tensor.squeeze(0)
        # Handle both [0, 1] and [-1, 1] for saving
        if img_tensor.min() < -0.01:
            img_tensor = (img_tensor + 1.0) / 2.0
        ToPILImage()(img_tensor.clamp(0, 1)).save(filename)
        print(f"Saved image to {filename}")

    img = Image.open("/public/chenyuzhuo/MODELS/image_watermarking_models/Image_Motion_Pred-dev/outputs/check_distortion_simulator/test_image.jpg").convert("RGB") # Change path
    input_tensor_pil = ToTensor()(img).unsqueeze(0) # This is [0, 1] range
    
    print("\n--- Testing with REAL image ---")
    
    # Our implementation
    adjuster = ContrastAdjustmentKornia()
    our_output = adjuster(input_tensor_pil, factor=contrast_factor)
    save_tensor_image(our_output, save_dir + "/output_our_final_contrast.png")

    # Compare with torchvision's standard implementation
    torchvision_output = F.adjust_contrast(input_tensor_pil, contrast_factor=contrast_factor)
    save_tensor_image(torchvision_output, save_dir + "/output_torchvision_contrast.png")
    
    # Check if the results are identical
    if torch.allclose(our_output, torchvision_output, atol=1e-6):
        print("\nSUCCESS: Our implementation's output is identical to torchvision's standard implementation!")
    else:
        print("\nFAILURE: Outputs do not match.")
    print('Error: ', torch.abs(our_output - torchvision_output).mean() * 255)
