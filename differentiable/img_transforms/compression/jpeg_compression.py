import torch
from .diff_jpeg import DiffJPEGCoding
from typing import *

class JpegCompression(torch.nn.Module):
    """
    Applies a differentiable approximation of JPEG compression.
    Requires the `diffjpeg` library.

    This implementation is DDP-safe.
    """
    def __init__(self, 
                 quality_range: Tuple[float, float] = None, 
                 quality_level: str = None,
                 per_image_randomization: bool = False): # In this implementation, 'False' or 'True' is equal in speed.
        super(JpegCompression, self).__init__()
        self.per_image_randomization = per_image_randomization
        # The DiffJPEG module is stateful, so we create it once.
        self.jpeg_compressor = DiffJPEGCoding(ste=False)

        quality_level_map = {"subtle": (75, 95), "moderate": (45, 75), "subtle and moderate": (45, 95), "strong": (10, 45)}
        if quality_range is None and quality_level is None:
            quality_level = "subtle and moderate"
        if quality_range is not None: self.quality_range = quality_range
        else: self.quality_range = quality_level_map[quality_level]
        mode = "Per-Image" if self.per_image_randomization else "Batch-Uniform"
        print(f"[JpegCompression] Quality Range: {self.quality_range}, Mode: {mode}")

    def forward(self, image: torch.Tensor, 
                quality: float = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        batch_size = image.size(0)
        
        # Pre-process image to [0, 255] range (unchanged)
        is_neg_one_range = image.min() < -0.01
        image_0_1 = (image + 1.0) / 2.0 if is_neg_one_range else image
        image_0_255 = self.preprocess(image_0_1)
        
        # --- DDP-SAFE Quality Tensor Generation ---
        if quality is not None:
            # Deterministic path
            quality_tensor = torch.full((batch_size, 1, 1, 1), quality, device=image.device, dtype=torch.float32)
        else:
            min_val, max_val = self.quality_range
            if not self.per_image_randomization:
                # "Batch-Uniform" random path
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                rand_float = torch.rand(1, generator=generator, device=image.device).item()
                rand_quality = min_val + (max_val - min_val) * rand_float
                quality_tensor = torch.full((batch_size, 1, 1, 1), rand_quality, device=image.device, dtype=torch.float32)
            else:
                # "Per-Image" random path
                ## DDP-SAFE CHANGE ##: Replace empty().uniform_
                rand_floats = torch.rand(batch_size, device=image.device, generator=generator)
                qualities = min_val + (max_val - min_val) * rand_floats
                quality_tensor = qualities.view(batch_size, 1, 1, 1).to(torch.float32)

        # Apply differentiable JPEG compression (unchanged)
        compressed_image_0_255 = self.jpeg_compressor(image_rgb=image_0_255, jpeg_quality=quality_tensor)
        
        # Post-process back to the original range (unchanged)
        image_back_to_0_1 = self.postprocess(compressed_image_0_255)
        output_final = image_back_to_0_1 * 2.0 - 1.0 if is_neg_one_range else image_back_to_0_1
        
        return output_final
    
    def preprocess(self, image_0_1: torch.Tensor) -> torch.Tensor:
        return torch.clamp(image_0_1 * 255, 0, 255)
    
    def postprocess(self, image_0_255: torch.Tensor) -> torch.Tensor:
        return torch.clamp(image_0_255 / 255, 0.0, 1.0)