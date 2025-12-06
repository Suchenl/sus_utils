# Blur-related transforms
from .blur import (
    defocus_blur,
    downsample_blur,
    gaussian_blur, 
    median_filter, 
    motion_blur, 
    zoom_blur,
)

# Color-related transforms  
from .color import (
    brightness_adjustment, 
    contrast_adjustment, 
    saturation_adjustment,
)

# Compression-related transforms
from .compression import (
    jpeg_compression,
)

# Geometry-related transforms
from .geometry import (
    elastic_transform,
)

# Noise-related transforms
from .noise import (
    gaussian_noise, 
    impulse_noise,
    salt_and_pepper_noise,
    shot_noise,
    speckle_noise, 
)

# Other transforms
from .others import (
    pixelation,
    upsample_artifacts,
    moire_pattern,
)

MODULE_MAP = {
    # Blur-related transforms
    'defocus_blur': defocus_blur.DefocusBlur,
    'downsample_blur': downsample_blur.DownsampleBlur,
    'gaussian_blur': gaussian_blur.GaussianBlur,
    'median_filter': median_filter.MedianFilter,
    'motion_blur': motion_blur.MotionBlur,
    'zoom_blur': zoom_blur.ZoomBlur,
    
    # Color-related transforms
    'brightness_adjustment': brightness_adjustment.BrightnessAdjustment,
    'contrast_adjustment': contrast_adjustment.ContrastAdjustment,
    'saturation_adjustment': saturation_adjustment.SaturationAdjustment,
    
    # Compression-related transforms
    'jpeg_compression': jpeg_compression.JpegCompression,

    # Geometry-related transforms
    'elastic_transform': elastic_transform.ElasticTransform,
    
    # Noise-related transforms
    'gaussian_noise': gaussian_noise.GaussianNoise,
    'impulse_noise': impulse_noise.ImpulseNoise,
    'salt_and_pepper_noise': salt_and_pepper_noise.SaltAndPepperNoise, 
    'shot_noise': shot_noise.ShotNoise,
    'speckle_noise': speckle_noise.SpeckleNoise,

    # Other transforms
    'moire_pattern': moire_pattern.MoirePattern,
    'pixelation': pixelation.Pixelation,
    'upsample_artifacts': upsample_artifacts.UpsampleArtifacts,
}

def create_transform(transform_name, **kwargs):
    """Factory function to create transform instances."""
    if transform_name not in MODULE_MAP:
        raise ValueError(f"Unknown transform: {transform_name}. Available: {list(MODULE_MAP.keys())}")
    return MODULE_MAP[transform_name](**kwargs)

# Provide a list grouped by category
BLUR_TRANSFORMS = ['defocus_blur', 'downsample_blur', 'gaussian_blur', 'median_filter', 'motion_blur', 'zoom_blur']
COLOR_TRANSFORMS = ['brightness_adjustment', 'contrast_adjustment', 'saturation_adjustment']
COMPRESSION_TRANSFORMS = ['jpeg_compression']
GEOMETRY_TRANSFORMS = ['elastic_transform']
NOISE_TRANSFORMS = ['gaussian_noise', 'impulse_noise', 'salt_pepper_noise', 'shot_noise', 'speckle_noise']
OTHER_TRANSFORMS = ['moire_pattern', 'pixelation', 'upsample_artifacts']

ALL_TRANSFORMS = NOISE_TRANSFORMS + COLOR_TRANSFORMS + BLUR_TRANSFORMS + COMPRESSION_TRANSFORMS + OTHER_TRANSFORMS