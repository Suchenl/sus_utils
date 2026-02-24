from PIL import Image
from typing import Literal

def resize_image(image: Image.Image, size: int, upper_or_lower: Literal["upper", "lower"] = "upper") -> Image.Image:
    """
    Resize the input image to the specified size while maintaining aspect ratio.
    
    Args:
        image (PIL.Image.Image): The input image to be resized.
        size (int): The desired size for the shorter edge of the image.
        upper_or_lower (Literal["upper", "lower"]): Determines whether to resize based on the upper or lower edge of the image. 
            - "upper": Resize based on the upper edge (default).
            - "lower": Resize based on the lower edge.
    
    Returns:
        PIL.Image.Image: The resized image.
    """
    # Get original dimensions
    original_width, original_height = image.size

    # Determine the scaling factor based on the specified edge
    if upper_or_lower == "upper":
        scale_factor = size / min(original_width, original_height)
    elif upper_or_lower == "lower":
        scale_factor = size / max(original_width, original_height)
    else:
        raise ValueError("upper_or_lower must be either 'upper' or 'lower'")
    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image