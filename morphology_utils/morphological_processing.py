import cv2
import numpy as np

def morphological_opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Perform morphological opening on a binary mask.

    Opening = Erosion -> Dilation.

    Functionality:
    - Removes small white noise (tiny foreground spots) from the mask.
    - Smooths object boundaries by eliminating thin protrusions.
    - Preserves the overall shape of larger objects while cleaning up small artifacts.

    Args:
        mask (np.ndarray): Binary input mask (values 0 or 255).
        kernel_size (int): Size of the structuring element (default=3).

    Returns:
        np.ndarray: Processed mask after opening.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened


def morphological_closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Perform morphological closing on a binary mask.

    Closing = Dilation -> Erosion.

    Functionality:
    - Fills small black holes inside white objects (foreground regions).
    - Connects nearby white regions that are separated by small black gaps.
    - Preserves the overall object size while closing small holes and gaps.

    Args:
        mask (np.ndarray): Binary input mask (values 0 or 255).
        kernel_size (int): Size of the structuring element (default=3).

    Returns:
        np.ndarray: Processed mask after closing.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed
