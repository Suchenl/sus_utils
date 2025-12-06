import numpy as np
from PIL import Image
import torch

class MaskSimilarity:
    """
    Compute similarity metrics between two binary masks.

    Supported metrics:
        - IoU (Intersection over Union)
        - Dice coefficient
        - Precision
        - Recall
    """

    def __init__(self, eps=1e-8, threshold=0.5):
        """
        Args:
            eps (float): Small constant to avoid division by zero.
            threshold (float): Binarization threshold for masks.
        """
        self.eps = eps
        self.threshold = threshold

    def _to_numpy(self, mask):
        """
        Convert a mask to a NumPy array (float32, range [0,1]).

        Supports:
            - PIL.Image
            - torch.Tensor
            - np.ndarray
        """
        if isinstance(mask, Image.Image):
            mask = np.array(mask, dtype=np.float32)
        elif isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy().astype(np.float32)
        elif isinstance(mask, np.ndarray):
            mask = mask.astype(np.float32)
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        # If RGB, convert to grayscale
        if mask.ndim == 3:
            mask = mask.mean(axis=-1)

        # Normalize to [0, 1]
        if mask.max() > 1.01:
            mask = mask / 255.0

        # Binarize
        mask = (mask > self.threshold).astype(np.float32)
        return mask

    def __call__(self, mask_pred, mask_gt):
        """
        Compute mask similarity metrics.

        Args:
            mask_pred: predicted mask (PIL.Image, np.ndarray, or torch.Tensor)
            mask_gt: ground truth mask (PIL.Image, np.ndarray, or torch.Tensor)
        Returns:
            dict: {'IoU': float, 'Dice': float, 'Precision': float, 'Recall': float}
        """
        mask_pred = self._to_numpy(mask_pred)
        mask_gt = self._to_numpy(mask_gt)

        intersection = np.sum(mask_pred * mask_gt)
        union = np.sum(mask_pred) + np.sum(mask_gt) - intersection

        iou = intersection / (union + self.eps)
        dice = 2 * intersection / (np.sum(mask_pred) + np.sum(mask_gt) + self.eps)
        precision = intersection / (np.sum(mask_pred) + self.eps)
        recall = intersection / (np.sum(mask_gt) + self.eps)

        return {
            "IoU": iou,
            "Dice": dice,
            "Precision": precision,
            "Recall": recall
        }
