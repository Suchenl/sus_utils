def dice_loss(pred, target, smooth=1e-6):
    # pred: [B, C, H, W]
    # target: [B, C, H, W]
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    # Calculate Dice coefficient
    dice_coeff = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice_coeff   # dice_loss = 1 - dice_coeff