import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class pHash(torch.nn.Module):
    def __init__(self, hash_size=8, image_size=32, device="cuda"):
        super().__init__()
        self.hash_size = hash_size
        self.image_size = image_size
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W], float32 [0,1]
        return: [B,B] similarity matrix
        """
        B, C, H, W = x.shape
        # Grayscale conversion
        x_gray = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
        # x_gray = 0.2126*x[:,0:1] + 0.7152*x[:,1:2] + 0.0722*x[:,2:3]
        # resize 
        x_gray = F.interpolate(x_gray, size=(self.image_size,self.image_size), 
                               mode='bicubic', align_corners=False, antialias=True)
        # Remove the mean
        x_gray = x_gray - x_gray.mean(dim=[2,3], keepdim=True)
        # DCT approx
        fx = torch.fft.fft2(x_gray).real[:, :, :self.hash_size, :self.hash_size].reshape(B, -1)
        # Binarization
        bx = (fx > fx.mean(dim=1, keepdim=True)).to(torch.uint8)
        return bx

if __name__ == "__main__":
    x = torch.randn(1,3,256,256)
    ph = pHash()
    print(ph(x).shape)
