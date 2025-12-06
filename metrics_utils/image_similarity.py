import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import Union, List

'''
Ready to do:
    - Implement the dinov3 similarity
'''
class ImageSimilarity(torch.nn.Module):
    """
    Supported metrics: 
        - clip: openai/clip-vit-base-patch32
        - openclip: laion/CLIP-ViT-L-14
        - dinov2: dinov2_vits14
        - dinov3: vit_small_patch16_224
        - psnr: Peak Signal-to-Noise Ratio
        - ssim: Structural Similarity Index Measure
        - lpips-alex: Perceptual Losses for Real-Time Style Transfer and Super-Resolution (AlexNet)
        - lpips-vgg: Perceptual Losses for Real-Time Style Transfer and Super-Resolution (VGG16)
        - phash: Perceptual Hash
    Input: [B,C,H,W] tensor, floating-point numbers, range [0,1]
    Output: [B] or [B,B] similarity matrix
    """
    def __init__(self, 
                 metric: str, 
                 device="cuda", 
                 compare_all: bool = False
                 ):
        """
        Args:
            metric (str): metric name
            device (str): device name
            compare_all (bool): whether to compare all images in 'x1' and 'x2'
            
        """
        super().__init__()
        self.metric = metric.lower()
        self.device = device
        self.compare_all = compare_all

        # -------------------- CLIP / OpenCLIP --------------------
        if self.metric in ["clip", "openclip"]:
            from transformers import CLIPModel, CLIPProcessor
            
            model_name = "openai/clip-vit-base-patch32" if self.metric=="clip" else "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.model.eval()
            self.processor = CLIPProcessor.from_pretrained(model_name)

        # -------------------- DINOv2 / DINOv3 --------------------
        elif self.metric in ["dinov2", "dinov3"]:
            from timm import create_model
            from transformers import AutoImageProcessor, AutoModel
            if self.metric == "dinov2":
                self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
            if self.metric == "dinov3":
                raise ValueError("Error! The similarity based on 'dinov3' has not yet been implemented !")
            self.model.eval()
            # ImageNet normalize
            self.transform = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        # -------------------- PSNR --------------------
        elif self.metric == "psnr":
            from . import PSNR
            self.model = PSNR(data_range=1.0, reduction='none')

        # -------------------- SSIM --------------------
        elif self.metric == "ssim":
            from . import SSIM
            self.model = SSIM(data_range=1.0, reduction='none')

        # -------------------- LPIPS --------------------
        elif self.metric in ["lpips-alex", "lpips-vgg"]:
            from lpips import LPIPS
            model_name = "alex" if self.metric=="lpips-alex" else "vgg"
            self.model = LPIPS(net=model_name).to(device).eval()

        # -------------------- pHash --------------------
        elif self.metric == "phash":
            from .torch_phash import pHash
            self.model = pHash(hash_size=8, image_size=32).to(device)
        else:
            raise ValueError(f"Unsupported metric {metric}")

    @torch.no_grad()
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x1, x2: [B, C, H, W] float tensor, [0,1]
        Return: [B] Similarity
        """
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        if valid_mask is not None:
            x1 = x1 * valid_mask
            x2 = x2 * valid_mask

        if self.metric in ["clip", "openclip"]:
            # Preprocess
            inputs1 = self.processor(images=x1, return_tensors="pt").to(self.device)
            inputs2 = self.processor(images=x2, return_tensors="pt").to(self.device)
            # Extract features
            features1 = self.model.get_image_features(**inputs1)
            features2 = self.model.get_image_features(**inputs2)
            # Normalization
            features1 = features1 / features1.norm(dim=-1, keepdim=True)  # 🔥 L2 normalize
            features2 = features2 / features2.norm(dim=-1, keepdim=True)  # 🔥 L2 normalize
            # Cosine similarity
            # [B, D] @ [B, D] -> [B, B] if compare_all else [B]
            sim = features1 @ features2.T if self.compare_all else torch.sum(features1 * features2, dim=-1)
            return sim
        
        elif self.metric in ["dinov2", "dinov3"]:
            # Preprocess
            inputs1 = self.processor(images=x1, return_tensors="pt").to(self.device)
            inputs2 = self.processor(images=x2, return_tensors="pt").to(self.device)
            # Extract features
            features1 = self.model(**inputs1).last_hidden_state.mean(dim=1)
            features2 = self.model(**inputs2).last_hidden_state.mean(dim=1)
            # Normalization
            features1 = features1 / features1.norm(dim=-1, keepdim=True)  # 🔥 L2 normalize
            features2 = features2 / features2.norm(dim=-1, keepdim=True)  # 🔥 L2 normalize
            # Cosine similarity
            # [B, D] @ [B, D] -> [B, B] if compare_all else [B]
            sim = features1 @ features2.T if self.compare_all else torch.sum(features1 * features2, dim=-1)
            return sim
        
        elif self.metric in ["psnr", "ssim", "lpips-alex", "lpips-vgg"]:
            # torchmetrics / lpips accept [B, C, H, W]
            # don't implement 'compare_all' for these metrics
            out = self.model(x1, x2)
            return out.view(out.shape[0]) 
        
        elif self.metric == "phash":
            bx1 = self.model(x1)
            bx2 = self.model(x2)
            if self.compare_all: sim = 1 - ((bx1.unsqueeze(1) ^ bx2.unsqueeze(0)).sum(dim=-1).float() / (self.model.hash_size**2)) # [B1, B2]
            else: sim = 1 - ((bx1 ^ bx2).sum(dim=-1).float() / (self.model.hash_size**2)) # [B]
            return sim
        
        else:
            raise ValueError(f"Unknown metric {self.metric}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from torchvision import transforms
    from PIL import Image
    x1_path = '/public/chenyuzhuo/MODELS/image_watermarking_models/UniAIF/benchmarks/images/human_faces/0001.png'
    x2_path = '/public/chenyuzhuo/MODELS/image_watermarking_models/UniAIF/benchmarks/images/human_faces/0001.png'
    device = 'cuda'
    x1 = transforms.ToTensor()(Image.open(x1_path)).unsqueeze(0).to(device)
    x2 = transforms.ToTensor()(Image.open(x1_path)).unsqueeze(0).to(device)

    # metrics = ["psnr", "ssim", "lpips-alex", "lpips-vgg", "phash"] 
    # metrics = ["clip", "openclip"]
    metrics = ["dinov2"]
    # metrics = ["dinov2"]
    for m in metrics:
        sim = ImageSimilarity(metric=m, device=device)
        out = sim(x1, x2)
        print(f"{m}: {out}")
