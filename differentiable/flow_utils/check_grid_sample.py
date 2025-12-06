import torch
import torch.nn.functional as F
    
def main(align_corners=False, device='cpu', dtype=torch.float32):
    B, C = 1, 1
    img_H, img_W = 32, 32
    H, W = 64, 64
    
    ys = torch.linspace(0, img_H-1, img_H, device=device, dtype=dtype)
    xs = torch.linspace(0, img_W-1, img_W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # HxW
    img = (grid_x + grid_y).expand(B, C, img_H, img_W)

    ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # HxW
    base_grid = torch.stack((grid_x, grid_y), dim=0).expand(B, 2, H, W)  # Bx2xHxW

    p = base_grid
    # normalize pixel coords to [-1,1] for grid_sample
    p_x, p_y = p[:,0], p[:,1]

    print('base_grid:', base_grid, "\n", base_grid.shape)
    print("p_x:", p_x, "\n", p_x.shape)
    print("p_y:", p_y, "\n", p_y.shape)
    
    if align_corners:
        # If True, the range [-1, 1] corresponds to the pixel centers [0, W-1] or [0, H-1].
        # If set to True, -1 and 1 are considered the center of the corner pixels, rather than the corners of the image.
        p_xn = (p_x / (W-1)) * 2 - 1
        p_yn = (p_y / (H-1)) * 2 - 1
    else:
        # If False, the range [-1, 1] corresponds to the pixel edges [-0.5, (W-1)+0.5] or [-0.5, (H-1)+0.5] = 
        # [-0.5, W-0.5] or [-0.5, H-0.5].
        p_xn = ((p_x+0.5) / W) * 2 - 1
        p_yn = ((p_y+0.5) / H) * 2 - 1
        
        # p_xn = (p_x / (W-1)) * 2 - 1
        # p_yn = (p_y / (H-1)) * 2 - 1

        # p_xn = (p_x / W) * 2 - 1
        # p_yn = (p_y / H) * 2 - 1
        
    print("p_xn:", p_xn)
    print("p_yn:", p_yn)
    sample_grid = torch.stack((p_xn, p_yn), dim=-1)  # BxHxWx2, format required by grid_sample
        
    sampled = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)

    # mse = (img - sampled).pow(2).mean()
    # print('mse:', mse)
    print("img:\n", img)
    print("sampled:\n", sampled)

if __name__ == '__main__':
    main()