import torch
import numpy as np
import argparse
import os
import h5py
import cv2

PREFERRED_RESOLUTIONS = [ # (W, H)
    (4096, 2160), # 4K (Standard) - 512: 270 ~= 17:9
    (3840, 2160), # 4K (UHDTV) - 16:9
    (2560, 1440), # 2K - 16:9
    (1920, 1080), # 1080p (Full HD) - 16:9
    (1280, 720), # 720p (HD) - 16:9
    (720, 480), # 480p (SD) ~ 3:2
    (854, 480), # 480p (SD) ~ 16:9
    (640, 480), # 480p (SD) ~ 12:9 = 4:3
    (640, 360), # 360p (Fluent) - 16:9
    (320, 240), # 240p (Fluent) - 12:9 = 4:3
    ]

def read_flo5(read_path):
    with h5py.File(read_path, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {read_path} does not have a 'flow' key. Is this a valid flo5 file?")
        flow = f["flow"][()]
    return flow

def writeFlo5File(flow, save_path):
    with h5py.File(save_path, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)

from torchvision.utils import flow_to_image
def flow_to_image_torch(flow: torch.Tensor) -> torch.Tensor:
    flow_img_tensor = flow_to_image(flow)
    flow_img_np = np.transpose(flow_img_tensor.numpy(), [1, 2, 0])
    return flow_img_np

import torch.nn.functional as F
def resize_flow(flow, H, W) -> torch.Tensor: 
    if isinstance(flow, np.ndarray):
        shape = flow.shape
        ori_H, ori_W = shape[0], shape[1]
        flow = np.transpose(flow, [2, 0, 1])
        flow = torch.from_numpy(flow).to(dtype=torch.float32).unsqueeze(0)

        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)[0]
        # W_scale = W / ori_W  # u scale factor
        # H_scale = H / ori_H  # v scale factor
        # scale_factors = torch.tensor([W_scale, H_scale], dtype=torch.float32).to(flow.device)
        # flow = flow * scale_factors[:, None, None]
        flow_resize_np = np.transpose(flow.numpy(), [1, 2, 0])
        print(f"--- Transform flow (np.ndarray with shape of {shape}) to flow (np.ndarray with shape of {flow_resize_np.shape}) ---")

    return flow_resize_np

def resize_single_file(flow_path, save_path, H, W):
    flow_np = read_flo5(flow_path)
    orig_dtype = flow_np.dtype
    occlusion_mask = np.isnan(flow_np[..., 0]) | np.isnan(flow_np[..., 1])
    flow_np = np.nan_to_num(flow_np, nan=0)
    flow_resize_np = resize_flow(flow_np, H, W)

    occlusion_mask = occlusion_mask.astype(np.float32)
    occlusion_mask_resized = cv2.resize(
        occlusion_mask,
        (W, H),
        interpolation=cv2.INTER_LINEAR
    ) >= 0.5
    # set occlusion mask to nan
    flow_resize_np[occlusion_mask_resized] = np.nan
    # save flow 
    writeFlo5File(flow_resize_np.astype(orig_dtype), save_path)
    
def set_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset_dir", type=str, default="/public/chenyuzhuo/DATASETS/Optical_Flow_DATASETS/Spring/spring", help="Directory to read spring dataset")
    parse.add_argument("--set", type=str, default="train", help="train or test")
    parse.add_argument("--flow_dir_name", type=str, default="flow_BW_left", help="'flow_BW_left' or 'flow_FW_left'")
    # parse.add_argument("--frame_dir_name", type=str, default="frame_left")
    parse.add_argument("--img_h", type=int, default=1080)
    parse.add_argument("--img_w", type=int, default=1920)
    args = parse.parse_args()
    return args

def main(args):
    set_dir = os.path.join(args.dataset_dir, args.set)
    scene_list = os.listdir(set_dir)
    scene_list.sort(key=lambda x: int(x))
    for scene in scene_list:
        scene_dir = os.path.join(set_dir, scene)
        # preprocess flows
        flow_dir = os.path.join(scene_dir, args.flow_dir_name)
        if not os.path.exists(flow_dir + '_ori'):
            os.rename(flow_dir, flow_dir + '_ori')
        save_dir = flow_dir
        flow_dir = flow_dir + '_ori'
        # save_dir = os.path.join(args.dataset_dir, args.set + f'_h{args.img_h}w{args.img_w}', scene, args.flow_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        flow_list = os.listdir(flow_dir)
        for flow_name in flow_list:
            flow_path = os.path.join(flow_dir, flow_name)
            save_path = os.path.join(save_dir, flow_name)
            print(f"Preprocessing: {flow_path}")
            if os.path.exists(save_path):
                print(f"The file exists: {save_path}, so skip it")
                continue
            else:
                resize_single_file(flow_path, save_path, H=args.img_h, W=args.img_w)
            
if __name__ == "__main__":
    args = set_args()
    main(args)
            

    
    

