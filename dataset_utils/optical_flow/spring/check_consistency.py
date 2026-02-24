from differentiable.flow_utils.compose_flows import compose_forward_flow_sequence
from differentiable.flow_utils.warp_img_by_flow import forward_warp, backward_warp
from io_utils.flow_io import read_flo5, write_flo5
from io_utils.image_io import tensor_to_pil
from viz_utils.flow_viz import viz_flow_with_rgb, viz_flow_with_arrows
from differentiable.flow_utils import resize_flow
import os
import argparse
import torch
import numpy as np
import shutil
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def set_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset_dir", type=str, default="/public/chenyuzhuo/DATASETS/Optical_Flow_DATASETS/Spring/spring", help="Directory to read spring dataset")
    parse.add_argument("--set", type=str, default="train", help="train or test")
    parse.add_argument("--flow_direction", type=str, default='FW', help="[FW, BW]")
    parse.add_argument("--eye_pos", type=str, default="left", help="[left, right]")
    parse.add_argument("--img_h", type=int, default=540)
    parse.add_argument("--img_w", type=int, default=960)
    parse.add_argument("--skip_num", type=int, default=5)
    parse.add_argument("--compose_num", type=int, default=30)
    parse.add_argument("--save_warped", type=bool, default=True)
    parse.add_argument("--save_exp_name", type=str, default="preprocessed_h540w960")
    parse.add_argument("--check_consistency", type=bool, default=True)
    args = parse.parse_args()
    args.flow_dir_name = f"flow_{args.flow_direction}_{args.eye_pos}"
    args.frame_dir_name = f"frame_{args.eye_pos}"
    return args