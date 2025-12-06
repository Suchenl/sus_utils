import torch
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.io_utils import mask_to_pil
from utils.viz_utils import (
    viz_flow_with_rgb, 
    viz_flow_with_arrows
    )

def viz_motions(motions, checkpoints_savedir, save_dir_name=None, suffix="", epoch_num=0, step_num=0, split_set="train", val_step=None, viz_with="rgb",
                background_images=None, arrows_step=16):
    if save_dir_name is not None:
        save_dir = Path(checkpoints_savedir) / split_set / save_dir_name
    else:
        save_dir = Path(checkpoints_savedir) / split_set
    os.makedirs(save_dir, exist_ok=True)
    for i, motion_i in enumerate(motions):
        motion_i_rgb = viz_flow_with_rgb(motion_i)
        background_image = background_images[i] if background_images is not None else None
        motion_i_arrows = viz_flow_with_arrows(motion_i, background_image=background_image, step=arrows_step)
        if split_set == "train":
            save_motion_i_rgb_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_sample-{i:02d}_rgb-{suffix}.png")
            save_motion_i_arrows_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_sample-{i:02d}_arrows-{suffix}.png")
        elif split_set == "val":
            save_motion_i_rgb_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_valstep-{val_step}_sample-{i:02d}_rgb-{suffix}.png")
            save_motion_i_arrows_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_valstep-{val_step}_sample-{i:02d}_arrows-{suffix}.png")

        if viz_with == "rgb":
            motion_i_rgb.save(save_motion_i_rgb_path)
        elif viz_with == "arrows":
            motion_i_arrows.save(save_motion_i_arrows_path)
        elif viz_with == "both":
            motion_i_rgb.save(save_motion_i_rgb_path)
            motion_i_arrows.save(save_motion_i_arrows_path)
        else:
            raise ValueError(f"viz_with must be one of ['rgb', 'arrows', 'both'], but got {viz_with}")
                
def viz_tensor_to_grey(tensors, checkpoints_savedir, save_dir_name=None, suffix="", epoch_num=0, step_num=0, split_set="train", val_step=None, binarize_thre=None):
    if save_dir_name is not None:
        save_dir = Path(checkpoints_savedir) / split_set / save_dir_name
    else:
        save_dir = Path(checkpoints_savedir) / split_set
    os.makedirs(save_dir, exist_ok=True)
    for i, tensor_i in enumerate(tensors):
        if binarize_thre is not None:
            tensor_i = (tensor_i > binarize_thre).float()
        if tensor_i.ndim == 3 and tensor_i.shape[0] == 1:
            tensor_i = tensor_i.squeeze(0)
        # print(f"tensor_i.shape: {tensor_i.shape}")
        tensor_i_pil = mask_to_pil(tensor_i)
        if split_set == "train":
            save_tensor_i_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_sample-{i:02d}-{suffix}.png")
        elif split_set == "val":
            save_tensor_i_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_valstep-{val_step}_sample-{i:02d}-{suffix}.png")
        tensor_i_pil.save(save_tensor_i_path)
        
def viz_tensors_to_heatmap(tensors, checkpoints_savedir, save_dir_name=None, suffix="", epoch_num=0, step_num=0, split_set="train", val_step=None, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    if save_dir_name is not None:
        save_dir = Path(checkpoints_savedir) / split_set / save_dir_name
    else:
        save_dir = Path(checkpoints_savedir) / split_set
    os.makedirs(save_dir, exist_ok=True)
    for i, tensor_i in enumerate(tensors):
        if tensor_i.ndim == 3 and tensor_i.shape[0] == 1:
            tensor_i = tensor_i.squeeze(0)
        numpy_array = tensor_i.cpu().numpy()
        # The cmap function accepts an array of floating point numbers in the range of 0-1 and 
        # returns a color array in RGBA format (also in the range of 0-1)
        heatmap_rgba = cmap(numpy_array)
        heatmap_rgb_uint8 = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_rgb_uint8)
        if split_set == "train":
            save_tensor_i_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_sample-{i:02d}-{suffix}.png")
        elif split_set == "val":
            save_tensor_i_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_valstep-{val_step}_sample-{i:02d}-{suffix}.png")
        heatmap_pil.save(save_tensor_i_path)

def process_tpl(tpl, mode="RGB"):
    x = tpl.detach().cpu().numpy()
    x_nor = (x - x.min()) / (x.max() - x.min())
    x_pil = Image.fromarray((x * 255).transpose(1, 2, 0).astype(np.uint8).squeeze()).convert(mode)
    x_nor_pil = Image.fromarray((x_nor* 255).transpose(1, 2, 0).astype(np.uint8).squeeze()).convert(mode)
    return x_pil, x_nor_pil

def viz_template(tensor, checkpoints_savedir, save_dir_name=None, suffix="", epoch_num=0, step_num=0, split_set="train", val_step=None, mode="RGB"):
    if save_dir_name is not None:
        save_dir = Path(checkpoints_savedir) / split_set / save_dir_name
    else:
        save_dir = Path(checkpoints_savedir) / split_set
    os.makedirs(save_dir, exist_ok=True)
    
    if tensor.ndim == 3:
        if split_set == "train":
            save_tpl_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}-{suffix}.png")
            save_tpl_nor_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}-{suffix}_nor.png")
        elif split_set == "val":
            save_tpl_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_valstep-{val_step}-{suffix}.png")
            save_tpl_nor_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_valstep-{val_step}-{suffix}_nor.png")
        tpl_pil, tpl_nor_pil = process_tpl(tensor, mode)
        tpl_pil.save(save_tpl_path)
        tpl_nor_pil.save(save_tpl_nor_path)
        
    elif tensor.ndim == 4:
        for i, tensor_i in enumerate(tensor):
            if split_set == "train":
                save_tensor_i_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_sample-{i:02d}-{suffix}.png")
            elif split_set == "val":
                save_tensor_i_path = str(save_dir / f"epochnum-{epoch_num}_trainstep-{step_num}_valstep-{val_step}_sample-{i:02d}-{suffix}.png")
            process_tpl(tensor_i).save(save_tensor_i_path)

    else:
        raise ValueError("tensor.ndim should be 3 or 4")