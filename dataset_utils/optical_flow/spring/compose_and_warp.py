from utils.differentiable.flow_utils import (
    compose_forward_flow_sequence, 
    group_compose_forward_flow_sequence,
    forward_warp, 
    backward_warp 
    )
from utils.io_utils.flow_io import read_flo5, write_flo5
from utils.io_utils.image_io import tensor_to_pil
from utils.viz_utils.flow_viz import viz_flow_with_rgb, viz_flow_with_arrows
from utils.differentiable.flow_utils import resize_flow
import os
import argparse
import torch
import numpy as np
import shutil
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def count_iters(flow_num, compose_num, skip_num):
    """
    Calculate the number of iterations for:
        range(0, flow_num - compose_num + 1, skip_num)
    Args:
        flow_num (int): total number of flows
        compose_num (int): number of flows to compose
        skip_num (int): step size
    Returns:
        int: number of iterations
    """
    return (flow_num - compose_num) // skip_num + 1

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
    parse.add_argument("--chosen_scenes", type=int, nargs="+", default=None)
    parse.add_argument("--content_threshold", type=float, default=0.1)
    parse.add_argument("--compose_mode", type=str, default='sample', help="[sample, scatter]")
    parse.add_argument("--align_corners", type=bool, default=False)
    args = parse.parse_args()
    args.flow_dir_name = f"flow_{args.flow_direction}_{args.eye_pos}"
    args.frame_dir_name = f"frame_{args.eye_pos}"
    return args

def main(args):
    img_transform = transforms.Compose([transforms.Resize((args.img_h, args.img_w)),
                                        transforms.ToTensor()])
    set_dir = os.path.join(args.dataset_dir, args.set)
    scene_list = os.listdir(set_dir)
    scene_list.sort(key=lambda x: int(x))
    for scene in scene_list:
        scene_dir = os.path.join(set_dir, scene)
        # preprocess flows
        flow_dir = os.path.join(scene_dir, args.flow_dir_name)
        flow_name_list = os.listdir(flow_dir)
        flow_name_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        flow_num = len(flow_name_list)

        if args.chosen_scenes is not None:
            if int(scene) not in args.chosen_scenes:
                continue
            
        iters = count_iters(flow_num, args.compose_num, args.skip_num)
        bar = tqdm(range(iters), desc=f"Processing")
        for i in bar:
            flow_start_ordinal = i * args.skip_num + 1
            # flow_start_ordinal = int(flow_name_list[i].split('.')[0].split('_')[-1])
            flow_end_ordinal = flow_start_ordinal + args.compose_num
            if args.flow_direction == 'FW':
                bar.set_description(f"Processing scene {scene}, frame {flow_start_ordinal:04d} to {flow_end_ordinal:04d}")
            elif args.flow_direction == 'BW':
                bar.set_description(f"Processing scene {scene}, frame {flow_end_ordinal:04d} to {flow_start_ordinal:04d}")
            else:
                raise ValueError("Invalid flow direction")
            # if the file exists, skip it
            save_dir = os.path.join(args.dataset_dir, args.save_exp_name, args.set, f"scene_{int(scene):09d}", 
                                    f"frame_{flow_start_ordinal:04d}-{flow_end_ordinal:04d}")
            save_flow_path = os.path.join(save_dir, f"{args.eye_pos}_{args.flow_direction}_flow.flo5")
            if os.path.exists(save_flow_path):
                print(f"The file exists: {save_flow_path}, so skip {save_flow_path}")
                # continue
                pass
            
            # compose flows
            os.makedirs(save_dir, exist_ok=True)
            flow_list = []
            valid_list = []
            for j in range(flow_start_ordinal, flow_end_ordinal):
                flow_name = flow_name_list[j - 1]
                flow_path = os.path.join(flow_dir, flow_name)
                flow, valid = read_flo5(flow_path, calc_valid=True)
                flow, valid = resize_flow(flow, args.img_h, args.img_w, valid_mask=valid)
                # print(f"Max of flow {j}: {flow.max()}, min of flow {j}: {flow.min()}")
                flow_list.append(flow.unsqueeze(0).detach().float())
                valid_list.append(valid.unsqueeze(0).detach())
            if args.flow_direction == 'BW':
                flow_list.reverse()
                valid_list.reverse()
            flow_total, valid_total = compose_forward_flow_sequence(flows=flow_list, 
                                                                    valids=valid_list, 
                                                                    mode=args.compose_mode, 
                                                                    align_corners=args.align_corners)
            flow_total, valid_total = flow_total.squeeze(0), valid_total.squeeze(0)
            write_flo5(flow_total.detach().numpy().astype(np.float16), save_flow_path) # save the flow (format: '.flo5')
                            
            # visualize the flow and valid and save them
            save_flow_viz_rgb_path = os.path.join(save_dir, f"{args.eye_pos}_{args.flow_direction}_flow_rgb.png")
            viz_flow_with_rgb(flow_total.float()).save(save_flow_viz_rgb_path)
            
            save_flow_viz_arrow_path = os.path.join(save_dir, f"{args.eye_pos}_{args.flow_direction}_flow_arrow.png")
            viz_flow_with_arrows(flow_total.float()).save(save_flow_viz_arrow_path)

            save_valid_path = os.path.join(save_dir, f"{args.eye_pos}_{args.flow_direction}_compose_valid_mask.png")
            tensor_to_pil(valid_total * 255, mode='L').save(save_valid_path)

            frame_read_dir = flow_dir.replace(args.flow_dir_name, args.frame_dir_name)
            frame_start_read_path = os.path.join(frame_read_dir, args.frame_dir_name + f"_{flow_start_ordinal:04d}.png")
            frame_end_read_path = os.path.join(frame_read_dir, args.frame_dir_name + f"_{flow_end_ordinal:04d}.png")
            frame_start = Image.open(frame_start_read_path).resize((args.img_w, args.img_h))
            frame_end = Image.open(frame_end_read_path).resize((args.img_w, args.img_h))
        
            frame_start_save_path = os.path.join(save_dir, f"{args.eye_pos}_frame_start.png")
            frame_end_save_path = os.path.join(save_dir, f"{args.eye_pos}_frame_end.png")
            frame_start.save(frame_start_save_path)
            frame_end.save(frame_end_save_path)
            
            if args.save_warped:
                if args.flow_direction == 'FW':
                    forward_frame_tensor = img_transform(frame_start).unsqueeze(0)
                    backward_frame_tensor = img_transform(frame_end).unsqueeze(0)
                    forward_warped_frame = "frame_start"
                    backward_warped_frame = "frame_end"
                elif args.flow_direction == 'BW':
                    forward_frame_tensor = img_transform(frame_end).unsqueeze(0)
                    backward_frame_tensor = img_transform(frame_start).unsqueeze(0)
                    forward_warped_frame = "frame_end"
                    backward_warped_frame = "frame_start"
                else:
                    raise ValueError("Invalid flow direction")
                # forward warp
                warped_image, valid_mask, weight_sum, content_mask = forward_warp(forward_frame_tensor, flow_total.unsqueeze(0), threshold=args.content_threshold)
                # save warped image and valid mask
                warped_image_np = warped_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                Image.fromarray((warped_image_np * 255).astype(np.uint8)).save(save_dir + f"/{args.eye_pos}_{args.flow_direction}_fwed_{forward_warped_frame}.png")
                valid_mask = valid_mask.squeeze(0, 1).detach().cpu().numpy()
                Image.fromarray((valid_mask * 255).astype(np.uint8)).convert("L").save(save_dir + f"/{args.eye_pos}_{args.flow_direction}_fwed_valid_mask.png")
                weight_sum = weight_sum.squeeze(0, 1).detach().cpu().numpy()
                weight_sum = (weight_sum - weight_sum.min()) / (weight_sum.max() - weight_sum.min()) 
                weight_sum = np.power(weight_sum, 0.5)
                Image.fromarray((weight_sum * 255)).convert("L").save(save_dir + f"/{args.eye_pos}_{args.flow_direction}_fwed_weight_sum.png")
                content_mask_np = content_mask.squeeze(0, 1).detach().cpu().numpy()
                Image.fromarray((content_mask_np * 255).astype(np.uint8)).convert("L").save(save_dir + f"/{args.eye_pos}_{args.flow_direction}_fwed_content_mask.png")

                # backward warp
                warped_image, valid_mask = backward_warp(backward_frame_tensor, flow_total.unsqueeze(0))
                # save warped image and valid mask
                warped_image_np = warped_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                Image.fromarray((warped_image_np * 255).astype(np.uint8)).save(save_dir + f"/{args.eye_pos}_{args.flow_direction}_bwed_{backward_warped_frame}.png")
                valid_mask = valid_mask.squeeze(0, 1).detach().cpu().numpy()
                Image.fromarray((valid_mask * 255).astype(np.uint8)).convert("L").save(save_dir + f"/{args.eye_pos}_{args.flow_direction}_bwed_valid_mask.png")

                frame_end_tensor = img_transform(frame_end)
                mse = ((frame_end_tensor - warped_image.squeeze(0)) * content_mask).pow(2).mean()
                bar.set_postfix(mse=f"{mse:.3f}")
            print(f"Save the outputs to {save_dir}")

if __name__ == "__main__":
    args = set_args()
    main(args)