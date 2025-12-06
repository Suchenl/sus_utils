from utils.differentiable.flow_utils.compose_flows import compose_forward_flow_sequence
from utils.io_utils.flow_io import read_flo5, write_flo5
from utils.io_utils.image_io import tensor_to_pil
from utils.viz_utils.flow_viz import viz_flow_with_rgb, viz_flow_with_arrows
import os
import argparse
import torch
import numpy as np
import shutil

def set_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset_dir", type=str, default="/public/chenyuzhuo/DATASETS/Optical_Flow_DATASETS/Spring/spring/preprocessed", help="Directory to read spring dataset")
    parse.add_argument("--set", type=str, default="train", help="train or test")
    parse.add_argument("--flow_dir_name", type=str, default="flow_FW_left")
    parse.add_argument("--frame_dir_name", type=str, default="frame_left")
    parse.add_argument("--flow_direction", type=str, default='FW', help="[FW, BW]")
    parse.add_argument("--img_h", type=int, default=540)
    parse.add_argument("--img_w", type=int, default=960)
    parse.add_argument("--skip_num", type=int, default=5)
    parse.add_argument("--compose_num", type=int, default=30)
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
        frame_save_base_dir = flow_dir.replace(args.flow_dir_name, args.frame_dir_name)
        frame_read_dir = frame_save_base_dir.replace("preprocessed/", "")
        os.makedirs(frame_save_base_dir, exist_ok=True)
        flow_name_list = os.listdir(flow_dir)
        flow_name_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1].split('-')[0]))
        flow_num = len(flow_name_list)
        for i in range(0, flow_num):
            flow_name = flow_name_list[i]
            src_tgt_ordinal = flow_name.split('.')[0].split('_')[-1]
            frame_src_name = args.frame_dir_name + src_tgt_ordinal.split('-')[0] + '.png'
            frame_tgt_name = args.frame_dir_name + src_tgt_ordinal.split('-')[1] + '.png'

            frame_warped_name = args.frame_dir_name + src_tgt_ordinal + '.png'
            save_dir = frame_save_base_dir.replace(args.frame_dir_name, src_tgt_ordinal)
            
            frame_src_read_path = os.path.join(frame_read_dir, frame_src_name)
            frame_tgt_read_path = os.path.join(frame_read_dir, frame_tgt_name)
            frame_src_save_path = os.path.join(frame_save_dir, frame_src_name)
            frame_tgt_save_path = os.path.join(frame_save_dir, frame_tgt_name)
            shutil.copyfile(frame_src_read_path, frame_src_save_path)
            frame_warped_save_path = os.path.join(frame_save_dir, frame_warped_name)
            

            
            # if the file exists, skip it
            base_save_path = os.path.join(save_dir, f"{args.flow_dir_name}_{flow_start_ordinal:04d}-{flow_start_ordinal + args.compose_num:04d}")
            save_flow_path = base_save_path + "_flow.flo5"
            if os.path.exists(save_flow_path):
                print(f"The file exists: {save_flow_path}, so skip it")
                continue
            else:
                flow_list = []
                valid_list = []
                for j in range(i, i + args.compose_num):
                    flow_name = flow_name_list[j]
                    flow_path = os.path.join(flow_dir, flow_name)
                    flow, valid = read_flo5(flow_path, calc_valid=True)
                    # print(f"Max of flow {j}: {flow.max()}, min of flow {j}: {flow.min()}")
                    flow_list.append(flow.unsqueeze(0).detach().float())
                    valid_list.append(valid.unsqueeze(0).detach())
                flow_total, valid_total = compose_forward_flow_sequence(flows=flow_list, valids=valid_list)
                flow_total, valid_total = flow_total.squeeze(0), valid_total.squeeze(0)
                write_flo5(flow_total.detach().numpy().astype(np.float16), save_flow_path) # save the flow (format: '.flo5')
                                
                # visualize the flow and valid and save them
                viz_flow_with_rgb(flow_total.float()).save(base_save_path + '_flow_rgb.png')
                viz_flow_with_arrows(flow_total.float()).save(base_save_path + '_flow_arrows.png')
                tensor_to_pil(valid_total * 255, mode='L').save(base_save_path + '_valid.png')
                print(f"Save the flow and valid to {save_flow_path}")

if __name__ == "__main__":
    args = set_args()
    main(args)
            