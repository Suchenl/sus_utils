from utils.differentiable.flow_utils import (
    compose_forward_flow_sequence, 
    group_compose_forward_flow_sequence,
    forward_warp, 
    backward_warp,
    check_flow_consistency
    )
from utils.io_utils.flow_io import read_flo5, write_flo5
from utils.io_utils.image_io import tensor_to_pil
from utils.io_utils import parse_args
from utils.viz_utils import viz_flow_with_rgb, viz_flow_with_arrows, viz_hist, plot_heatmap
from utils.differentiable.flow_utils import resize_flow
from utils.cluster_utils import kmeans_2d_cluster, superpixels_2d_cluster
from utils.numerical_utils import kmeans_threshold, otsu_threshold
from utils.morphology_utils import morphological_opening, morphological_closing

import os
import argparse
import torch
import numpy as np
import shutil
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from core.sea_raft import load_ckpt
from core.sea_raft import RAFT
from typing import *

def init_SEA_RAFT(args):
    model = RAFT(args.of_model_config)
    print("Loading pre-trained (optical flow estimation) model from {}".format(args.of_model_weight))
    load_ckpt(model, args.of_model_weight)
    return model
    
def count_iters(frame_num, compose_num, skip_num):
    """
    Calculate the number of iterations for:
        range(0, frame_num - compose_num + 1, skip_num)
    Args:
        frame_num (int): total number of frames
        compose_num (int): number of flows to compose
        skip_num (int): step size
    Returns:
        int: number of iterations
    """
    return (frame_num - compose_num) // skip_num + 1

def set_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parse.add_argument("--dataset_dir", type=str, default="/public/chenyuzhuo/DATASETS/Optical_Flow_DATASETS/Spring/spring", help="Directory to read spring dataset")
    parse.add_argument("--set", type=str, default="train", help="train or test")
    parse.add_argument("--eye_pos", type=str, default="left", help="[left, right]")
    parse.add_argument("--img_h", type=int, default=540) 
    parse.add_argument("--img_w", type=int, default=960)
    parse.add_argument("--skip_num", type=int, default=10)
    parse.add_argument("--compose_num", type=int, default=5)
    parse.add_argument("--save_warped", type=bool, default=True)
    parse.add_argument("--save_exp_name", type=str, default="preprocessed_h540w960_test")
    parse.add_argument("--chosen_scenes", type=int, nargs="+", default=None)
    parse.add_argument("--content_threshold", type=float, default=0.25)
    # Settings for Flow Consistency Check
    parse.add_argument("--consistency_threshold", type=float, default=2.0, help="consistency threshold for backward warping, unit is pixel")
    parse.add_argument("--nbins", type=int, default=256)
    parse.add_argument("--save_hist", type=bool, default=True)
    # Settings for Compose Mode
    parse.add_argument("--compose_mode", type=str, default='sample', help="[sample, scatter]")
    # Settings for Sample Mode
    parse.add_argument("--align_corners", type=bool, default=False)
    # Settings for Oprical Flow Estimation Model
    parse.add_argument("--of_model_config", type=str, default="/public/chenyuzhuo/MODELS/image_watermarking_models/Image_Motion_Pred-dev/core/sea_raft/configs/eval/spring-L.json")
    parse.add_argument("--of_model_weight", type=str, default="/public/chenyuzhuo/MODELS/Optical_Flow_Estimation/checkpoints/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M.pth")
    args = parse_args(parse)
    args.frame_dir_name = f"frame_{args.eye_pos}"
    return args

def main(args):
    # set device
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # initialize the SEA-RAFT model
    of_model = init_SEA_RAFT(args).to(device)

    # set image transform
    img_transform = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    # walk through the dataset
    set_dir = os.path.join(args.dataset_dir, args.set)
    scene_list = os.listdir(set_dir)
    scene_list.sort(key=lambda x: int(x))
    for scene in scene_list:
        scene_dir = os.path.join(set_dir, scene)
        if args.chosen_scenes is not None:
            if int(scene) not in args.chosen_scenes:
                continue
        frame_dir = os.path.join(scene_dir, args.frame_dir_name)
        frame_list = os.listdir(frame_dir)
        frame_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        
        frame_num = len(frame_list)
        iters = count_iters(frame_num, args.compose_num, args.skip_num)
        bar = tqdm(range(iters), desc=f"Processing")
        for i in bar:
            flow_start_ordinal = i * args.skip_num + 1
            flow_end_ordinal = flow_start_ordinal + args.compose_num
            if flow_end_ordinal > frame_num:
                break
            bar.set_description(f"Processing scene {scene}, frame {flow_start_ordinal:04d} to {flow_end_ordinal:04d}")
            # Set save dirs
            save_dir = os.path.join(args.dataset_dir, args.save_exp_name, args.set, f"scene_{int(scene):09d}", 
                                    f"frame_{flow_start_ordinal:04d}-{flow_end_ordinal:04d}")
            os.makedirs(save_dir, exist_ok=True)

            # Read the frames and save them to save dir
            frame_start_read_path = os.path.join(frame_dir, args.frame_dir_name + f"_{flow_start_ordinal:04d}.png")
            frame_end_read_path = os.path.join(frame_dir, args.frame_dir_name + f"_{flow_end_ordinal:04d}.png")
            frame_start = Image.open(frame_start_read_path).resize((args.img_w, args.img_h))
            frame_end = Image.open(frame_end_read_path).resize((args.img_w, args.img_h))
            frame_start_save_path = os.path.join(save_dir, f"{args.eye_pos}_frame_start.png")
            frame_end_save_path = os.path.join(save_dir, f"{args.eye_pos}_frame_end.png")
            frame_start.save(frame_start_save_path)
            frame_end.save(frame_end_save_path)
            
            # Transform the frames
            frame_start_tensor = img_transform(frame_start).unsqueeze(0).to(device)
            frame_end_tensor = img_transform(frame_end).unsqueeze(0).to(device)

            ###---------------------------continue write
            # Predict the flow
            with torch.no_grad():
                if args.compose_num != 1:
                    forward_out = of_model(frame_start_tensor, frame_end_tensor, iters=args.of_model_config.iters, test_mode=False)
                    forward_flow = forward_out['final']
                    forward_conf = of_model.calc_confidence(forward_out['info'][-1])
                    plot_heatmap(forward_conf, save_path=os.path.join(save_dir, f"{args.eye_pos}_FW_flow_conf.png"))
                    # check
                    forward_flows = forward_out['flow']
                    for i, flow in enumerate(forward_flows):
                        print(f"forward flow {i} shape: {flow.shape}")
                        # visualize the flow and valid and save them
                        save_flow_viz_rgb_path = os.path.join(save_dir, f"{args.eye_pos}_FW_flow_{i}_rgb.png")
                        viz_flow_with_rgb(flow.float().squeeze(0)).save(save_flow_viz_rgb_path)
                        
                        save_flow_viz_arrow_path = os.path.join(save_dir, f"{args.eye_pos}_FW_flow_{i}_arrow.png")
                        viz_flow_with_arrows(flow.float().squeeze(0)).save(save_flow_viz_arrow_path)
                    
                    backward_out = of_model(frame_end_tensor, frame_start_tensor, iters=args.of_model_config.iters, test_mode=False)
                    backward_flow = backward_out['final']
                    backward_conf = of_model.calc_confidence(backward_out['info'][-1])
                    plot_heatmap(backward_conf, save_path=os.path.join(save_dir, f"{args.eye_pos}_BW_flow_conf.png"))

                else:
                    forward_flow_read_path = frame_start_read_path.replace("frame", "flow_FW").replace(".png", ".flo5")
                    forward_flow = read_flo5(forward_flow_read_path, calc_valid=False).unsqueeze(0).to(device)
                    forward_flow = resize_flow(forward_flow, args.img_h, args.img_w, valid_mask=None)
                    # if os.path.exists(forward_flow_read_path):
                    #     forward_flow = read_flo5(forward_flow_read_path, calc_valid=False)
                    #     forward_flow = resize_flow(forward_flow, args.img_h, args.img_w, valid_mask=None)
                    # else:
                    #     forward_flow = of_model(frame_start_tensor, frame_end_tensor, iters=args.of_model_config.iters, test_mode=False)['final']

                    backward_flow_read_path = frame_end_read_path.replace("frame", "flow_BW").replace(".png", ".flo5")
                    backward_flow = read_flo5(backward_flow_read_path, calc_valid=False).unsqueeze(0).to(device)
                    backward_flow = resize_flow(backward_flow, args.img_h, args.img_w, valid_mask=None)
                    # if os.path.exists(backward_flow_read_path):
                    #     backward_flow = read_flo5(backward_flow_read_path, calc_valid=False)
                    #     backward_flow = resize_flow(backward_flow, args.img_h, args.img_w, valid_mask=None)
                    # else:
                    #     backward_flow = of_model(frame_end_tensor, frame_start_tensor, iters=args.of_model_config.iters, test_mode=False)['final']
                        
                # check consistency
                def check_consistency(flow_fw, flow_bw, check_direction: Literal['forward', 'backward']):
                    # check
                    save_hist_path = os.path.join(save_dir, f"{args.eye_pos}_check_error_{check_direction}_hist.png")
                    check_error = check_flow_consistency(flow_fw, flow_bw, check_direction)
                    # get check mask
                    ## 1. kmeans 2d cluster
                    # check_mask_np = kmeans_2d_cluster(data=check_error.squeeze(0, 1), n_clusters=2, data_norm='minmax',
                    #                                   viz=True, viz_path=save_dir + f"/{args.eye_pos}_check_mask_{check_direction}.png")
                    ## 2. superpixels 2d cluster
                    # check_mask_np = superpixels_2d_cluster(data=check_error.squeeze(0, 1))

                    ## 3. kmeans 1d cluster
                    # check_error = (check_error - check_error.min()) / (check_error.max() - check_error.min())
                    # check_error = check_error ** 0.7
                    # thre = kmeans_threshold(check_error)
                    # check_mask = (check_error >= thre).float()
                    # check_mask_np = check_mask.squeeze(0, 1).detach().cpu().numpy()

                    ## 4. fixed threshold
                    # check_error = (check_error - check_error.min()) / (check_error.max() - check_error.min())
                    # check_error = check_error ** 0.7
                    thre = args.consistency_threshold
                    check_mask = (check_error >= thre).float()
                    check_mask_np = check_mask.squeeze(0, 1).detach().cpu().numpy()
                    
                    # morphological opening and closing
                    check_mask_np = morphological_opening(check_mask_np, kernel_size=5)
                    check_mask_np = morphological_closing(check_mask_np, kernel_size=5)
                    
                    # save
                    Image.fromarray((check_mask_np * 255).astype(np.uint8)).convert("L").save(save_dir + f"/{args.eye_pos}_check_mask_{check_direction}.png")
                    check_error_np = check_error.squeeze(0, 1).detach().cpu().numpy()
                    check_error_np_norm = (check_error_np - check_error_np.min()) / (check_error_np.max() - check_error_np.min()) 
                    # check_error_np = np.power(check_error_np, 0.5)
                    Image.fromarray((check_error_np_norm * 255)).convert("L").save(save_dir + f"/{args.eye_pos}_check_error_{check_direction}.png")
                    
                check_consistency(flow_fw=forward_flow, flow_bw=backward_flow, check_direction='forward')
                check_consistency(flow_fw=forward_flow, flow_bw=backward_flow, check_direction='backward')
                    
                def post_process(flow_total, flow_direction, forward_frame_tensor, forward_frame_name, backward_frame_tensor, backward_frame_name):
                    save_flow_path = os.path.join(save_dir, f"{args.eye_pos}_{flow_direction}_flow.flo5")
                    # write_flo5(flow_total.detach().numpy().astype(np.float16), save_flow_path) # save the flow (format: '.flo5')
                    write_flo5(flow_total.detach().cpu().numpy(), save_flow_path) # save the flow (format: '.flo5')
                                    
                    # visualize the flow and valid and save them
                    save_flow_viz_rgb_path = os.path.join(save_dir, f"{args.eye_pos}_{flow_direction}_flow_rgb.png")
                    viz_flow_with_rgb(flow_total.float()).save(save_flow_viz_rgb_path)
                    
                    save_flow_viz_arrow_path = os.path.join(save_dir, f"{args.eye_pos}_{flow_direction}_flow_arrow.png")
                    viz_flow_with_arrows(flow_total.float()).save(save_flow_viz_arrow_path)

                    # forward warp
                    fwed_image_tensor, valid_mask, weight_sum, content_mask = forward_warp(forward_frame_tensor, flow_total.unsqueeze(0), threshold=args.content_threshold)
                    ##  save warped image and valid mask
                    fwed_image_np = ((fwed_image_tensor.detach() + 1) / 2 * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    Image.fromarray(fwed_image_np.astype(np.uint8)).save(save_dir + f"/{args.eye_pos}_{flow_direction}_fwed_{forward_frame_name}.png")
                    valid_mask = valid_mask.squeeze(0, 1).detach().cpu().numpy()
                    Image.fromarray((valid_mask * 255).astype(np.uint8)).convert("L").save(save_dir + f"/{args.eye_pos}_{flow_direction}_fwed_valid_mask.png")
                    ## viz weight sum
                    weight_sum = weight_sum.squeeze(0, 1).detach().cpu().numpy()
                    weight_sum = (weight_sum - weight_sum.min()) / (weight_sum.max() - weight_sum.min()) 
                    weight_sum = np.power(weight_sum, 0.5)
                    weight_sum_img = Image.fromarray((weight_sum * 255)).convert("L")
                    weight_sum_img.save(save_dir + f"/{args.eye_pos}_{flow_direction}_fwed_weight_sum.png")
                    ## viz weight sum with arrows
                    viz_flow_with_arrows(flow_total.float(), weight_sum_img).save(save_dir + f"/{args.eye_pos}_{flow_direction}_fwed_weight_sum_and_arrow.png")
                    
                    content_mask_np = content_mask.squeeze(0, 1).detach().cpu().numpy()
                    # morphological opening and closing
                    content_mask_np = morphological_opening(content_mask_np, kernel_size=5)
                    content_mask_np = morphological_closing(content_mask_np, kernel_size=5)
                    Image.fromarray((content_mask_np * 255).astype(np.uint8)).convert("L").save(save_dir + f"/{args.eye_pos}_{flow_direction}_fwed_content_mask.png")

                    # backward warp
                    bwed_image_tensor, valid_mask = backward_warp(backward_frame_tensor, flow_total.unsqueeze(0))
                    ##  save warped image and valid mask
                    bwed_image_np = ((bwed_image_tensor.detach() + 1) / 2 * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    Image.fromarray(bwed_image_np.astype(np.uint8)).save(save_dir + f"/{args.eye_pos}_{flow_direction}_bwed_{backward_frame_name}.png")
                    valid_mask = valid_mask.squeeze(0, 1).detach().cpu().numpy()
                    Image.fromarray((valid_mask * 255).astype(np.uint8)).convert("L").save(save_dir + f"/{args.eye_pos}_{flow_direction}_bwed_valid_mask.png")

                post_process(forward_flow.squeeze(0), flow_direction='FW', 
                            forward_frame_tensor=frame_start_tensor, forward_frame_name='frame_start',
                            backward_frame_tensor=frame_end_tensor, backward_frame_name='frame_end')
                post_process(backward_flow.squeeze(0), flow_direction='BW', 
                            forward_frame_tensor=frame_end_tensor, forward_frame_name='frame_end',
                            backward_frame_tensor=frame_start_tensor, backward_frame_name='frame_start')
                
                print(f"Save the outputs to {save_dir}")

                
                # get composed warped image
                frame_start_np = np.array(Image.open(save_dir + f"/{args.eye_pos}_frame_start.png"))
                frame_end_np = np.array(Image.open(save_dir + f"/{args.eye_pos}_frame_end.png"))
                
                ## 1. composed warped frame_start
                warped_img = np.array(Image.open(save_dir + f"/{args.eye_pos}_BW_bwed_frame_start.png"))
                non_content_mask = (np.array(Image.open(save_dir + f"/{args.eye_pos}_FW_fwed_content_mask.png")) / 255) < 0.5
                non_val_mask = (np.array(Image.open(save_dir + f"/{args.eye_pos}_BW_bwed_valid_mask.png")) / 255) < 0.5
                check_mask = (np.array(Image.open(save_dir + f"/{args.eye_pos}_check_mask_backward.png")) / 255) >= 0.5
                add_content_mask = non_content_mask | non_val_mask | check_mask # calculate the add_content_mask by logical or

                recovered_frame_end = warped_img * (1 - add_content_mask[..., None])
                composed_warped_img = warped_img * (1 - add_content_mask[..., None]) + frame_end_np * add_content_mask[..., None]
                Image.fromarray(recovered_frame_end.astype(np.uint8)).save(save_dir + f"/{args.eye_pos}_recovered_frame_end.png")
                Image.fromarray(composed_warped_img.astype(np.uint8)).save(save_dir + f"/{args.eye_pos}_BW_bwed_frame_start_composed.png")
                Image.fromarray(add_content_mask).save(save_dir + f"/{args.eye_pos}_BW_bwed_frame_start_composed_mask-ADD.png")
                
                ## 2. composed backward frame_end
                warped_img = np.array(Image.open(save_dir + f"/{args.eye_pos}_FW_bwed_frame_end.png"))
                non_content_mask = (np.array(Image.open(save_dir + f"/{args.eye_pos}_BW_fwed_content_mask.png")) / 255) < 0.5
                non_val_mask = (np.array(Image.open(save_dir + f"/{args.eye_pos}_FW_fwed_valid_mask.png")) / 255) < 0.5
                check_mask = (np.array(Image.open(save_dir + f"/{args.eye_pos}_check_mask_forward.png")) / 255) >= 0.5
                del_content_mask = non_content_mask | non_val_mask | check_mask  # calculate the del_content_mask by logical or

                recovered_frame_start = warped_img * (1 - del_content_mask[..., None])
                composed_warped_img = warped_img * (1 - del_content_mask[..., None]) + frame_start_np * del_content_mask[..., None]
                Image.fromarray(recovered_frame_start.astype(np.uint8)).save(save_dir + f"/{args.eye_pos}_recovered_frame_start.png") 
                Image.fromarray(composed_warped_img.astype(np.uint8)).save(save_dir + f"/{args.eye_pos}_FW_bwed_frame_end_composed.png")
                Image.fromarray(del_content_mask).save(save_dir + f"/{args.eye_pos}_FW_bwed_frame_end_composed_mask-DEL.png")
                
if __name__ == "__main__":
    args = set_args()
    main(args)