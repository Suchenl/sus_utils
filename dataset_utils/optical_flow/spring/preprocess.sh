#!/bin/bash

#SBATCH -N 1                                        # number of nodes
#SBATCH -p gpu4                                     # partition
#SBATCH -c 8                                        # number of cpus
#SBATCH --mem 30G                                    # memory
#SBATCH --gres gpu:1                                # number of gpus of each node
#SBATCH --mail-user=yz.chen@mail.ustc.edu.cn
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --output %j-%x.log                          # %x is job-name, %j is job ID

# #SBATCH --job-name train
# #SBATCH -o logs/%j.sleep                          # standerd output file
# #SBATCH -e logs/%j.sleep                          # error output file

# conda activate SEA-RAFT
cd /public/chenyuzhuo/MODELS/image_watermarking_models/Image_Motion_Pred-dev    # SLURM_SUBMIT_DIR
echo "job begin"

compose_num=50
skip_num=10
h=540
w=960

python -m utils.dataset_utils.optical_flow.spring.compose_and_warp --flow_direction BW --save_exp_name preprocessed_h540w960 --compose_num $compose_num --skip_num $skip_num --img_h $h --img_w $w

echo "job end"