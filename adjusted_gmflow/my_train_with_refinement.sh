#!/bin/bash

# GMFlow with refinement

# number of gpus for training, please set according to your hardware
# by default use all gpus on a machine
# can be trained on 4x 32G V100 or 4x 40GB A100 or 8x 16G V100 gpus
NUM_GPUS=4

name=gmflow-refine-ad+s-noc
port=9988

# chairs
CHECKPOINT_DIR=checkpoints/${name} && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage augmenteddiml
--batch_size 16 \
--val_dataset sintel kitti \
--lr 4e-4 \
--image_size 384 512 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--with_speed_metric \
--val_freq 10000 \
--save_ckpt_freq 10000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
