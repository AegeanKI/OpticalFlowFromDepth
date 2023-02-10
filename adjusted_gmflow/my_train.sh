#!/bin/bash

NUM_GPUS=4

# checkpoint_dir=checkpoints/gmflow-augmenteddiml-512-1382-384-1152-1e5-1 && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9987 main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --stage augmenteddiml \
# --batch_size 8 \
# --val_dataset kitti sintel \
# --lr 4e-4 \
# --image_size 384 1152 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 100000 \
# --early_stop 30000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1672944585.6048822 \
# --classifier_checkpoint_train_acc 0.675 \
# --classifier_checkpoint_test_acc 0.804 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00001 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# checkpoint_dir=checkpoints/gmflow-augmenteddiml-512-1382-384-1152-no-c && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9988 main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --stage augmenteddiml \
# --batch_size 8 \
# --val_dataset kitti sintel \
# --lr 4e-4 \
# --image_size 384 1152 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 100000 \
# --early_stop 30000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# ==== 
# name=gmflow-ar+s-c-2e5-1
# port=9989
# checkpoint_dir=checkpoints/$name && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --stage augmentedredweb \
# --batch_size 16 \
# --val_dataset kitti sintel \
# --lr 4e-4 \
# --image_size 368 496 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 100000 \
# --early_stop 16000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1672944585.6048822 \
# --classifier_checkpoint_train_acc 0.675 \
# --classifier_checkpoint_test_acc 0.804 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log


# CHECKPOINT_DIR=checkpoints/$name-t && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --resume checkpoints/$name/step_016000.pth \
# --stage things \
# --batch_size 8 \
# --val_dataset kitti sintel \
# --lr 2e-4 \
# --image_size 384 768 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 800000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# Feb 1, 6
# gmflow + ad_s

# name=gmflow-fc-ad-s-c-1e5-1
# port=9989
# checkpoint_dir=checkpoints/$name && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --resume checkpoints/pretrained/gmflow_chairs-1d776046.pth \
# --stage augmenteddiml \
# --batch_size 16 \
# --val_dataset kitti sintel \
# --lr 4e-4 \
# --image_size 368 496 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1672944585.6048822 \
# --classifier_checkpoint_train_acc 0.675 \
# --classifier_checkpoint_test_acc 0.804 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00001 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Feb 4
# gmflow + ad_s + FC

# name=gmflow-ad-s-c-1e5-1-fc
# port=9989
# checkpoint_dir=checkpoints/$name && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --resume checkpoints/gmflow-ad-s-c-1e5-1/step_045000.pth \
# --stage chairs \
# --batch_size 16 \
# --val_dataset kitti sintel \
# --lr 2e-4 \
# --image_size 384 512 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log
# --add_classifier \
# --classifier_checkpoint_timestamp 1672944585.6048822 \
# --classifier_checkpoint_train_acc 0.675 \
# --classifier_checkpoint_test_acc 0.804 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00001 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \

# Feb 10
# gmflow AD+s
# AD+s vs FD+s

# name=gmflow-ad+s-noc
# port=9988
# checkpoint_dir=checkpoints/$name && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --stage augmenteddiml \
# --batch_size 16 \
# --val_dataset kitti sintel \
# --lr 4e-4 \
# --image_size 384 512 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Feb 10
# gmflow + FC + ad+s
# batch size 8, lr 2e-4, test stage things, kitti different

name=gmflow-fc-ad+sk-noc
port=9989
checkpoint_dir=checkpoints/$name && \
mkdir -p ${checkpoint_dir} && \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
--launcher pytorch \
--checkpoint_dir ${checkpoint_dir} \
--resume checkpoints/pretrained/gmflow_chairs-1d776046.pth \
--stage augmenteddiml \
--batch_size 8 \
--val_dataset kitti sintel \
--lr 2e-4 \
--image_size 384 512 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 1000 \
--save_ckpt_freq 1000 \
--num_steps 200000 \
2>&1 | tee -a ${checkpoint_dir}/train.log
