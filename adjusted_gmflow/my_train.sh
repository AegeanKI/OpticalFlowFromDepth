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

# name=gmflow-fc-ad+sk-noc
# port=9989
# checkpoint_dir=checkpoints/$name && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --resume checkpoints/pretrained/gmflow_chairs-1d776046.pth \
# --stage augmenteddiml \
# --batch_size 8 \
# --val_dataset kitti sintel \
# --lr 2e-4 \
# --image_size 384 512 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 200000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Feb 13
# gmflow + ad+s + FC
# batch size 8, lr 2e-4, test stage things, kitti different

# name=gmflow-ad+s-fck-noc
# port=9989
# checkpoint_dir=checkpoints/$name && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --resume checkpoints/backup-gmflow-ad+s-noc/step_084000.pth \
# --stage chairs \
# --batch_size 8 \
# --val_dataset kitti sintel \
# --lr 2e-4 \
# --image_size 384 512 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 200000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Feb 20 eva 0,1,2,3
# gmflow-test-ar+s-newc-2e5-1

# name=gmflow-test-ar+s-newc-2e5-1
# port=9988
# checkpoint_dir=checkpoints/$name && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --stage test-augmentedredweb \
# --batch_size 16 \
# --val_dataset kitti sintel \
# --lr 4e-4 \
# --image_size 384 512 \
# --original_image_size 480 640 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1676428131.235737 \
# --classifier_checkpoint_train_acc 0.939 \
# --classifier_checkpoint_test_acc 0.913 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Feb 20 eva 4,5,6,7
# gmflow-ad+s-newc-2e5-1

# name=gmflow-ad+s-newc-2e5-1
# port=9989
# checkpoint_dir=checkpoints/$name && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --stage augmenteddiml \
# --batch_size 16 \
# --val_dataset kitti sintel \
# --lr 4e-4 \
# --image_size 384 512 \
# --original_image_size 384 640 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1676428131.235737 \
# --classifier_checkpoint_train_acc 0.939 \
# --classifier_checkpoint_test_acc 0.913 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Feb 20 eva4 0,1,2,3
# gmflow-mixed-newc-2e5-1
# ad ar original image size not same, test (480, 640)
# num steps 200000

# name=gmflow-mixed-newc-2e5-1
# port=9988
# checkpoint_dir=checkpoints/$name && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --stage mixed \
# --batch_size 16 \
# --val_dataset kitti sintel \
# --lr 4e-4 \
# --image_size 384 512 \
# --original_image_size 480 640 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 200000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1676428131.235737 \
# --classifier_checkpoint_train_acc 0.939 \
# --classifier_checkpoint_test_acc 0.913 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Feb 20 eva4 4,5,6,7
# gmflow-c-t-mixed-newc-2e5-1
# ad ar original image size not same, test (480, 640)
# batch size 8, lr 2e-4, kitti scale
# num steps 200000

# name=gmflow-c-t-mixed-newc-2e5-1
# port=9989
# checkpoint_dir=checkpoints/$name && \
# mkdir -p ${checkpoint_dir} && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${checkpoint_dir} \
# --resume checkpoints/pretrained/gmflow_things-e9887eda.pth \
# --stage mixed \
# --batch_size 8 \
# --val_dataset kitti sintel \
# --lr 2e-4 \
# --image_size 384 512 \
# --original_image_size 480 640 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 200000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1676428131.235737 \
# --classifier_checkpoint_train_acc 0.939 \
# --classifier_checkpoint_test_acc 0.913 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# Feb 28 twcc 4,5
# gmflow ad+s
# name=gmflow-ad+s-noc
# port=9989
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=4,5 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage augmenteddiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 384 512 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Feb 28 twcc 6,7
# gmflow fd+s
# name=gmflow-fd+s-noc
# port=9990
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=6,7 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage flowdiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 384 512 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Feb 28 twcc 6,7
# gmflow vd+s
# name=gmflow-vd+s-noc
# port=9990
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=6,7 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage vemdiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 384 512 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 1 twcc 0,1
# gmflow ar+s
# name=gmflow-ar+s-noc
# port=9987
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage augmentedfiltedredweb \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 432 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 1 twcc 2,3
# gmflow fr+s
# name=gmflow-fr+s-noc
# port=9988
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=2,3 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage flowfiltedredweb \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 432 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 1 twcc 4,5
# gmflow vr+s
# name=gmflow-vr+s-noc
# port=9989
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=4,5 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage vemfiltedredweb \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 432 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 3 twcc 6,7
# gmflow mixed+s-c
# name=gmflow-mixed+s-c-2e5-1
# port=9990
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=6,7 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage mixed \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1677566045.275271 \
# --classifier_checkpoint_train_acc 0.805 \
# --classifier_checkpoint_test_acc 0.802 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 3 twcc 4,5
# gmflow mixed+s-c-2nd
# name=gmflow-mixed+s-c-2e5-1-2nd
# port=9989
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=4,5 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage mixed \
# --batch_size 8 \
# --val_dataset sintel kitti \
# --lr 2e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 200000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1677566045.275271 \
# --classifier_checkpoint_train_acc 0.805 \
# --classifier_checkpoint_test_acc 0.802 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 1 twcc 2,3
# gmflow ctmixed+s
# batch 8, 2e-4
# name=gmflow-ctmixed+s-noc
# port=9988
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=2,3 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage mixed \
# --resume checkpoints/pretrained/gmflow_things-e9887eda.pth \
# --batch_size 8 \
# --val_dataset sintel kitti \
# --lr 2e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 3 twcc 0,1
# gmflow ad+s-c
# name=gmflow-ad+s-c-2e5-1
# port=9987
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage augmenteddiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1677566045.275271 \
# --classifier_checkpoint_train_acc 0.805 \
# --classifier_checkpoint_test_acc 0.802 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 5 twcc 0,1
# gmflow ad+s-c
# name=gmflow-ad+s-oldc-2e5-1
# port=9987
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage augmenteddiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1672944585.6048822 \
# --classifier_checkpoint_train_acc 0.675 \
# --classifier_checkpoint_test_acc 0.804 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 5 twcc 2,3
# gmflow fd+s-c
# name=gmflow-fd+s-oldc-2e5-1
# port=9988
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=2,3 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage flowdiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1672944585.6048822 \
# --classifier_checkpoint_train_acc 0.675 \
# --classifier_checkpoint_test_acc 0.804 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 5 twcc 4,5
# gmflow avd+s-c
# name=gmflow-avd+s-oldc-2e5-1
# port=9989
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=4,5 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage augmentedvemdiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1672944585.6048822 \
# --classifier_checkpoint_train_acc 0.675 \
# --classifier_checkpoint_test_acc 0.804 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 5 twcc 6,7
# gmflow fd+s-c
# name=gmflow-fd+s-c-2e5-1
# port=9990
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=6,7 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage flowdiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1677566045.275271 \
# --classifier_checkpoint_train_acc 0.805 \
# --classifier_checkpoint_test_acc 0.802 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 5 gmflow2 0,1
# gmflow ad+s-c
# name=gmflow-ad+s-c-5e5-1
# port=9987
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage augmenteddiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1677566045.275271 \
# --classifier_checkpoint_train_acc 0.805 \
# --classifier_checkpoint_test_acc 0.802 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00005 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 1 gmflow2 2,3
# gmflow ctmixed+s
# batch 8, 2e-4
# name=gmflow-ctmixed+s-c-5e5-1
# port=9988
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=2,3 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage mixed \
# --resume checkpoints/pretrained/gmflow_things-e9887eda.pth \
# --batch_size 8 \
# --val_dataset sintel kitti \
# --lr 2e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1677566045.275271 \
# --classifier_checkpoint_train_acc 0.805 \
# --classifier_checkpoint_test_acc 0.802 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00005 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 5 gmflow2 4,5
# gmflow fd+s-c
# name=gmflow-fd+s-c-5e5-1
# port=9989
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=4,5 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage flowdiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1677566045.275271 \
# --classifier_checkpoint_train_acc 0.805 \
# --classifier_checkpoint_test_acc 0.802 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00005 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 5 gmflow2 6,7
# gmflow fd+s-c
# name=gmflow-avd+s-c1-5e5-1
# port=9990
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=6,7 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage augmentedvemdiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1677566045.275271 \
# --classifier_checkpoint_train_acc 0.805 \
# --classifier_checkpoint_test_acc 0.802 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00005 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 6 gmflow 0,1
# gmflow ad+s-c
# name=gmflow-ad+s-c1-5e5-1
# port=9987
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage augmenteddiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1677566045.275271 \
# --classifier_checkpoint_train_acc 0.805 \
# --classifier_checkpoint_test_acc 0.802 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00005 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 6 gmflow 2,3
# gmflow ad+s-c
# name=gmflow-ad+s-c1-2e5-1
# port=9988
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=2,3 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage augmenteddiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --add_classifier \
# --classifier_checkpoint_timestamp 1677566045.275271 \
# --classifier_checkpoint_train_acc 0.805 \
# --classifier_checkpoint_test_acc 0.802 \
# --classify_loss_weight_init 1 \
# --classify_loss_weight_increase -0.00002 \
# --max_classify_loss_weight 1 \
# --min_classify_loss_weight 0 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log

# Mar 6 gmflow 4,5
# gmflow ad+s-c
# name=gmflow-ad+s-noc-2nd
# port=9989
# NUM_GPUS=2

# CHECKPOINT_DIR=checkpoints/${name} && \
# mkdir -p ${CHECKPOINT_DIR} && \
# CUDA_VISIBLE_DEVICES=4,5 \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage augmenteddiml \
# --batch_size 16 \
# --val_dataset sintel kitti \
# --lr 4e-4 \
# --image_size 368 560 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${checkpoint_dir}/train.log


# Mar 6 gmflow 6,7
# gmflow ad+s-c
name=gmflow-ad+s-c1-5e5-1-3rd
port=9990
NUM_GPUS=2

CHECKPOINT_DIR=checkpoints/${name} && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${port} main.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage augmenteddiml \
--batch_size 16 \
--val_dataset sintel kitti \
--lr 4e-4 \
--image_size 368 512 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 1000 \
--save_ckpt_freq 10000 \
--num_steps 100000 \
--add_classifier \
--classifier_checkpoint_timestamp 1677566045.275271 \
--classifier_checkpoint_train_acc 0.805 \
--classifier_checkpoint_test_acc 0.802 \
--classify_loss_weight_init 1 \
--classify_loss_weight_increase -0.00005 \
--max_classify_loss_weight 1 \
--min_classify_loss_weight 0 \
2>&1 | tee -a ${checkpoint_dir}/train.log
