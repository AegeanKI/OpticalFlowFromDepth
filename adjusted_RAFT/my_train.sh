#!/bin/bash
mkdir -p checkpoints
# python -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 224 224 --wdecay 0.0001
# python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001
# python -u train.py --name raft-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85
# python -u train.py --name raft-kitti  --stage kitti --validation kitti --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 1 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85

# python -u train.py --name raft-augmentedredweb-classifier-0005 --stage augmentedredweb --validation kitti --gpus 0 1 --num_steps 1000000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
# python -u train.py --name raft-augmentedredweb-classifier-0001 --stage augmentedredweb --validation kitti --gpus 2 3 --num_steps 1000000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
# python -u train.py --name raft-augmentedredweb-classifier-0 --stage augmentedredweb --validation kitti --gpus 4 5 --num_steps 1000000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
# python -u train.py --name raft-augmentedredweb --stage augmentedredweb --validation kitti --gpus 6 7 --num_steps 1000000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001


# python -u train.py --name raft-augmentedredweb-classifier-000002-1-0 --stage augmentedredweb \
#     --validation kitti \
#     --gpus 3 --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0 \
#     --mixed_precision

# python -u train.py --name raft-augmentedredweb-classifier-0000002-01-002 --stage augmentedredweb \
#     --validation kitti \
#     --gpus 2 --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 --add_classifier \
#     --classifier_checkpoint_timestamp 1671161250.3501413 \
#     --classifier_checkpoint_train_acc 0.763 \
#     --classifier_checkpoint_test_acc 0.774 \
#     --classify_loss_weight_init 0.1 \
#     --classify_loss_weight_increase -0.000002 \
#     --max_classify_loss_weight 0.1 \
#     --min_classify_loss_weight 0.02 \
#     --mixed_precision

# python -u train.py --name raft-augmentedredweb-no-classifier --stage augmentedredweb \
#     --validation kitti \
#     --gpus 4 --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001

# python -u train.py --name raft-augmenteddiml-no-classifier-368-768 --stage augmenteddiml \
#     --validation kitti \
#     --gpus 3 --num_steps 30000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 768 --wdecay 0.0001

# python -u train.py --name raft-augmenteddiml-512-1382-384-1152-1e5-1 --stage augmenteddiml \
# python -u train.py --name raft-augmenteddiml-436-1024-384-896-1e5-1 --stage augmenteddiml \
#     --validation kitti \
#     --gpus 7 --num_steps 120000 --batch_size 4 --lr 0.00025 --val_freq 1000 \
#     --early_stop 30000 \
#     --image_size 384 896 --wdecay 0.0001 --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00001 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0 \
#     --mixed_precision

# ===========
# Jan22
# name=raft-ar+s-c-2e5-1
# gpu=7

# python -u train.py --name ${name} --stage augmentedredweb \
#     --validation kitti \
#     --gpus $gpu --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --early_stop 16000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0 \

# python -u train.py --name $name-t --stage things \
#     --validation kitti \
#     --restore_ckpt checkpoints/$name.pth \
#     --gpus $gpu --num_steps 120000 --batch_size 5 --lr 0.0001 --val_freq 1000\
#     --image_size 400 720 --wdecay 0.0001 \
#     --mixed_precision

# ============
# Jan27
# name=raft-ar+s-c-2e5-1-3rd
# gpu=7

# python -u train.py --name ${name} --stage augmentedredweb \
#     --validation kitti \
#     --gpus $gpu --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --early_stop 16000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# ============
# Jan28 29
# name=raft-ar+s-c-2e5-1-7th
# gpu=7

# python -u train.py --name ${name} --stage augmentedredweb \
#     --validation kitti \
#     --gpus $gpu --num_steps 120000 --batch_size 8 --lr 0.0004 --val_freq 1000 \
#     --early_stop 30000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss 0 \
#     --mixed_precision

# Jan30
# check first train 1k
# name=raft-ar+s-c-2e5-1-8th
# gpu=7

# python -u train.py --name ${name} --stage augmentedredweb \
#     --validation kitti \
#     --restore_ckpt checkpoints/1000_raft-ar+s-noc.pth \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0 \
#     --mixed_precision

# Jan 30
# check DIML LR replace img1
# name=raft-ad-s-c-2e5-1
# gpu=7

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0 \
#     --mixed_precision

# # Feb 1
# # check AD_s + FT
# name=raft-ad-s-c-2e5-1
# gpu=7

# python -u train.py --name $name-t --stage things \
#     --validation kitti \
#     --restore_ckpt checkpoints/$name.pth \
#     --gpus $gpu --num_steps 120000 --batch_size 5 --lr 0.0001 --val_freq 1000 \
#     --image_size 400 720 --wdecay 0.0001 \
#     --mixed_precision

# Feb 3
# check AD_s + FT, lr 0.00005
# name=raft-ad-s-c-2e5-1
# gpu=7

# python -u train.py --name $name-t --stage things \
#     --validation kitti \
#     --restore_ckpt checkpoints/$name.pth \
#     --gpus $gpu --num_steps 120000 --batch_size 5 --lr 0.00005 --val_freq 1000 \
#     --image_size 400 720 --wdecay 0.0001 \
#     --mixed_precision
#     --mixed_precision

# Feb 3
# check AD_s + FT, lr 0.00005
# name=raft-ad-s-c-2e5-1
# gpu=7

# python -u train.py --name $name-t --stage things \
#     --validation kitti \
#     --restore_ckpt checkpoints/$name.pth \
#     --gpus $gpu --num_steps 120000 --batch_size 5 --lr 0.00005 --val_freq 1000 \
#     --image_size 400 720 --wdecay 0.0001 \
#     --mixed_precision

# Feb 3
# check AD+s
# name=raft-ad+s-c-2e5-1
# gpu=3

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0 \
#     --mixed_precision

# Feb 4
# check test AR
# name=raft-test-ar-s-c-2e5-1
# gpu=3

# python -u train.py --name ${name} --stage test-augmentedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0 \
#     --mixed_precision

# Feb 8
# check test_AR

# name=raft-ad+s-c-2e5-1
# gpu=3

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0 \
#     --mixed_precision

# Feb 9
# check test AR (>350, >350), resize (480, 640), 1698 images
# name=raft-test-ar+s-c-2e5-1
# gpu=3

# python -u train.py --name ${name} --stage test-augmentedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0 \
#     --mixed_precision


# Feb 9
# check test FR (>350, >350), resize (480, 640), 1698 images, FD
# name=raft-test-fd+s-c-2e5-1
# gpu=7

# python -u train.py --name ${name} --stage flowdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0 \
#     --mixed_precision


# Feb 15 eva3
# # check VEMDIML, FlowDIML again
# # gpu 4 VEMDIML, gpu 5 FlowDIML
# name=raft-fd+s-noc
# gpu=5

# python -u train.py --name ${name} --stage flowdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision
#     # --add_classifier \
#     # --classifier_checkpoint_timestamp 1672944585.6048822 \
#     # --classifier_checkpoint_train_acc 0.675 \
#     # --classifier_checkpoint_test_acc 0.804 \
#     # --classify_loss_weight_init 1 \
#     # --classify_loss_weight_increase -0.00002 \
#     # --max_classify_loss_weight 1 \
#     # --min_classify_loss_weight 0 \

# Feb 15 eva3
# check VEMReDWeb, FlowReDWeb again
# gpu 6 VEMReDWeb, gpu 7 FlowReDWeb
name=raft-test-fr+s-noc
gpu=7

python -u train.py --name ${name} --stage test-flowredweb \
    --validation kitti \
    --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
    --image_size 368 496 --wdecay 0.0001 \
    --mixed_precision
    # --add_classifier \
    # --classifier_checkpoint_timestamp 1672944585.6048822 \
    # --classifier_checkpoint_train_acc 0.675 \
    # --classifier_checkpoint_test_acc 0.804 \
    # --classify_loss_weight_init 1 \
    # --classify_loss_weight_increase -0.00002 \
    # --max_classify_loss_weight 1 \
    # --min_classify_loss_weight 0 \



