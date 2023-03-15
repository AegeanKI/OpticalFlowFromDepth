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
# name=raft-test-fr+s-noc
# gpu=7

# python -u train.py --name ${name} --stage test-flowredweb \
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


# Feb 28 twcc 0
# check AD+s
# name=raft-ad+s-noc
# gpu=0

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision

# Feb 28 twcc 1
# check FD+s
# name=raft-fd+s-noc
# gpu=1

# python -u train.py --name ${name} --stage flowdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision

# Feb 28 twcc 2
# check AR+s
# name=raft-ar+s-noc
# gpu=2

# python -u train.py --name ${name} --stage augmentedfiltedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision

# Feb 28 twcc 3
# check FR+s
# name=raft-fr+s-noc
# gpu=3

# python -u train.py --name ${name} --stage flowfiltedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision

# Mar 1 twcc 1
# check VD+s
# name=raft-vd+s-noc
# gpu=1

# python -u train.py --name ${name} --stage vemdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision

# Mar 1 twcc 3
# check VR+s
# name=raft-vr+s-noc
# gpu=3

# python -u train.py --name ${name} --stage vemfiltedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision

# Mar 1 twcc 0
# ReDWeb 432, 560
# check AR+s (2nd)
# name=raft-ar+s-noc
# gpu=0

# python -u train.py --name ${name} --stage augmentedfiltedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 432 560 --wdecay 0.0001 \
#     --mixed_precision

# Mar 1 twcc 2
# ReDWeb 432, 560
# check FR+s
# name=raft-fr+s-noc-2nd
# gpu=2

# python -u train.py --name ${name} --stage flowfiltedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 432 560 --wdecay 0.0001 \
#     --mixed_precision

# Mar 1 twcc 1
# check VR+s
# name=raft-vr+s-noc-2nd
# gpu=1

# python -u train.py --name ${name} --stage vemfiltedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 432 560 --wdecay 0.0001 \
#     --mixed_precision

# Mar 1 twcc 3
# check ad+s-c
# name=raft-ad+s-c-2e5-1
# gpu=3

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 2 twcc 2
# check ar+s-c
# name=raft-ar+s-c-2e5-1
# gpu=2

# python -u train.py --name ${name} --stage augmentedfiltedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 432 560 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 2 twcc 0
# check mixed-noc
# name=raft-mixed-noc
# gpu=0

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 560 --wdecay 0.0001 \
#     --mixed_precision

# Mar 2 twcc 1
# check mixed-c-2e5-1
# name=raft-mixed-c-2e5-1
# gpu=1

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 560 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0


# Mar 3 twcc 0
# check ar+s-noc
# name=raft-ar+s-noc
# gpu=0

# python -u train.py --name ${name} --stage augmentedfiltedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 560 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision


# Mar 3 twcc 1
# check fr+s-noc
# name=raft-fr+s-noc
# gpu=1

# python -u train.py --name ${name} --stage flowfiltedredweb \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 560 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision


# Mar 3 twcc 2
# check fd+s-noc
# name=raft-fd+s-noc
# gpu=2

# python -u train.py --name ${name} --stage flowdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 560 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision

# Mar 3 twcc 3
# check mixed-c-2e5-1
# name=raft-mixed-c-2e5-1
# gpu=3

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 560 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# vd-noc, vr-noc
# mixed-noc
# mixed-c-low
# ar-c, ad-c

# Mar 3 twcc 2
# check ctmixed-noc
# kitti aug params
# name=raft-ctmixed-noc
# gpu=2

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 50000 --batch_size 5 --lr 0.0001 --val_freq 1000 \
#     --image_size 368 600 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/pretrained/raft-things.pth \
#     --mixed_precision

# Mar 3 twcc 0
# check ctmixed-c-2e5-1
# kitti aug params
# name=raft-ctmixed-c-2e5-1
# gpu=0

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 50000 --batch_size 5 --lr 0.0001 --val_freq 1000 \
#     --image_size 368 600 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/pretrained/raft-things.pth \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 raft 0
# check ctmixed-c-2e5-1
# kitti aug params
# name=raft-ctmixed-c-2e5-1
# gpu=0

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 50000 --batch_size 5 --lr 0.0001 --val_freq 1000 \
#     --image_size 368 600 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/pretrained/raft-things.pth \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 raft 1
# check-2e5-1
# name=raft-ad+s-c1-2e5-1
# gpu=1

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 raft 2
# check ad+s-c-2e5-1
# name=raft-ad+s-c2-2e5-1
# gpu=2

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 raft 3
# check ctmixed-c-5e5-1
# kitti aug params
# name=raft-ctmixed-c2-5e5-1
# gpu=3

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 50000 --batch_size 5 --lr 0.0001 --val_freq 1000 \
#     --image_size 368 600 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/pretrained/raft-things.pth \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00005 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 classifier 0
# check ctmixed-c1-5e5-1
# kitti aug params
# name=raft-ctmixed-c1-5e5-1
# gpu=0

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 50000 --batch_size 5 --lr 0.0001 --val_freq 1000 \
#     --image_size 368 600 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/pretrained/raft-things.pth \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00005 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 classifier 1
# check fd+s-c2-5e5-1
# name=raft-fd+s-c2-5e5-1
# gpu=1

# python -u train.py --name ${name} --stage flowdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00005 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 classifier 2
# check fd+s-c1-5e5-1
# name=raft-fd+s-c1-5e5-1
# gpu=2

# python -u train.py --name ${name} --stage flowdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00005 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 classifier 3
# check ad+s-c-5e5-1
# name=raft-ad+s-c1-5e5-1
# gpu=3

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00005 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 pre 0
# check vd+s-c2-2e5-1
# name=raft-vd+s-c2-2e5-1
# gpu=0

# python -u train.py --name ${name} --stage vemdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 pre 1
# check vd+s-c1-2e5-1
# name=raft-vd+s-c1-2e5-1
# gpu=1

# python -u train.py --name ${name} --stage vemdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 pre 0
# check avd+s-c1-2e5-1
# name=raft-avd+s-c1-2e5-1
# gpu=0

# python -u train.py --name ${name} --stage augmentedvemdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 raft 2
# check ad+s-c2-5e5-1
# name=raft-ad+s-c2-5e5-1
# gpu=2

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1672944585.6048822 \
#     --classifier_checkpoint_train_acc 0.675 \
#     --classifier_checkpoint_test_acc 0.804 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00005 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 5 raft 0
# check avd+s-c1-5e5-1
# name=raft-avd+s-c1-5e5-1
# gpu=0

# python -u train.py --name ${name} --stage augmentedvemdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00005 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 6 raft 3
# check ctmixed-noc
# kitti aug params
# name=raft-ctmixed-noc-2nd
# gpu=3

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 50000 --batch_size 5 --lr 0.0001 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/pretrained/raft-things.pth \
#     --mixed_precision

# Mar 6 classifier 0
# check ctmixed-c1-5e5-1
# name=raft-ctmixed-c1-5e5-1-2nd
# gpu=0

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 50000 --batch_size 5 --lr 0.0001 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/pretrained/raft-things.pth \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00005 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 6 classifier 0
# check ad+s-c3-2e5-1
# name=raft-ad+s-c3-2e5-1
# gpu=0

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.723 \
#     --classifier_checkpoint_test_acc 0.728 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 6 classifier 1
# check ad+s-c4-2e5-1
# name=raft-ad+s-c4-2e5-1
# gpu=1

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.569 \
#     --classifier_checkpoint_test_acc 0.691 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 6 classifier 2
# check ad+s-c5-2e5-1
# name=raft-ad+s-c5-2e5-1
# gpu=2

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.743 \
#     --classifier_checkpoint_test_acc 0.727 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 6 pre 1
# check ad+s-c
# name=raft-ad+s-c-2e5-1
# gpu=1

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0


# Mar 6 raft 3
# check ctmixed-c-5e5-1-2nd-finetune-noc
# name=raft-ctmixedfinetune-noc
# gpu=3

# python -u train.py --name ${name} --stage finetunekitti \
#     --validation finetunekitti \
#     --gpus ${gpu} --num_steps 30000 --batch_size 3 --lr 0.0001 --val_freq 1000 \
#     --image_size 288 960 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/raft-ctmixed-c1-5e5-1-2nd.pth \
#     --mixed_precision


# Mar 6 gmflow 6
# check mixed-c-2e5-1-finetune-noc
# name=raft-mixedfinetune-noc
# gpu=6

# python -u train.py --name ${name} --stage finetunekitti \
#     --validation finetunekitti \
#     --gpus ${gpu} --num_steps 30000 --batch_size 3 --lr 0.0001 --val_freq 1000 \
#     --image_size 288 960 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/raft-mixed-c-2e5-1.pth \
#     --mixed_precision


# Mar 6 gmflow 7
# check ctmixed-c-5e5-1-2nd-finetune-noc-2nd
# name=raft-ctmixedfinetune-noc-2nd
# gpu=7

# python -u train.py --name ${name} --stage finetunekitti \
#     --validation finetunekitti \
#     --gpus ${gpu} --num_steps 20000 --batch_size 3 --lr 0.0001 --val_freq 1000 \
#     --image_size 288 960 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/raft-ctmixed-c1-5e5-1-2nd.pth \
#     --mixed_precision

# Mar 6 gmflow 7
# check mixed-c-2e5-1-finetune-noc-2nd
# name=raft-mixedfinetune-noc-2nd
# gpu=7

# python -u train.py --name ${name} --stage finetunekitti \
#     --validation finetunekitti \
#     --gpus ${gpu} --num_steps 20000 --batch_size 3 --lr 0.0001 --val_freq 1000 \
#     --image_size 288 960 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/raft-mixed-c-2e5-1.pth \
#     --mixed_precision

# Mar 10 raft 0, 2 continuous
# check fd-c
# name=raft-fd+s-c1-2e5-1
# gpu=0

# python -u train.py --name ${name} --stage flowdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# # avd-c
# name=raft-avd+s-noc
# gpu=0

# python -u train.py --name ${name} --stage augmentedvemdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision

# Mar 12 raft 0
# check fix random_augment ad c (normal:augment = 1:1)
# name=raft-fixr-ad+s-c1-2e5-1
# gpu=0

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 12 raft 1
# check fix random_augment ad noc (normal:augment = 1:1)
# name=raft-fixr-ad+s-noc
# gpu=1

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --mixed_precision

# Mar 13 raft 0                log deleted
# check fixr-ctmixed-c1-5e5-1
# name=raft-fixr-ctmixed-c1-5e5-1
# gpu=0

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 50000 --batch_size 5 --lr 0.0001 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.00001 --gamma 0.85 \
#     --restore_ckpt checkpoints/pretrained/raft-things.pth \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00005 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 13 raft 1, ctrl-c
# check fix random_augment ad c (normal:augment = 1:1)
# name=raft-fixr-ad+s-c1-5e5-1
# gpu=1

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00005 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 13 raft2 0
# name=raft-avd+s-noc
# gpu=0

# python -u train.py --name ${name} --stage augmentedvemdiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision

# Mar 13 raft2 1
# check fix random_augment ad c (normal:augment = 1:1)
# name=raft-fixr-ad+s-c1-2e5-1
# gpu=1

# python -u train.py --name ${name} --stage augmenteddiml \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 14 raft 1
# check mixed-c-2e5-1
# name=raft-fixr-mixed-c-2e5-1
# gpu=1

# python -u train.py --name ${name} --stage mixed \
#     --validation kitti \
#     --gpus ${gpu} --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
#     --image_size 368 496 --wdecay 0.0001 \
#     --is_first_stage \
#     --mixed_precision \
#     --add_classifier \
#     --classifier_checkpoint_timestamp 1677566045.275271 \
#     --classifier_checkpoint_train_acc 0.805 \
#     --classifier_checkpoint_test_acc 0.802 \
#     --classify_loss_weight_init 1 \
#     --classify_loss_weight_increase -0.00002 \
#     --max_classify_loss_weight 1 \
#     --min_classify_loss_weight 0

# Mar 14 raft 0
# check ctfixrmixed-c-5e5-1-2nd-finetune-noc
name=raft-ctfixrmixedfinetune-noc
gpu=0

python -u train.py --name ${name} --stage finetunekitti \
    --validation finetunekitti \
    --gpus ${gpu} --num_steps 20000 --batch_size 3 --lr 0.0001 --val_freq 1000 \
    --image_size 288 960 --wdecay 0.00001 --gamma 0.85 \
    --restore_ckpt checkpoints/raft-fixr-ctmixed-c1-5e5-1.pth \
    --mixed_precision
