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


python -u train.py --name raft-augmentedredweb-classifier-0-02-02 --stage augmentedredweb \
    --validation kitti \
    --gpus 7 --num_steps 120000 --batch_size 8 --lr 0.00025 --val_freq 1000 \
    --image_size 368 496 --wdecay 0.0001 --add_classifier \
    --classifier_checkpoint_timestamp 1671161250.3501413 \
    --classifier_checkpoint_train_acc 0.763 \
    --classifier_checkpoint_test_acc 0.774 \
    --classify_loss_weight_init 0.2 \
    --classify_loss_weight_increase 0 \
    --max_classify_loss_weight 0.2 \
    --min_classify_loss_weight 0.2 \
    --mixed_precision
