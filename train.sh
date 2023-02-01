#!/bin/sh

python train_classifier.py --use_small --dropout=0.9 --output_dim=64 --use_depth_in_classifier\
    --use_dropout_in_encoder --use_dropout_in_classify --use_average_pooling \
    --batch_size 1 --dataset AugmentedReDWeb \
    --lr=0.00002 --lr_decay=0.000001 --min_lr=0.000002 --gpu=0 
