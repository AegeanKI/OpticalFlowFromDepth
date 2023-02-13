#!/bin/sh


gpu=2
dataset=test-augmentedredweb
dropout=0.6
output_dim=64
batch_size=8
max_epoch=30
lr=0.0002
python train_classifier.py --gpu $gpu \
    --dataset $dataset \
    --use_small \
    --use_depth_in_classifier \
    --dropout $dropout \
    --max_epoch $max_epoch \
    --output_dim $output_dim \
    --use_dropout_in_encoder \
    --use_average_pooling \
    --lr $lr \
    --batch_size $batch_size
