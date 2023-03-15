#!/bin/sh

# Feb 13 eva2 
# train classifier on AD+s, test-AR+s

# gpu=2
# dataset=test-augmentedredweb
# dropout=0.6
# output_dim=64
# batch_size=8
# max_epoch=30
# lr=0.0002
# python train_classifier.py --gpu $gpu \
#     --dataset $dataset \
#     --use_small \
#     --use_depth_in_classifier \
#     --dropout $dropout \
#     --max_epoch $max_epoch \
#     --output_dim $output_dim \
#     --use_dropout_in_encoder \
#     --use_average_pooling \
#     --lr $lr \
#     --batch_size $batch_size

# Feb 14 eva2
# train classifier on AD+s, test-AR+s without dropout
# lr 0.0005 epoch 50
# with depth vs without depth

# gpu=3
# dataset=test-augmentedredweb
# output_dim=64
# batch_size=8
# max_epoch=50
# lr=0.0005
# python train_classifier.py --gpu $gpu \
#     --dataset $dataset \
#     --use_small \
#     --max_epoch $max_epoch \
#     --output_dim $output_dim \
#     --use_average_pooling \
#     --lr $lr \
#     --batch_size $batch_size
#     # --use_depth_in_classifier \

# Feb 15 eva2
# train classifier on merge (AD+s, test-AR+s) without dropout
# lr 0.0005
# epoch 50, 100
# with depth vs without depth

# gpu=2
# dataset=merge
# output_dim=64
# batch_size=8
# max_epoch=100
# lr=0.0005
# python train_classifier.py --gpu $gpu \
#     --dataset $dataset \
#     --use_small \
#     --max_epoch $max_epoch \
#     --output_dim $output_dim \
#     --use_average_pooling \
#     --lr $lr \
#     --batch_size $batch_size
#     --use_depth_in_classifier \

# Feb 28 twcc
# train classifier on merge (AD+s, test-AR+s) without dropout
# lr 0.0005, epoch 50
# normalize vs not normalize

# gpu=0
# dataset=mixed
# output_dim=64
# batch_size=16
# max_epoch=50
# lr=0.0005
# python train_classifier.py --gpu $gpu \
#     --dataset $dataset \
#     --use_small \
#     --max_epoch $max_epoch \
#     --output_dim $output_dim \
#     --use_average_pooling \
#     --not_normalize_dataset \
#     --lr $lr \
#     --batch_size $batch_size

# gpu=1
# dataset=mixed
# output_dim=64
# batch_size=16
# max_epoch=50
# lr=0.0005
# python train_classifier.py --gpu $gpu \
#     --dataset $dataset \
#     --use_small \
#     --max_epoch $max_epoch \
#     --output_dim $output_dim \
#     --use_average_pooling \
#     --lr $lr \
#     --batch_size $batch_size

# Mar 1
# mixed, batch 16 epoch 50, depth
# dim 64 vs 128
# not normalize vs normalize

# gpu=0
# dataset=mixed
# output_dim=64
# batch_size=16
# max_epoch=50
# lr=0.0005
# python train_classifier.py --gpu $gpu \
#     --dataset $dataset \
#     --use_small \
#     --use_depth_in_classifier \
#     --not_normalize_dataset \
#     --max_epoch $max_epoch \
#     --output_dim $output_dim \
#     --use_average_pooling \
#     --lr $lr \
#     --batch_size $batch_size

# gpu=1
# dataset=mixed
# output_dim=64
# batch_size=16
# max_epoch=50
# lr=0.0005
# python train_classifier.py --gpu $gpu \
#     --dataset $dataset \
#     --use_small \
#     --use_depth_in_classifier \
#     --max_epoch $max_epoch \
#     --output_dim $output_dim \
#     --use_average_pooling \
#     --lr $lr \
#     --batch_size $batch_size

# gpu=2
# dataset=mixed
# output_dim=128
# batch_size=16
# max_epoch=50
# lr=0.0005
# python train_classifier.py --gpu $gpu \
#     --dataset $dataset \
#     --use_small \
#     --use_depth_in_classifier \
#     --not_normalize_dataset \
#     --max_epoch $max_epoch \
#     --output_dim $output_dim \
#     --use_average_pooling \
#     --lr $lr \
#     --batch_size $batch_size

# gpu=3
# dataset=mixed
# output_dim=128
# batch_size=16
# max_epoch=50
# lr=0.0005
# python train_classifier.py --gpu $gpu \
#     --dataset $dataset \
#     --use_small \
#     --use_depth_in_classifier \
#     --max_epoch $max_epoch \
#     --output_dim $output_dim \
#     --use_average_pooling \
#     --lr $lr \
#     --batch_size $batch_size


# Mar 2 twcc
# train classifier on merge (AD+s, test-AR+s) without dropout
# lr 0.0002, epoch 100, batch 32, not normalize, output_dim 64
# no depth vs depth

# gpu=0
# dataset=mixed
# output_dim=64
# batch_size=32
# max_epoch=100
# lr=0.0002
# python train_classifier.py --gpu $gpu \
#     --dataset $dataset \
#     --use_small \
#     --max_epoch $max_epoch \
#     --output_dim $output_dim \
#     --use_average_pooling \
#     --not_normalize_dataset \
#     --lr $lr \
#     --batch_size $batch_size

gpu=1
dataset=mixed
output_dim=64
batch_size=32
max_epoch=100
lr=0.0002
python train_classifier.py --gpu $gpu \
    --dataset $dataset \
    --use_small \
    --max_epoch $max_epoch \
    --output_dim $output_dim \
    --use_average_pooling \
    --use_depth_in_classifier \
    --not_normalize_dataset \
    --lr $lr \
    --batch_size $batch_size

