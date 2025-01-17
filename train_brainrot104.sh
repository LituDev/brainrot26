#!/bin/bash

python train.py \
    --d_model 512 \
    --nhead 8 \
    --num_layers 10 \
    --dim_feedforward 2048 \
    --dropout 0.1 \
    --batch_size 16 \
    --iterations 100 \
    --seq_length 1024 \
    --num_samples 100 \
    --rot 104 \
    --learning_rate 0.0001 \
    --wandb_project "brainrot104" \
    --checkpoint_dir "checkpoints"