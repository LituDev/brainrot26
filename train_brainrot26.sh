#!/bin/bash

python train.py \
    --d_model 256 \
    --nhead 4 \
    --num_layers 6 \
    --dim_feedforward 1024 \
    --dropout 0.1 \
    --batch_size 8 \
    --iterations 10 \
    --seq_length 1024 \
    --num_samples 100 \
    --rot 26 \
    --learning_rate 0.0001 \
    --wandb_project "brainrot26" \
    --checkpoint_dir "checkpoints"