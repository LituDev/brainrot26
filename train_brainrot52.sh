#!/bin/bash

python train.py \
    --d_model 384 \
    --nhead 6 \
    --num_layers 8 \
    --dim_feedforward 1536 \
    --dropout 0.1 \
    --batch_size 16 \
    --iterations 100 \
    --seq_length 1024 \
    --num_samples 100 \
    --rot 52 \
    --learning_rate 0.0001 \
    --wandb_project "brainrot52" \
    --checkpoint_dir "checkpoints"