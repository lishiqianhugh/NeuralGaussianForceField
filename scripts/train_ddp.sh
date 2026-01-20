#!/bin/bash

#SBATCH --partition=h100
#SBATCH --job-name=ngff_8
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --time=2-23:00:00
#SBATCH --output=outerr/%j.out
#SBATCH --error=outerr/%j.err

echo "Running on $SLURM_JOB_NODELIST"
echo "Using $SLURM_GPUS GPUs"

# # ngff
PYTHONUNBUFFERED=1 MASTER_PORT=29500 torchrun --nproc-per-node=8 --master-port=29500 train_ddp.py \
--num_keypoints 2048 \
--ode_method euler \
--step_size 2e-2 \
--lr 1e-5 \
--min_lr 1e-7 \
--hidden_dim 200 \
--num_layers 4 \
--batch_size 9 \
--steps 80 \
--chunk 80 \
--epochs 1001 \
--sample_num 3000 \
--threshold 5e-2 \
--reload ./exps/ngff/out_2025-09-19-15-25-29/ngff_best.pth \
--save_interval 1

# # sgnn
# PYTHONUNBUFFERED=1 MASTER_PORT=29501 torchrun --nproc-per-node=8 --master-port=29501 train_ddp.py \
# --num_keypoints 2048 \
# --lr 1e-4 \
# --min_lr 1e-6 \
# --hidden_dim 80 \
# --num_layers 1 \
# --batch_size 2 \
# --steps 80 \
# --chunk 80 \
# --epochs 2001 \
# --sample_num 3000

# pointformer
# PYTHONUNBUFFERED=1 MASTER_PORT=19500 torchrun --nproc-per-node=4 --master-port=19500 train_ddp.py \
# --num_keypoints 2048 \
# --lr 5e-4 \
# --min_lr 5e-6 \
# --hidden_dim 128 \
# --num_layers 3 \
# --batch_size 8 \
# --steps 80 \
# --chunk 80 \
# --epochs 2001 \
# --sample_num 3000 \
# --dynamic_model pointformer