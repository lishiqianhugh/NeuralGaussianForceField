#! /bin/bash

#SBATCH --partition=h100
#SBATCH --job-name=ngff
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time 2-23:00:00
#SBATCH --nodelist=node008
#SBATCH --output=slurm_logs/%j.out 
#SBATCH --error=slurm_logs/%j.err

accelerate launch --multi_gpu --num_processes=8 --main_process_port=29601 -m dynamic_models.train \
    --num_keypoints 2048 \
    --ode_method euler \
    --step_size 1e-2 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --hidden_dim 256 \
    --num_layers 4 \
    --batch_size 50 \
    --steps 80 \
    --chunk 80 \
    --epochs 2000 \
    --dynamic_model ngff \
    --sample_num 3000 \
    --threshold 5e-2 \
    --save_interval 50 \
    --training_tracker swanlab

# python -m dynamic_models.train_single_profiler \
#     --num_keypoints 2048 \
#     --ode_method euler \
#     --step_size 2e-2 \
#     --lr 1e-4 \
#     --min_lr 1e-6 \
#     --hidden_dim 256 \
#     --num_layers 4 \
#     --batch_size 50 \
#     --steps 80 \
#     --chunk 80 \
#     --epochs 5 \
#     --dynamic_model ngff \
#     --sample_num 50 \
#     --threshold 5e-2 \
#     --save_interval 50 
# pointformer
# python -u train.py --num_keypoints 2048 --lr 1e-3 --min_lr 1e-4 --hidden_dim 128 --num_layers 3 --batch_size 8 --steps 80 --chunk 80 --epoch 2001 --dynamic_model pointformer --sample_num 1

# segnn
# python -u train.py --num_keypoints 1024 --lr 5e-4 --min_lr 5e-5 --hidden_dim 64 --num_layers 1 --batch_size 30 --steps 60 --chunk 60 --epochs 2000 --dynamic_model segno

# gcn
# python -u train.py --num_keypoints 2048 --lr 1e-3 --min_lr 1e-4 --hidden_dim 128 --num_layers 4 --batch_size 30 --steps 80 --chunk 80 --epochs 500 --dynamic_model gcn --sample_num 3000

# sgnn
# python -u train.py --num_keypoints 2048 --lr 6e-4 --min_lr 2e-4 --hidden_dim 80 --num_layers 1 --batch_size 2 --steps 80 --chunk 80 --epochs 1001 --dynamic_model sgnn --sample_num 2 --save_interval 10
# ngff

# python -u train.py \
# --num_keypoints 2048 \
# --ode_method euler \
# --step_size 2e-2 \
# --lr 1e-3 \
# --min_lr 1e-4 \
# --hidden_dim 128 \
# --num_layers 4 \
# --batch_size 14 \
# --steps 80 \
# --chunk 80 \
# --epochs 1000 \
# --dynamic_model ngff \
# --sample_num 2700

# python -u train.py --num_keypoints 2048 --ode_method euler --step_size 2e-2 --lr 3e-4 --min_lr 1e-4 --hidden_dim 200 --num_layers 4 --batch_size 10 --steps 80 --chunk 80 --epochs 2000 --dynamic_model ngff --sample_num 20 --threshold 5e-2