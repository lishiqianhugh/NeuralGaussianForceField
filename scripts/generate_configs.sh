#! /bin/bash

#SBATCH --partition=h100
#SBATCH --job-name=config2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time 2-23:00:00
#SBATCH --output=outerr/%j.out 
#SBATCH --error=outerr/%j.err

python -m dataset.generate_configs --obj_num 2 --scene_num 5