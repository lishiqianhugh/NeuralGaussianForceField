#! /bin/bash

#SBATCH --partition=h100
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --output=outerr/%j.out
#SBATCH --error=outerr/%j.err

echo "Running on $SLURM_JOB_NODELIST"
echo "Using $SLURM_GPUS GPUs"

CONFIG=./config/dynamic_config.json
GROUP='3_9'

SCENES=(
  2740_pillow_bowl_miku
  2901_cloth_ball_rope
  2749_pillow_panda_miku
)

for SCENE in "${SCENES[@]}"; do
  echo "=== Running scene: $SCENE ==="

  # NGFF
  python -m videogen.eval \
    --model_path ./data/GSCollision/scenes/${GROUP}/${SCENE} \
    --dynamic_model ngff \
    --config $CONFIG \
    --output_path ./output/dynamic_prediction/ngff/${SCENE} \
    --compile_video \
    --white_b \
    --single_view 0

  # PointFormer
  python -m videogen.eval \
    --model_path ./data/GSCollision/scenes/${GROUP}/${SCENE} \
    --dynamic_model pointformer \
    --config $CONFIG \
    --output_path ./output/dynamic_prediction/pointformer/${SCENE} \
    --compile_video \
    --white_b \
    --single_view 0

  # GCN
  python -m videogen.eval \
    --model_path ./data/GSCollision/scenes/${GROUP}/${SCENE} \
    --dynamic_model gcn \
    --config $CONFIG \
    --output_path ./output/dynamic_prediction/gcn/${SCENE} \
    --compile_video \
    --white_b \
    --single_view 0

  # GT (MPM)
  python -m dataset.render \
    --model_path ./data/GSCollision/mpm/${GROUP}/${SCENE} \
    --config $CONFIG \
    --output_path ./output/dynamic_prediction/gt/${SCENE} \
    --compile_video \
    --white_bg \
    --single_view 0

  # MPM-VLM
  python -m dataset.render \
    --model_path ./data/GSCollision/mpm_vlm/${GROUP}/${SCENE} \
    --config $CONFIG \
    --output_path ./output/dynamic_prediction/mpm_vlm/${SCENE} \
    --compile_video \
    --white_bg \
    --single_view 0
done
