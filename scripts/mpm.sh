#!/bin/bash

BASE_DIR="./data/GSCollision/scenes"
OUTPUT_BASE="./data/GSCollision/mpm"
CONFIG="./config/dynamic_config.json"

for scene_group in "$BASE_DIR"/*; do
  if [ -d "$scene_group" ]; then
    group_name=$(basename "$scene_group")

    if [[ "$group_name" != "2" ]]; then
      continue
    fi

    sbatch <<EOT
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=mpm_${group_name}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=2-23:00:00
#SBATCH --output=outerr/mpm_${group_name}_%j.out
#SBATCH --error=outerr/mpm_${group_name}_%j.err

scenes=( "$scene_group"/* )
total=\${#scenes[@]}
count=0

for scene_dir in "\${scenes[@]}"; do
  if [ -d "\$scene_dir" ]; then
    count=\$((count+1))
    scene_name=\$(basename "\$scene_dir")
    out_dir="$OUTPUT_BASE/${group_name}/\${scene_name}"

    if [ -d "\$out_dir" ]; then
      echo "[SKIP] \$out_dir already exists."
      continue
    fi

    printf "[%d/%d] Running %s/%s\n" "\$count" "\$total" "${group_name}" "\$scene_name"

    python -m dataset.gs_simulation_scene \
      --model_path "\$scene_dir" \
      --config "$CONFIG" \
      --output_path "\$out_dir" \
      --save_h5
  fi
done
EOT

  fi
done
