#!/bin/bash

SCENES_DIR="./data/GSCollision/scene_configs"
OUTPUT_BASE="./data/GSCollision/scenes"

for json_file in "$SCENES_DIR"/*.json; do
  base_name=$(basename "$json_file" .json)
  if [[ ! "$base_name" =~ ^2 ]]; then
      continue
  fi
  output_dir="$OUTPUT_BASE/$base_name"
  mkdir -p "$output_dir"
  
  sbatch <<EOT
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=scene_${base_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time 12:00:00
#SBATCH --output=outerr/scene_${base_name}_%j.out
#SBATCH --error=outerr/scene_${base_name}_%j.err

set -euo pipefail

python -m dataset.generate_scene \
  --input "$json_file" \
  --output "$output_dir"

echo "Completed generation for $base_name"
EOT
done