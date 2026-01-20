#!/bin/bash

# Submit one sbatch job per subdirectory under the mpm directory.
# Each job will loop through scenes inside that mpm subdir and all backgrounds.

MPM_ROOT="./data/GSCollision/mpm"
BACKGROUNDS_ROOT="./data/GSCollision/backgrounds_pt"
OUTPUT_BASE="./data/GSCollision/dynamic"
INITIAL_BASE="./data/GSCollision/initial"

for mpm_subdir in "$MPM_ROOT"/*; do
    if [ ! -d "$mpm_subdir" ]; then
        continue
    fi
    mpm_name=$(basename "$mpm_subdir")
    echo "Submitting sbatch for mpm subdir: $mpm_name"

    sbatch <<EOT
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=render_${mpm_name}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --time=2-23:00:00
#SBATCH --output=outerr/render_${mpm_name}_%j.out
#SBATCH --error=outerr/render_${mpm_name}_%j.err

set -euo pipefail

for scene_dir in "$MPM_ROOT/${mpm_name}"/*; do
    if [ ! -d "\$scene_dir" ]; then
        continue
    fi
    scene_name=\$(basename "\$scene_dir")
    for background_dir in "$BACKGROUNDS_ROOT"/*; do
        background_name=\$(basename "\$background_dir")
        out_dir="$OUTPUT_BASE/${mpm_name}/\$scene_name/\$background_name/"
        if [ -d "\$out_dir" ]; then
            echo "Output exists, skipping: \$out_dir"
            continue
        fi
        echo "Running compile for \$scene_name with background \$background_name (mpm: ${mpm_name})"
        python -m dataset.render --model_path "\$scene_dir" --background_path "\$background_dir" --config config/dynamic_config.json --output_path "$OUTPUT_BASE/${mpm_name}/\$scene_name/\$background_name/" --compile_video --white_bg --single_view -1
    done
done
EOT

done


# submit initial render job: this job will loop over ALL mpm subdirectories and render initial frames

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=render_initial
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --time=2-23:00:00
#SBATCH --output=outerr/render_initial_%j.out
#SBATCH --error=outerr/render_initial_%j.err

set -euo pipefail

for mpm_subdir in "$MPM_ROOT"/*; do
    if [ ! -d "\$mpm_subdir" ]; then
        continue
    fi
    mpm_name=\$(basename "\$mpm_subdir")
    for scene_dir in "\$mpm_subdir"/*; do
        if [ ! -d "\$scene_dir" ]; then
            continue
        fi
        scene_name=\$(basename "\$scene_dir")
        for background_dir in "$BACKGROUNDS_ROOT"/*; do
            background_name=\$(basename "\$background_dir")
            out_dir="$INITIAL_BASE/\$mpm_name/\$scene_name/\$background_name/"
            if [ -d "\$out_dir" ]; then
                echo "Output exists, skipping: \$out_dir"
                continue
            fi
            echo "Running initial render for \$scene_name with background \$background_name (mpm: \$mpm_name)"
            python -m dataset.render --model_path "\$scene_dir" --background_path "\$background_dir" --config config/dynamic_config.json --initial_path "$INITIAL_BASE/\$mpm_name/\$scene_name/\$background_name/" --compile_video --white_bg
        done
    done
done
EOT