#!/bin/bash
#SBATCH --job-name=aced-marl-train
#SBATCH --output=runs/tmp/%A_%a/output_%A_%a.txt
#SBATCH --error=runs/tmp/%A_%a/error_%A_%a.txt
#SBATCH --time=0-10:0
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --array=0-3

# change this to your script path
SCRIPT_DIR="."
experiment_path="${SCRIPT_DIR}/src/train_mappo.py"
config_dir="${SCRIPT_DIR}/src/configs"

# Config permutations
configs=(
  # "quick_test.yaml"         # Quick test
  "mappo_sync.yaml"         # MAPPO baseline
  "mappo_event.yaml"        # MAPPO + event driven
  "mappo_atoc_sync.yaml"    # MAPPO + ATOC
  "mappo_atoc_event.yaml"   # MAPPO + ATOC + event driven
  # "mat_sync.yaml"           # MAT baseline
  # "mat_event.yaml"          # MAT + event driven
  # "mat_atoc_sync.yaml"      # MAT + ATOC
  # "mat_atoc_event.yaml"     # MAT + ATOC + event driven
)

cfg="${configs[$SLURM_ARRAY_TASK_ID]}"

# Source conda.sh script to enable conda commands
source $(conda info --base)/etc/profile.d/conda.sh
conda activate aced-marl

# Run the experiment
python "${experiment_path}" --config "${config_dir}/${cfg}"
