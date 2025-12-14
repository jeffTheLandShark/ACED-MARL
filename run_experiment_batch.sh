#!/bin/bash
#SBATCH --job-name=aced-marl-train
#SBATCH --output=runs/tmp/%A_%a/output_%A_%a.txt
#SBATCH --error=runs/tmp/%A_%a/error_%A_%a.txt
#SBATCH --time=0-10:0
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --array=0-3

# change this to your script path
SCRIPT_DIR="/home/ad.msoe.edu/goetschm/CSC5661/final-project"
experiment_path="${SCRIPT_DIR}/ACED-MARL/src/train_mappo.py"
config_dir="${SCRIPT_DIR}/ACED-MARL/src/configs"

# Config permutations
configs=(
  "mappo_sync.yaml"         # MAPPO baseline
  "mappo_event.yaml"        # MAPPO + event driven
  "mappo_atoc_sync.yaml"    # MAPPO + ATOC
  "mappo_atoc_event.yaml"   # MAPPO + ATOC + event driven
)

cfg="${configs[$SLURM_ARRAY_TASK_ID]}"

# Source conda.sh script to enable conda commands
source $(conda info --base)/etc/profile.d/conda.sh
conda activate aced-marl

# Run the experiment
python "${experiment_path}" --config "${config_dir}/${cfg}"
