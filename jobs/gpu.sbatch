#!/bin/bash
# =====================================================================
#  GENERIC GPU job. Usage (from the repository directory):
#     sbatch -J job_name jobs/gpu.sbatch  script.py  --script-arguments
#  Example:
#     sbatch -J base jobs/gpu.sbatch 04_baseline.py
#  Output in logs/job_name_<id>.out (stdout) and logs/job_name_<id>.err (errors).
#  The logs directory must exist:  mkdir -p logs
# =====================================================================
#SBATCH --job-name=splare_lce
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lce
cd "$SLURM_SUBMIT_DIR"
set -e

echo "Node: $(hostname) | Job: $SLURM_JOB_ID | Start: $(date)"
echo "WORK_DIR=${WORK_DIR:-(src/config.py default)}  SPLARE_DIR=${SPLARE_DIR:-(src/config.py default)}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

python -u src/"$1" "${@:2}"

echo "End: $(date)"
