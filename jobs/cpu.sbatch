#!/bin/bash
# =====================================================================
#  GENERIC job without GPU (data preparation). Usage:
#     sbatch -J job_name jobs/cpu.sbatch  script.py  --arguments
#  Example:
#     sbatch --mem=96G -J prep jobs/cpu.sbatch src/01_prepare_data.py
# =====================================================================
#SBATCH --job-name=splare_lce_cpu
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lce
cd "$SLURM_SUBMIT_DIR"
set -e

echo "Node: $(hostname) | Job: $SLURM_JOB_ID | Start: $(date)"
echo "WORK_DIR=${WORK_DIR:-(src/config.py default)}  SPLARE_DIR=${SPLARE_DIR:-(src/config.py default)}"

python -u src/"$1" "${@:2}"

echo "End: $(date)"
