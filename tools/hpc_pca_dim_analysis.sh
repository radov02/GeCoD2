#!/bin/bash
#SBATCH --job-name=pca_dim_analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/d/hpc/home/er52565/GECO2/logs/pca_dim_analysis_%j.log

# PCA variance check of the depth features on all three datasets, see tools/pca_dim_analysis.py
# sbatch tools/hpc_pca_dim_analysis.sh [extra pca_dim_analysis.py args]

set -euo pipefail
PROJECT=/d/hpc/home/er52565/GECO2
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
# gpu nodes have no internet, prewarm the DAv2-Large ckpt into the HF cache from the login node
export HF_HUB_OFFLINE=1
mkdir -p "$PROJECT/logs"

module load Anaconda3 && eval "$(conda shell.bash hook)" && conda activate cnt2

python tools/pca_dim_analysis.py "$@"
echo "[done] outputs in $PROJECT/pca_dim_analysis_out/"
