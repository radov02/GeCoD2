#!/bin/bash
#SBATCH --job-name=sam3-bbox-annot-H100
#SBATCH --output=/d/hpc/home/er52565/GECO2/logs/sam3_bbox_annot_H100_%j.out
#SBATCH --error=/d/hpc/home/er52565/GECO2/logs/sam3_bbox_annot_H100_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# optional positional args: start, limit (e.g. "1000 500" does [1000, 1500))
START_ARG="${1:-}"
LIMIT_ARG="${2:-}"

if [[ -n "$START_ARG" && -n "$LIMIT_ARG" ]]; then
    RANGE_TAG="[${START_ARG},$((START_ARG + LIMIT_ARG))]"
elif [[ -n "$START_ARG" ]]; then
    RANGE_TAG="[${START_ARG},end]"
else
    RANGE_TAG="[0,end]"
fi
# SBATCH lines can't use vars, set the ranged job name here
scontrol update JobId="$SLURM_JOB_ID" JobName="sam3-bbox-annot-H100${RANGE_TAG}" 2>/dev/null || true

# paths
PROJECT=/d/hpc/home/er52565/GECO2
DATASET_SCRIPTS=$PROJECT/IOCfish5kDataset

# 24 fits the 80 GB H100, drop to 16/8 on OOM
BATCH_SIZE=24

mkdir -p "$PROJECT/logs"

module load Anaconda3/2023.07-2
module load CUDA/12.1.1
source /d/hpc/home/er52565/.bashrc
module load Anaconda3 && eval "$(conda shell.bash hook)" && conda activate cnt2

cd "$DATASET_SCRIPTS"

# env info at the top of the .out log
echo "[setup] host=$(hostname) date=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

RANGE_FLAGS=(--start "${START_ARG:-0}")
if [[ -n "$LIMIT_ARG" ]]; then
    RANGE_FLAGS+=(--limit "$LIMIT_ARG")
fi

srun --unbuffered python SAM3_hpc_annotation.py \
    "${RANGE_FLAGS[@]}" \
    --rgbdanddepth \
    --batch_size "$BATCH_SIZE" \
    --overwrite

# Outputs:
#   bbox XMLs : $DATASET_SCRIPTS/dataset/IOCfish5k/bbox_annotations/*.xml
#   segmaps   : $DATASET_SCRIPTS/dataset/IOCfish5k/segment_maps/*_segmap.png
