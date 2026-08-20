#!/bin/bash
#SBATCH --job-name=geco2-cocovis
#SBATCH --output=/d/hpc/home/er52565/GECO2/logs/cocovis_%j.out
#SBATCH --error=/d/hpc/home/er52565/GECO2/logs/cocovis_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# cocoeval txt + matcher visualizations for existing FSCD147 _coco_preds.json dumps
#SBATCH --time=06:00:00

# VIS_EVERY=N draws every Nth image, VIS_MAX caps PNGs per dump (0 = no cap)
EXPERIMENTS="${EXPERIMENTS:-geco2 conv_depth_add conv_hiera depth_dim sep_hiera ffm}"
REGIME="${REGIME:-all}"
SPLITS="${SPLITS:-test val}"
INFERS="${INFERS:-whole tiled}"
VIS_IOU_THR="${VIS_IOU_THR:-0.5}"
VIS_EVERY="${VIS_EVERY:-1}"
VIS_MAX="${VIS_MAX:-0}"

PROJECT=/d/hpc/home/er52565/GECO2
DATA="${DATA:-/d/hpc/projects/FRI/pelhanj/fsc147}"
MODELS="${MODELS:-$PROJECT/results_FSCD147}"

# copy the SLURM .out/.err next to the outputs on exit
LOGS_DIR=/d/hpc/home/er52565/GECO2/logs
_run_logs_copied=0
copy_run_logs() {
    [[ "$_run_logs_copied" == "1" ]] && return 0
    _run_logs_copied=1
    local jid="${SLURM_JOB_ID:-}"
    [[ -z "$jid" ]] && return 0
    mkdir -p "$MODELS/${jid}_cocovis"
    local f
    for f in "$LOGS_DIR"/*"${jid}".out "$LOGS_DIR"/*"${jid}".err; do
        [[ -e "$f" ]] && cp -f "$f" "$MODELS/${jid}_cocovis/"
    done
    echo "[logs] copied this run's .out/.err (job ${jid}) into $MODELS/${jid}_cocovis/"
}
trap 'copy_run_logs' EXIT
trap 'copy_run_logs; exit 143' TERM
trap 'copy_run_logs; exit 130' INT

mkdir -p "$PROJECT/logs"

module load Anaconda3/2023.07-2
module load CUDA/12.1.1
source /d/hpc/home/er52565/.bashrc
module load Anaconda3 && eval "$(conda shell.bash hook)" && conda activate cnt2

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PYTHONPATH"

echo "[setup] host=$(hostname) date=$(date -Iseconds)"
echo "[setup] data=$DATA models=$MODELS"
echo "[setup] experiments=[$EXPERIMENTS] regime=$REGIME splits=[$SPLITS] infers=[$INFERS]"
echo "[setup] vis_iou_thr=$VIS_IOU_THR vis_every=$VIS_EVERY vis_max=$VIS_MAX"

VIS_FLAGS=(--vis_matches --vis_iou_thr "$VIS_IOU_THR" --vis_every "$VIS_EVERY")
[[ "$VIS_MAX" != "0" ]] && VIS_FLAGS+=(--vis_max_images "$VIS_MAX")

# rebuild MODEL_NAME the same way the combined FSCD shell does, same env vars override
CROP_P="${CROP_P:-0}"
USE_DEPTHMAPS="${USE_DEPTHMAPS:-0}"
USE_AVAILABLE_DEPTHMAPS="${USE_AVAILABLE_DEPTHMAPS:-0}"
# name reconstruction only, see the train shells
CACHE_DEPTHFEATS="${CACHE_DEPTHFEATS:-0}"
DEPTH_PCA_CHANNELS="${DEPTH_PCA_CHANNELS:-16}"
if [[ "$DEPTH_PCA_CHANNELS" == "-1" ]]; then _DFSUF="raw"; else _DFSUF="$DEPTH_PCA_CHANNELS"; fi
USE_AVAILABLE_DEPTHFEATS="${USE_AVAILABLE_DEPTHFEATS:-0}"
_DFI_ENV="${DEPTH_FUSE_IDENTITY_INIT:-}" # raw env, sep_hiera defaults idinit on in the loop below
DEPTH_FEAT_NORM="${DEPTH_FEAT_NORM:-group}"
DEPTH_TARGET_SIZE="${DEPTH_TARGET_SIZE:-256}"
# USE_DEPTHFEATS=1 = path_1 decoder features, CACHE_DEPTHFEATS=1 = on-disk cache
# (forces the 1-ch map cache on, names carry _pdm+_pdf)
USE_DEPTHFEATS="${USE_DEPTHFEATS:-1}"
if [[ "$USE_DEPTHFEATS" == "1" ]]; then DEPTH_SOURCE="decoder"; else DEPTH_SOURCE="scalar"; CACHE_DEPTHFEATS=0; fi
[[ "$CACHE_DEPTHFEATS" == "1" && "$USE_DEPTHMAPS" != "1" ]] && USE_DEPTHMAPS=1
DEPTH_ADAPT="${DEPTH_ADAPT:-conv}"
# adapter levers (modes 1/2/5), defaults on. DEPTH_CUES=fixed is needed to load
# ckpts trained with fixed Sobel/Laplacian cues, DEPTH_ADAPT_INIT=orthogonal tags _oinit
DEPTH_CUES="${DEPTH_CUES:-learned}"
DEPTH_ADAPT_INIT="${DEPTH_ADAPT_INIT:-orthogonal}"
# masked conv in the depth adapter, set 0 for plain-conv checkpoints (tag _plainconv)
DEPTH_ADAPT_MASKED_CONV="${DEPTH_ADAPT_MASKED_CONV:-1}"
# sep_hiera second-Hiera input: cues (default) or replicate (grayscale)
SEP_HIERA_INPUT="${SEP_HIERA_INPUT:-cues}"
SEP_HIERA_FULLRES="${SEP_HIERA_FULLRES:-1}" # 1 = cues/adapter/GroupNorm at image_size (tag _sfr), train+infer must match
SEP_HIERA_PER_LEVEL_GATE="${SEP_HIERA_PER_LEVEL_GATE:-1}" # 1 = per-level gamma gates (tag _plg, needs _idinit), train+infer must match
# ffm norm: group (default) or batch (tag _ffmbn)
FFM_NORM="${FFM_NORM:-group}"
DINO_INPUT_SIZE="${DINO_INPUT_SIZE:-0}"
DEPTH_HIRES_FUSION="${DEPTH_HIRES_FUSION:-0}"
DEPTH_HIRES_NORM="${DEPTH_HIRES_NORM:-1}" # 1 = masked GroupNorm on the hires depth input (_hires becomes _hiresn), 0 = plain _hires
DEPTH_CHECKPOINT="${DEPTH_CHECKPOINT:-depth_anything_v2_vitl.pth}"
USE_DENSITY="${USE_DENSITY:-0}"
DENSITY_LOSS_TYPE="${DENSITY_LOSS_TYPE:-dmcount}"
DENSITY_HEAD_TYPE="${DENSITY_HEAD_TYPE:-simple}"
DENSITY_GUIDED="${DENSITY_GUIDED:-0}"
DENSITY_ADAPTIVE_SIGMA="${DENSITY_ADAPTIVE_SIGMA:-0}"
DENSITY_ABS_COUNT_WEIGHT="${DENSITY_ABS_COUNT_WEIGHT:-0.5}"
PROBE_EPOCHS="${PROBE_EPOCHS:-0}"
PROBE_ADAPTIVE="${PROBE_ADAPTIVE:-0}"
AUX_LR_WARMUP_EPOCHS="${AUX_LR_WARMUP_EPOCHS:-0}"
SEED="${SEED:-42}"
# captured once so the loop doesn't carry a mode-3/4 value into a later experiment
_DEPTH_FEAT_CHANNELS_ENV="${DEPTH_FEAT_CHANNELS:-}"
_DEPTH_KERNEL_SIZE_ENV="${DEPTH_KERNEL_SIZE:-}"
# tags that don't depend on the experiment
CROP_TAG=""
if awk "BEGIN{exit !($CROP_P>0)}"; then
    CROP_TAG="_crop$(awk "BEGIN{printf \"%d\", ($CROP_P*100)+0.5}")"
fi
DENS_TAG=""
if [[ "$USE_DENSITY" != "0" ]]; then
    if [[ "$DENSITY_LOSS_TYPE" == "mse" ]]; then DENS_TAG="_densmse"; else DENS_TAG="_dens"; fi
    [[ "$DENSITY_HEAD_TYPE" == "fpn" ]] && DENS_TAG="${DENS_TAG}f"
    [[ "$DENSITY_ADAPTIVE_SIGMA" != "0" ]] && DENS_TAG="${DENS_TAG}s"
    [[ "$DENSITY_GUIDED" != "0" ]] && DENS_TAG="${DENS_TAG}g"
    if [[ "$DENSITY_ABS_COUNT_WEIGHT" != "0.5" ]]; then
        DENS_TAG="${DENS_TAG}a$(awk "BEGIN{printf \"%d\", ($DENSITY_ABS_COUNT_WEIGHT*100)+0.5}")"
    fi
fi
AW_TAG=""; [[ "$AUX_LR_WARMUP_EPOCHS" != "0" ]] && AW_TAG="_aw${AUX_LR_WARMUP_EPOCHS}"
SEED_TAG=""; [[ "$SEED" != "42" ]] && SEED_TAG="_s${SEED}"

for E in $EXPERIMENTS; do
    # per-experiment depth config, mirrors the combined shell
    case "$E" in
        conv_depth_add) USE_DEPTH=1; DEPTH_FEAT_CHANNELS="${_DEPTH_FEAT_CHANNELS_ENV:-16}"; DEPTH_KERNEL_SIZE="${_DEPTH_KERNEL_SIZE_ENV:-1}" ;;
        conv_hiera)     USE_DEPTH=2; DEPTH_FEAT_CHANNELS="${_DEPTH_FEAT_CHANNELS_ENV:-16}"; DEPTH_KERNEL_SIZE="${_DEPTH_KERNEL_SIZE_ENV:-1}" ;;
        depth_dim)      USE_DEPTH=3; DEPTH_FEAT_CHANNELS=1; DEPTH_KERNEL_SIZE="${_DEPTH_KERNEL_SIZE_ENV:-1}" ;;
        sep_hiera)      USE_DEPTH=4; DEPTH_FEAT_CHANNELS=3; DEPTH_KERNEL_SIZE="" ;;
        ffm)            USE_DEPTH=5; DEPTH_FEAT_CHANNELS="${_DEPTH_FEAT_CHANNELS_ENV:-16}"; DEPTH_KERNEL_SIZE="" ;; # ffm keeps the parser's 3x3 default, no _k tag
        *)              USE_DEPTH="";  DEPTH_FEAT_CHANNELS=""; DEPTH_KERNEL_SIZE="" ;; # geco2 baseline: no depth
    esac
    # train shells tag the RGB baseline "geco2_finetuned"
    E_TAG="$E"; [[ "$E" == "geco2" ]] && E_TAG="geco2_finetuned"
    # sep_hiera defaults idinit on (tags _idinit + _plg), env DEPTH_FUSE_IDENTITY_INIT wins
    if [[ "$USE_DEPTH" == "4" ]]; then DEPTH_FUSE_IDENTITY_INIT="${_DFI_ENV:-1}"; else DEPTH_FUSE_IDENTITY_INIT="${_DFI_ENV:-0}"; fi
    DMODEL_TAG=""
    if [[ -n "$USE_DEPTH" ]]; then
        case "$DEPTH_CHECKPOINT" in
            depth_anything_v2_vitl.pth) DMODEL_TAG="_vitl" ;;
            depth_anything_v2_vitb.pth) DMODEL_TAG="_vitb" ;;
            depth_anything_v2_vits.pth) DMODEL_TAG="_vits" ;;
            dav2_sdt_vitb.pth)          DMODEL_TAG="_sdtvitb" ;;
            da3_sdt_vitl.pth)           DMODEL_TAG="_sdtvitl" ;;
            *) DMODEL_TAG="_d$(echo "$DEPTH_CHECKPOINT" | sed 's/\.pth$//; s/[^A-Za-z0-9]//g')" ;;
        esac
    fi
    KERNEL_TAG=""; IDINIT_TAG=""; DNORM_TAG=""; DR_TAG=""; PATH1_TAG=""; DCONV_TAG=""; DIN_TAG=""; HIRES_TAG=""; DFC_TAG=""; LCUES_TAG=""; OINIT_TAG=""; MCONV_TAG=""; FFMBN_TAG=""; PDM_TAG=""; PDF_TAG=""; PROBE_TAG=""; SEPCUES_TAG=""; SFR_TAG=""; PLG_TAG=""
    if [[ -n "$USE_DEPTH" ]]; then
        if [[ "$USE_DEPTH" == "3" ]]; then
            # mode-3 initial_depth_fuse is always 1x1
            [[ "$DEPTH_KERNEL_SIZE" == "3" ]] && echo "[warn] depth_dim (mode 3) fusion is 1x1; ignoring DEPTH_KERNEL_SIZE=3 -> tagging _k1"
            KERNEL_TAG="_k1"
        elif [[ -n "$DEPTH_KERNEL_SIZE" ]]; then
            KERNEL_TAG="_k${DEPTH_KERNEL_SIZE}" # modes 1/2 only
        fi
        [[ "$DEPTH_FUSE_IDENTITY_INIT" != "0" ]] && IDINIT_TAG="_idinit"
        [[ "$DEPTH_FEAT_NORM" != "none" ]] && DNORM_TAG="_dnorm"
        [[ "$USE_DEPTH" == "4" && "$SEP_HIERA_INPUT" == "cues" ]] && SEPCUES_TAG="_sepcues"
        [[ "$USE_DEPTH" == "4" && "$SEP_HIERA_FULLRES" != "0" ]] && SFR_TAG="_sfr"
        [[ "$USE_DEPTH" == "4" && "$SEP_HIERA_PER_LEVEL_GATE" != "0" && "$DEPTH_FUSE_IDENTITY_INIT" != "0" ]] && PLG_TAG="_plg"
        [[ "$DEPTH_TARGET_SIZE" != "0" ]] && DR_TAG="_dr${DEPTH_TARGET_SIZE}"
        [[ "$DINO_INPUT_SIZE" != "0" ]] && DIN_TAG="_dinoInSize${DINO_INPUT_SIZE}"
        if [[ "$USE_DEPTH" == "1" || "$USE_DEPTH" == "2" || "$USE_DEPTH" == "5" ]]; then # modes 1/2/5
            [[ "$DEPTH_SOURCE" == "decoder" ]] && PATH1_TAG="_path1"
            [[ "$DEPTH_ADAPT" == "conv" ]] && DCONV_TAG="_dconv"
            [[ "$DEPTH_CUES" == "learned" ]] && LCUES_TAG="_lcues"
            [[ "$DEPTH_ADAPT_INIT" == "orthogonal" ]] && OINIT_TAG="_oinit"
            [[ "$DEPTH_ADAPT_MASKED_CONV" == "0" ]] && MCONV_TAG="_plainconv"
            [[ "$USE_DEPTH" == "5" && "$FFM_NORM" == "batch" ]] && FFMBN_TAG="_ffmbn"
            [[ "$DEPTH_FEAT_CHANNELS" != "256" ]] && DFC_TAG="_dfc${DEPTH_FEAT_CHANNELS}"
            [[ "$DEPTH_HIRES_FUSION" != "0" ]] && HIRES_TAG="_hires"
            [[ "$DEPTH_HIRES_FUSION" != "0" && "$DEPTH_HIRES_NORM" != "0" ]] && HIRES_TAG="_hiresn"
        fi
        # precomputed depth maps blank the depth-model/source/input tags
        if [[ "$USE_DEPTHMAPS" == "1" ]]; then
            if [[ "$USE_AVAILABLE_DEPTHMAPS" == "1" ]]; then PDM_TAG="_pdm"; else PDM_TAG="_pdmgen"; fi
            if [[ "$CACHE_DEPTHFEATS" == "1" ]]; then
                if [[ "$USE_AVAILABLE_DEPTHFEATS" == "1" ]]; then PDF_TAG="_pdf${_DFSUF}"; else PDF_TAG="_pdf${_DFSUF}gen"; fi
            fi
            DMODEL_TAG=""; PATH1_TAG=""
            # mode 3 fuses a cached map at full image_size, tag _drfull not _dr<N>
            if [[ "$USE_DEPTH" == "3" ]]; then DR_TAG="_drfull"; else DR_TAG=""; fi
            [[ "$CACHE_DEPTHFEATS" != "1" ]] && DIN_TAG="" # _pdf cache is keyed by dinoInSize, keep its tag
        fi
        if [[ "$PROBE_ADAPTIVE" == "1" ]]; then PROBE_TAG="_probeA"; elif [[ "$PROBE_EPOCHS" != "0" ]]; then PROBE_TAG="_probe${PROBE_EPOCHS}"; fi
    fi
    MODEL_NAME="GECO2_FSCD147_${E_TAG}${CROP_TAG}${PDM_TAG}${PDF_TAG}${DMODEL_TAG}${DIN_TAG}${DR_TAG}${PATH1_TAG}${DCONV_TAG}${LCUES_TAG}${OINIT_TAG}${MCONV_TAG}${DFC_TAG}${DNORM_TAG}${SEPCUES_TAG}${SFR_TAG}${KERNEL_TAG}${FFMBN_TAG}${IDINIT_TAG}${PLG_TAG}${HIRES_TAG}${DENS_TAG}${PROBE_TAG}${AW_TAG}${SEED_TAG}_${REGIME}"
    for SPLIT in $SPLITS; do
        for INFER in $INFERS; do
            SPLIT_UC=$(echo "$SPLIT" | tr '[:lower:]' '[:upper:]')
            SUF="_${SPLIT_UC}_${INFER}" # IOC-style layout
            SUF_LEGACY="_TEST_${SPLIT}_${INFER}" # legacy FSCD layout
            # dumps live flat in $MODELS or in per-job subfolders, outputs land next to each dump
            DUMPS=()
            for DUMP in "${MODELS}/${MODEL_NAME}${SUF}_coco_preds.json" "${MODELS}"/*/"${MODEL_NAME}${SUF}_coco_preds.json" \
                        "${MODELS}/${MODEL_NAME}${SUF_LEGACY}_coco_preds.json" "${MODELS}"/*/"${MODEL_NAME}${SUF_LEGACY}_coco_preds.json"; do
                [[ -f "$DUMP" ]] && DUMPS+=("$DUMP")
            done
            if [[ ${#DUMPS[@]} -eq 0 ]]; then
                echo "[skip] no dump: ${MODELS}[/<jobid>_<exp>]/${MODEL_NAME}{${SUF}|${SUF_LEGACY}}_coco_preds.json"
                continue
            fi
            for DUMP in "${DUMPS[@]}"; do
                OUT_DIR="$(dirname "$DUMP")"
                echo "==================== ${DUMP#"${MODELS}/"} ===================="
                # tee into the per-dump _cocoeval.txt
                srun --unbuffered --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
                    python eval_bboxes.py --data_path "$DATA" \
                        --pred_json "$DUMP" \
                        --split "$SPLIT" \
                        "${VIS_FLAGS[@]}" \
                        --vis_dir "${OUT_DIR}/${MODEL_NAME}${SUF}_cocovis" \
                    2>&1 | tee "${OUT_DIR}/${MODEL_NAME}${SUF}_cocoeval.txt"
            done
        done
    done
done

echo "[done] cocovis dirs + cocoeval txt land NEXT TO each processed _coco_preds.json dump:"
echo "[done]   ${MODELS}[/<jobid>_<exp>]/GECO2_FSCD147_<exp>_${REGIME}_TEST_<split>_<infer>_cocovis/  and  ..._cocoeval.txt"
