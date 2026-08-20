#!/bin/bash
#SBATCH --job-name=GECO2-trinf-H100
#SBATCH --output=/d/hpc/home/er52565/GECO2/logs/traininf_H100_%j.out
#SBATCH --error=/d/hpc/home/er52565/GECO2/logs/traininf_H100_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# override walltime at submit with sbatch --time=HH:MM:SS
#SBATCH --time=56:00:00
# SIGTERM 120 s early so the train script can write its summary
#SBATCH --signal=TERM@120

# train + inference driver for the FSCD147 depth-fusion runs. stage 1 finetunes
# the chosen variant from the GeCo2 prior, stage 2 runs whole/tiled inference,
# stage 3 scores the COCO dump with eval_bboxes.py (official AP, 0-100; the AP
# in *_results.txt is the 0-1 greedy one). geco2_pretrained = inference-only.
#
# args: [both|train|inference] <experiment> [all|lt1500] [epochs] [train_scale]
#   [infer_type] [test_split] [vis_every] [sweep] [probe|noprobe] [-- extra train args...]
# example:
#   sbatch hpc_H100_train_and_inference_FSCD147_dataset.sh both conv_depth_add all 200 whole both test

# optional config (training/config_FSCD147.sh): source it before sbatch, or pass
# CONFIG_FILE=/abs/path to source inside the job. a 'VAR=x CONFIG_FILE=... sbatch'
# prefix loses VAR for every var the config exports. positionals always come from
# the sbatch line.
if [[ -n "${CONFIG_FILE:-}" ]]; then
    if [[ -f "$CONFIG_FILE" ]]; then
        echo "[config] sourcing $CONFIG_FILE"
        source "$CONFIG_FILE"
    else
        echo "[config][warn] CONFIG_FILE set but not found: $CONFIG_FILE" >&2
    fi
fi

# knobs go in as explicit CLI flags, keep utils/run_config.py from re-applying them
export GECO2_RUN_CONFIG=skip

# arg 1: both (default) | train | inference ('infer' ok). inference loads the
# already-trained checkpoint named by these same knobs.
RUN_STAGES="${1:-both}"
case "$RUN_STAGES" in
    both|train|inference) : ;;
    infer)                RUN_STAGES=inference ;;
    *)
        echo "[error] Unknown run-stage: '$RUN_STAGES'. Accepted: both | train | inference"
        exit 1
        ;;
esac

# arg 2: experiment id or tag, mapped to a train script in the case block below
EXPERIMENT="${2:-1}"

# arg 3: all = full FSC147 train split, lt1500 = only train images with <1500
# objects (baked into MODEL_NAME)
REGIME="${3:-all}"

# arg 4: max epochs, baked into the checkpoint name and forwarded to stage 2 as
# --ckpt_epochs
EPOCHS="${4:-200}"

# arg 5: whole (CROP_P=0) | tiled (CROP_P=0.5, zoom-in crops, tag _crop50, crop
# range 192-320 around the 256px tile)
SCALE="${5:-whole}"
case "$SCALE" in
    whole)        SCALE_CROP_DEFAULT=0   ;;
    tiled|tiling) SCALE_CROP_DEFAULT=0.5 ; SCALE=tiled ;;
    *)
        echo "[error] Unknown train scale: '$SCALE'. Accepted: whole | tiled"
        exit 1
        ;;
esac

# arg 6: whole (default) = single forward at image_size | tiled = overlapping
# crops (better recall on tiny objects) | both | none = train only
INFER_TYPE="${6:-whole}"
case "$INFER_TYPE" in
    whole) INFER_TYPES=(whole) ;;
    tiled) INFER_TYPES=(tiled) ;;
    both)  INFER_TYPES=(whole tiled) ;;
    none)  INFER_TYPES=() ;;
    *)
        echo "[error] Unknown inference type: '$INFER_TYPE'. Accepted: whole | tiled | both | none"
        exit 1
        ;;
esac

# arg 7: test (1190 images) | test_coco | val (1286) | val_coco. _coco = the
# COCO-overlapping category subsets. results/visuals get a _<split> suffix.
TEST_SPLIT="${7:-test}"
case "$TEST_SPLIT" in
    test|test_coco|val|val_coco) : ;;
    *)
        echo "[error] Unknown test split: '$TEST_SPLIT'. Accepted: test | test_coco | val | val_coco"
        exit 1
        ;;
esac

# arg 8: visuals every N images (0 = off)
VISUALS_EVERY="${8:-0}"
if ! [[ "$VISUALS_EVERY" =~ ^[0-9]+$ ]]; then
    echo "[error] vis_every must be a non-negative integer, got: '$VISUALS_EVERY'"
    exit 1
fi

# arg 9 (tiled only): 0 = normal run, 1 = val grid sweep of SCORE_ABS_THR x
# NMS_IOU x EDGE_MARGIN, 2 = sweep then test with the winning thresholds.
# grids via SWEEP_ABS/SWEEP_NMS/SWEEP_EDGE (comma lists).
SWEEP="${9:-0}"
if [[ ! "$SWEEP" =~ ^[0-2]$ ]]; then
    echo "[error] sweep positional (=$SWEEP) must be 0, 1 or 2. (A probe arg goes one slot later: ... vis sweep probe|noprobe.)"
    exit 1
fi

# arg 10: noprobe forces PROBE_ADAPTIVE=0 PROBE_EPOCHS=0 (for geco2_finetuned,
# nothing to probe); probe or '' keeps the PROBE_* knobs. inference-only runs must
# repeat the probe setting the checkpoint was trained with (it's in the run name).
PROBE_ARG=""
case "${10:-}" in
    probe|noprobe) PROBE_ARG="${10}"; _NPOS=10 ;;
    "")            _NPOS=10 ;;  # not passed (shift caps at $#) or an explicit "" placeholder
    -*)            _NPOS=9 ;; # extra train args start here (covers the -- separator)
    *)
        echo "[error] Unknown 10th positional: '${10}'. Accepted: probe | noprobe (or -- extra train args)."
        exit 1
        ;;
esac

# remaining args are forwarded verbatim to the train script (e.g. -- --some_flag val)
shift $(( $# > _NPOS ? _NPOS : $# ))
# drop a leading -- separator so "... sweep -- --foo" works
[[ "${1:-}" == "--" ]] && shift
EXTRA_ARGS=("$@")
# an sbatch option placed after the script path lands here instead of reaching
# slurm (job would run with the default walltime while the flag corrupts a train arg)
for _a in "${EXTRA_ARGS[@]}"; do
    if [[ "$_a" =~ ^--(time|mem|partition|gres|constraint|cpus-per-task|nodes|ntasks|qos|account)(=|$) ]]; then
        echo "[error] '$_a' is an sbatch option passed as a script arg."
        echo "        sbatch options go before the script path:"
        echo "        sbatch $_a ... training/<this script>.sh <positionals>"
        exit 1
    fi
done

if [[ "$PROBE_ARG" == "noprobe" ]]; then
    PROBE_ADAPTIVE=0
    PROBE_EPOCHS=0
    echo "[setup] probe disabled via positional arg 10 (noprobe): PROBE_ADAPTIVE=0 PROBE_EPOCHS=0"
fi

case "$REGIME" in
    all)    REGIME_TAG="all";    MAX_OBJECTS_FLAGS=() ;;
    lt1500) REGIME_TAG="lt1500"; MAX_OBJECTS_FLAGS=(--max_objects 1500) ;;
    *)
        echo "[error] Unknown regime: '$REGIME'. Accepted: all | lt1500"
        exit 1
        ;;
esac

PROJECT=/d/hpc/home/er52565/GECO2

# per-experiment flags (shared by both stages); defaults match the
# parser.set_defaults() block in each train script
case "$EXPERIMENT" in
    1|conv_depth_add)
        EXP_TAG="conv_depth_add"
        TRAIN_SCRIPT="experiments_fscd/conv_on_depth_feats_addition_with_Hiera_feats/train_conv_on_depth_feats_addition_with_Hiera_feats_on_FSCD147_dataset.py"
        USE_DEPTH=1
        DEPTH_FEAT_CHANNELS="${DEPTH_FEAT_CHANNELS:-16}" # default 16 (PCA-16 ceiling), non-256 tags _dfc<N>
        DEPTH_KERNEL_SIZE="${DEPTH_KERNEL_SIZE:-1}" # 1x1 default, 3 for the k3 ablation (_k3)
        ;;
    2|conv_hiera)
        EXP_TAG="conv_hiera"
        TRAIN_SCRIPT="experiments_fscd/conv_with_Hiera_feats/train_conv_with_Hiera_feats_on_FSCD147_dataset.py"
        USE_DEPTH=2
        DEPTH_FEAT_CHANNELS="${DEPTH_FEAT_CHANNELS:-16}" # default 16 (PCA-16 ceiling), non-256 tags _dfc<N>
        DEPTH_KERNEL_SIZE="${DEPTH_KERNEL_SIZE:-1}" # 1x1 default, 3 for the k3 ablation (_k3)
        ;;
    3|depth_dim)
        EXP_TAG="depth_dim"
        TRAIN_SCRIPT="experiments_fscd/depth_dim_added_and_conv/train_depth_dim_added_and_conv_on_FSCD147_dataset.py"
        USE_DEPTH=3
        DEPTH_FEAT_CHANNELS=1
        DEPTH_KERNEL_SIZE="${DEPTH_KERNEL_SIZE:-1}" # 1x1 default, 3 for the k3 ablation (_k3)
        ;;
    4|sep_hiera)
        EXP_TAG="sep_hiera"
        TRAIN_SCRIPT="experiments_fscd/separate_Hiera_on_RGB_and_depth/train_separate_Hiera_on_RGB_and_depth_on_FSCD147_dataset.py"
        USE_DEPTH=4
        DEPTH_FEAT_CHANNELS=3
        DEPTH_KERNEL_SIZE=""
        ;;
    5|geco2|geco2_finetuned)
        EXP_TAG="geco2_finetuned"
        TRAIN_SCRIPT="experiments_fscd/geco2/train_geco2_on_FSCD147_dataset.py"
        USE_DEPTH=""
        DEPTH_FEAT_CHANNELS=""
        DEPTH_KERNEL_SIZE=""
        ;;
    geco2_pretrained)
        # inference-only: eval PRETRAINED_INIT as-is (symlinked under this run's
        # MODEL_NAME). also triggers stage 0, the upstream inference.py +
        # eval_bboxes.py rerun (UPSTREAM_EVAL=0 skips it).
        EXP_TAG="geco2_pretrained"
        TRAIN_SCRIPT=""
        USE_DEPTH=""
        DEPTH_FEAT_CHANNELS=""
        DEPTH_KERNEL_SIZE=""
        PRETRAINED_EVAL=1
        ;;
    6|ffm)
        EXP_TAG="ffm"
        TRAIN_SCRIPT="experiments_fscd/ffm/train_ffm_on_FSCD147_dataset.py"
        USE_DEPTH=5
        DEPTH_FEAT_CHANNELS="${DEPTH_FEAT_CHANNELS:-16}" # default 16 (matches modes 1/2), non-256 tags _dfc<N>
        DEPTH_KERNEL_SIZE="${DEPTH_KERNEL_SIZE:-3}" # ffm always passes + tags its kernel, default paper 3x3 (_k3)
        ;;
    *)
        echo "[error] Unknown experiment selector: '$EXPERIMENT'."
        echo "Accepted values: 1|conv_depth_add, 2|conv_hiera, 3|depth_dim, 4|sep_hiera, 5|geco2_finetuned, 6|ffm, geco2_pretrained (inference-only)"
        exit 1
        ;;
esac

if [[ "${PRETRAINED_EVAL:-0}" == "1" && "$RUN_STAGES" != "inference" ]]; then
    echo "[error] experiment geco2_pretrained is inference-only (it evaluates PRETRAINED_INIT as-is)."
    echo "        Use run-stage 'inference', or experiment 'geco2_finetuned' to train the baseline."
    exit 1
fi

# zoom-in crop aug: with prob CROP_P training takes a random square crop in
# [CROP_MIN_PX,CROP_MAX_PX] original px upscaled to IMAGE_SIZE; CROP_P>0 tags
# the run name _crop<pct>
CROP_P="${CROP_P:-$SCALE_CROP_DEFAULT}"
CROP_MIN_PX="${CROP_MIN_PX:-192}"
CROP_MAX_PX="${CROP_MAX_PX:-320}"
CROP_TAG=""
if awk "BEGIN{exit !($CROP_P>0)}"; then
    CROP_TAG="_crop$(awk "BEGIN{printf \"%d\", ($CROP_P*100)+0.5}")"
fi

# density head: USE_DENSITY=1 adds the DensityHead and counts by integrating its
# map (DM-Count OT loss by default), detection branch stays for AP. the tag picks
# the matching _dens[f][s][g][a##] checkpoint.
USE_DENSITY="${USE_DENSITY:-0}"
DENSITY_WEIGHT="${DENSITY_WEIGHT:-1.0}"
DENSITY_LOSS_TYPE="${DENSITY_LOSS_TYPE:-dmcount}"
DENSITY_ABS_COUNT_WEIGHT="${DENSITY_ABS_COUNT_WEIGHT:-0.5}"
DENSITY_LR="${DENSITY_LR:-1e-4}"
DENSITY_HEAD_TYPE="${DENSITY_HEAD_TYPE:-simple}"
DENSITY_GUIDED="${DENSITY_GUIDED:-0}"
DENSITY_ADAPTIVE_SIGMA="${DENSITY_ADAPTIVE_SIGMA:-0}"
DENSITY_SIGMA="${DENSITY_SIGMA:-8.0}"
DENSITY_SIGMA_K="${DENSITY_SIGMA_K:-3}"
DENSITY_SIGMA_BETA="${DENSITY_SIGMA_BETA:-0.3}"
DENSITY_SIGMA_MIN="${DENSITY_SIGMA_MIN:-2.0}"
DENSITY_SIGMA_MAX="${DENSITY_SIGMA_MAX:-15.0}"
# ckpt/plateau selection metric: RMSE weight in the MAE blend (0 = pure MAE)
SELECT_RMSE_WEIGHT="${SELECT_RMSE_WEIGHT:-0.0}"
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

# depth-fusion knobs, shared by both stages so stage 2 rebuilds what stage 1 saved.
# tags: _k1/_k3 kernel, _idinit identity-init, _dnorm GroupNorm.
# sep_hiera defaults identity-init on (gamma warm start + _plg gate need it).
if [[ "$USE_DEPTH" == "4" ]]; then
    DEPTH_FUSE_IDENTITY_INIT="${DEPTH_FUSE_IDENTITY_INIT:-1}"
else
    DEPTH_FUSE_IDENTITY_INIT="${DEPTH_FUSE_IDENTITY_INIT:-0}"
fi
DEPTH_FEAT_NORM="${DEPTH_FEAT_NORM:-group}"
DEPTH_FEAT_NORM_GROUPS="${DEPTH_FEAT_NORM_GROUPS:-0}" # arch-changing, train + inference must match
DEPTH_TARGET_SIZE="${DEPTH_TARGET_SIZE:-256}" # depth predicted+fused at this res; 256 = finest Hiera FPN grid; 0 = auto, tag _dr<N>
# depth front-end: DEPTH_SOURCE=decoder taps DPT path_1 (_path1), DEPTH_ADAPT=conv
# adds the conv adapter (_dconv), DINO_INPUT_SIZE sets the ViT scale (_dinoInSize<N>).
# USE_DEPTHFEATS=1 (default) = path_1 features, 0 = 1-ch scalar disparity;
# CACHE_DEPTHFEATS reads path_1 from disk; USE_DEPTHMAPS = 1-ch map cache.
USE_DEPTHFEATS="${USE_DEPTHFEATS:-1}"
CACHE_DEPTHFEATS="${CACHE_DEPTHFEATS:-0}"
if [[ ( "$USE_DEPTH" == "1" || "$USE_DEPTH" == "2" || "$USE_DEPTH" == "5" ) && "$USE_DEPTHFEATS" == "1" ]]; then
    DEPTH_SOURCE="decoder"
else
    # modes 3/4 (scalar forced by the model), or the 1-ch scalar-disparity signal:
    DEPTH_SOURCE="scalar"; CACHE_DEPTHFEATS=0
fi
DEPTH_ADAPT="${DEPTH_ADAPT:-conv}"
# adapter levers (modes 1/2/5): DEPTH_CUES=learned (_lcues) vs fixed Sobel/Laplacian
# (needed to load ckpts trained that way); DEPTH_ADAPT_INIT=orthogonal (_oinit).
DEPTH_CUES="${DEPTH_CUES:-learned}"
DEPTH_ADAPT_INIT="${DEPTH_ADAPT_INIT:-orthogonal}"
# masked conv in the adapter: renormalize each window by its valid-pixel fraction
# so the letterbox pad doesn't dilute edges. 0 loads plain-conv ckpts (_plainconv).
DEPTH_ADAPT_MASKED_CONV="${DEPTH_ADAPT_MASKED_CONV:-1}"
# sep_hiera second-Hiera input: cues = [disparity, Sobel, Laplacian] (default),
# replicate = plain grayscale
SEP_HIERA_INPUT="${SEP_HIERA_INPUT:-cues}"
SEP_HIERA_FULLRES="${SEP_HIERA_FULLRES:-1}" # 1 = cues/adapter/GroupNorm at image_size (_sfr), 0 = S-res; train+infer must match
SEP_HIERA_PER_LEVEL_GATE="${SEP_HIERA_PER_LEVEL_GATE:-1}" # 1 = per-level gamma gates (_plg, needs _idinit), 0 = single scalar
# ffm only: group (default) | batch (BiSeNet BatchNorm ablation, _ffmbn)
FFM_NORM="${FFM_NORM:-group}"
# depth+cue visuals every N epochs for one fixed val sample (depth runs only)
VIS_EVERY="${VIS_EVERY:-$VISUALS_EVERY}"
DINO_INPUT_SIZE="${DINO_INPUT_SIZE:-0}"
# frozen depth model, name-tagged below; train + inference must use the same one
DEPTH_CHECKPOINT="${DEPTH_CHECKPOINT:-depth_anything_v2_vitl.pth}"; export DEPTH_CHECKPOINT
DMODEL_TAG=""
if [[ -n "$USE_DEPTH" ]]; then
    case "$DEPTH_CHECKPOINT" in
        depth_anything_v2_vitl.pth) DMODEL_TAG="_vitl" ;; # DAv2 Large (default)
        depth_anything_v2_vitb.pth) DMODEL_TAG="_vitb" ;; # DAv2 Base
        depth_anything_v2_vits.pth) DMODEL_TAG="_vits" ;; # DAv2 Small
        dav2_sdt_vitb.pth)          DMODEL_TAG="_sdtvitb" ;; # AnyDepth SDT (ablation)
        da3_sdt_vitl.pth)           DMODEL_TAG="_sdtvitl" ;; # DA3 SDT (ablation)
        *) DMODEL_TAG="_d$(echo "$DEPTH_CHECKPOINT" | sed 's/\.pth$//; s/[^A-Za-z0-9]//g')" ;;
    esac
    [[ -n "$DMODEL_TAG" ]] && echo "[setup] Depth model: ${DEPTH_CHECKPOINT} (tag ${DMODEL_TAG})"
fi
# input-level hi-res depth injection (modes 1/2), tag _hires; train+infer must match
DEPTH_HIRES_FUSION="${DEPTH_HIRES_FUSION:-0}"
DEPTH_HIRES_NORM="${DEPTH_HIRES_NORM:-1}" # 1 = masked GroupNorm on the hires input, tag becomes _hiresn
KERNEL_TAG=""; IDINIT_TAG=""; DNORM_TAG=""; DR_TAG=""; PATH1_TAG=""; DCONV_TAG=""; DIN_TAG=""; HIRES_TAG=""; DFC_TAG=""; LCUES_TAG=""; OINIT_TAG=""; MCONV_TAG=""; FFMBN_TAG=""; SEPCUES_TAG=""; SFR_TAG=""; PLG_TAG=""
if [[ -n "$USE_DEPTH" ]]; then
    if [[ "$USE_DEPTH" == "3" ]]; then
        # mode-3 fusion is 1x1, a k3 label would be wrong
        [[ "$DEPTH_KERNEL_SIZE" == "3" ]] && echo "[warn] depth_dim (mode 3) fusion is 1x1; ignoring DEPTH_KERNEL_SIZE=3 -> tagging _k1"
        KERNEL_TAG="_k1"
    elif [[ -n "$DEPTH_KERNEL_SIZE" ]]; then
        KERNEL_TAG="_k${DEPTH_KERNEL_SIZE}" # modes 1/2 always, ffm only when explicitly set
    fi
    [[ "$DEPTH_FUSE_IDENTITY_INIT" != "0" ]] && IDINIT_TAG="_idinit"
    [[ "$DEPTH_FEAT_NORM" != "none" ]] && DNORM_TAG="_dnorm"
    [[ "$USE_DEPTH" == "4" && "$SEP_HIERA_INPUT" == "cues" ]] && SEPCUES_TAG="_sepcues"
    [[ "$USE_DEPTH" == "4" && "$SEP_HIERA_FULLRES" != "0" ]] && SFR_TAG="_sfr"
    [[ "$USE_DEPTH" == "4" && "$SEP_HIERA_PER_LEVEL_GATE" != "0" && "$DEPTH_FUSE_IDENTITY_INIT" != "0" ]] && PLG_TAG="_plg"
    [[ "$DEPTH_TARGET_SIZE" != "0" ]] && DR_TAG="_dr${DEPTH_TARGET_SIZE}"
    [[ "$DINO_INPUT_SIZE" != "0" ]] && DIN_TAG="_dinoInSize${DINO_INPUT_SIZE}"
    if [[ "$USE_DEPTH" == "1" || "$USE_DEPTH" == "2" || "$USE_DEPTH" == "5" ]]; then   # decoder/conv adapter + hi-res: modes 1/2/5
        [[ "$DEPTH_SOURCE" == "decoder" ]] && PATH1_TAG="_path1"
        [[ "$DEPTH_ADAPT" == "conv" ]] && DCONV_TAG="_dconv"
        [[ "$DEPTH_CUES" == "learned" ]] && LCUES_TAG="_lcues"
        [[ "$DEPTH_ADAPT_INIT" == "orthogonal" ]] && OINIT_TAG="_oinit"
        [[ "$DEPTH_ADAPT_MASKED_CONV" == "0" ]] && MCONV_TAG="_plainconv"
        [[ "$USE_DEPTH" == "5" && "$FFM_NORM" == "batch" ]] && FFMBN_TAG="_ffmbn"
        # tag adapter width when != 256
        [[ "$DEPTH_FEAT_CHANNELS" != "256" ]] && DFC_TAG="_dfc${DEPTH_FEAT_CHANNELS}"
        [[ "$DEPTH_HIRES_FUSION" != "0" ]] && HIRES_TAG="_hires"
        [[ "$DEPTH_HIRES_FUSION" != "0" && "$DEPTH_HIRES_NORM" != "0" ]] && HIRES_TAG="_hiresn"
    fi
    echo "[setup] Depth fusion: kernel=${DEPTH_KERNEL_SIZE:-n/a} identity_init=${DEPTH_FUSE_IDENTITY_INIT} feat_norm=${DEPTH_FEAT_NORM} (groups=${DEPTH_FEAT_NORM_GROUPS}) source=${DEPTH_SOURCE} (path1_depth=$([[ "$DEPTH_SOURCE" == "decoder" ]] && echo on || echo off)) adapt=${DEPTH_ADAPT} cues=${DEPTH_CUES} adapt_init=${DEPTH_ADAPT_INIT} ffm_norm=${FFM_NORM} dinoInSize=${DINO_INPUT_SIZE}"
fi

# probe-then-finetune: PROBE_EPOCHS=N trains only the depth-fusion params for N
# epochs (body frozen), then unfreezes. DEPTH_FUSE_LR = separate LR group for
# the fusion params (0 = follow LR). tag _probe<N>.
PROBE_EPOCHS="${PROBE_EPOCHS:-0}"
DEPTH_FUSE_LR="${DEPTH_FUSE_LR:-0}"
# adaptive probe: runs until val_select stalls for PROBE_PLATEAU_PATIENCE epochs
# (LR cut every PROBE_LR_PATIENCE stalled epochs). tag _probeA.
PROBE_ADAPTIVE="${PROBE_ADAPTIVE:-0}"
PROBE_ONLY="${PROBE_ONLY:-0}" # 1 = stop when the probe ends, tag suffix "only"
PROBE_PLATEAU_PATIENCE="${PROBE_PLATEAU_PATIENCE:-5}"
PROBE_LR_PATIENCE="${PROBE_LR_PATIENCE:-2}"
# a probe without depth fusion trains nothing, force it off for geco2_finetuned
if [[ -z "$USE_DEPTH" && ( "$PROBE_EPOCHS" != "0" || "$PROBE_ADAPTIVE" == "1" ) ]]; then
    echo "[setup][warn] ${EXP_TAG}: no depth fusion to probe -> forcing PROBE_EPOCHS=0 PROBE_ADAPTIVE=0 (was probe_epochs=${PROBE_EPOCHS} adaptive=${PROBE_ADAPTIVE}); the noprobe positional does this explicitly"
    PROBE_EPOCHS=0
    PROBE_ADAPTIVE=0
fi
PROBE_TAG=""
if [[ -n "$USE_DEPTH" && "$PROBE_ADAPTIVE" == "1" ]]; then
    PROBE_TAG="_probeA"
    echo "[setup] ADAPTIVE probe: plateau_patience=${PROBE_PLATEAU_PATIENCE} lr_patience=${PROBE_LR_PATIENCE} depth_fuse_lr=${DEPTH_FUSE_LR} min_epochs=${PROBE_EPOCHS} (tag ${PROBE_TAG})"
elif [[ -n "$USE_DEPTH" && "$PROBE_EPOCHS" != "0" ]]; then
    PROBE_TAG="_probe${PROBE_EPOCHS}"
    echo "[setup] Probe-then-finetune: probe_epochs=${PROBE_EPOCHS} depth_fuse_lr=${DEPTH_FUSE_LR} (tag ${PROBE_TAG})"
fi

if [[ -n "$PROBE_TAG" && "$PROBE_ONLY" == "1" ]]; then
    PROBE_TAG="${PROBE_TAG}only"
    echo "[setup] PROBE_ONLY: training stops when the probe ends -- body never unfrozen (tag ${PROBE_TAG})"
fi

# once best val_select reaches GOOD_SELECT, plateau patience drops to
# GOOD_PLATEAU_PATIENCE. 0 = disabled.
GOOD_SELECT="${GOOD_SELECT:-0}"
GOOD_PLATEAU_PATIENCE="${GOOD_PLATEAU_PATIENCE:-2}"

# FSC147 is read-only-shared at this path on the cluster
DATA=/d/hpc/projects/FRI/pelhanj/fsc147

# precomputed depth-map cache (DAv2-L, 4 scales averaged). USE_DEPTHMAPS=1 reads
# depth from disk and skips the in-model ViT. USE_AVAILABLE_DEPTHMAPS=0 generates
# missing maps once (rank 0), =1 requires them.
USE_DEPTHMAPS="${USE_DEPTHMAPS:-0}"
DEPTHMAPS_DIR="${DEPTHMAPS_DIR:-/d/hpc/home/er52565/GECO2/FSCD147Dataset/depthmaps}"
USE_AVAILABLE_DEPTHMAPS="${USE_AVAILABLE_DEPTHMAPS:-0}"
# feature cache (CACHE_DEPTHFEATS=1): path_1 cached to disk (fp16 npy +
# pca_basis.npz), one dir per (channels, dinoInSize, pca on/off)
DEPTH_PCA_CHANNELS="${DEPTH_PCA_CHANNELS:-16}"
# -1 = raw (PCA off) -> "raw" suffix
if [[ "$DEPTH_PCA_CHANNELS" == "-1" ]]; then _DFSUF="raw"; else _DFSUF="$DEPTH_PCA_CHANNELS"; fi
# dir also keyed by the resolved dinoInSize (518 stays suffix-free)
_DFDINO="$DINO_INPUT_SIZE"
[[ "$_DFDINO" == "0" ]] && _DFDINO=$(( DEPTH_TARGET_SIZE > 518 ? DEPTH_TARGET_SIZE : 518 ))
_DFDIRSUF="$_DFSUF"; [[ "$_DFDINO" != "518" ]] && _DFDIRSUF="${_DFSUF}_dino${_DFDINO}"
DEPTHFEATS_DIR="${DEPTHFEATS_DIR:-/d/hpc/home/er52565/GECO2/FSCD147Dataset/depthfeats${_DFDIRSUF}}"
USE_AVAILABLE_DEPTHFEATS="${USE_AVAILABLE_DEPTHFEATS:-0}" # 0 = generate missing once (rank 0), 1 = require
# cached features need the 1-ch map cache, so force it on; the live decoder tap
# gets the 1-ch from the in-model ViT, so force it off there
if [[ "$CACHE_DEPTHFEATS" == "1" ]]; then
    if [[ "$USE_DEPTHMAPS" != "1" ]]; then
        echo "[setup] CACHE_DEPTHFEATS=1: cached path_1 rides on the 1-ch map -> forcing USE_DEPTHMAPS=1"
        USE_DEPTHMAPS=1
    fi
elif [[ "$DEPTH_SOURCE" == "decoder" ]]; then
    USE_DEPTHMAPS=0
fi
# timing helpers; the total line prints from the exit trap so it fires on
# scancel / time-limit too
_RUN_T0=$(date +%s)
declare -A _STAGE_T0
_fmt_hms() { printf '%02d:%02d:%02d' $(( $1 / 3600 )) $(( $1 % 3600 / 60 )) $(( $1 % 60 )); }
stage_start() { _STAGE_T0["$1"]=$(date +%s); echo "[time] ---- $1 START $(date '+%Y-%m-%d %H:%M:%S')"; }
stage_end() {
    local _now; _now=$(date +%s)
    echo "[time] ---- $1 END   $(date '+%Y-%m-%d %H:%M:%S') (elapsed $(_fmt_hms $(( _now - ${_STAGE_T0["$1"]:-$_now} ))))"
}
print_total_runtime() {
    echo "[time] ==== TOTAL RUN $(date '+%Y-%m-%d %H:%M:%S') (elapsed $(_fmt_hms $(( $(date +%s) - _RUN_T0 ))) since script start)"
}
echo "[time] ==== RUN START $(date '+%Y-%m-%d %H:%M:%S') (job ${SLURM_JOB_ID:-local})"

# per-run results folder <RESULTS_ROOT>/<jobid>_<experiment>/; RUN_SUBDIR
# overrides (RUN_SUBDIR=. = flat layout)
RESULTS_ROOT=/d/hpc/home/er52565/GECO2/results_FSCD147
RUN_SUBDIR="${RUN_SUBDIR:-${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}_${EXP_TAG}}"
MODELS="${RESULTS_ROOT}/${RUN_SUBDIR}"
mkdir -p "$MODELS"

# copy this run's SLURM .out/.err into the results dir on exit
LOGS_DIR=/d/hpc/home/er52565/GECO2/logs
_run_logs_copied=0
copy_run_logs() {
    [[ "$_run_logs_copied" == "1" ]] && return 0
    _run_logs_copied=1
    print_total_runtime
    local jid="${SLURM_JOB_ID:-}"
    [[ -z "$jid" ]] && return 0
    mkdir -p "$MODELS"
    local f
    for f in "$LOGS_DIR"/*"${jid}".out "$LOGS_DIR"/*"${jid}".err; do
        [[ -e "$f" ]] && cp -f "$f" "$MODELS/"
    done
    echo "[logs] copied this run's .out/.err (job ${jid}) into $MODELS/"
}
trap 'copy_run_logs' EXIT
trap 'copy_run_logs; exit 143' TERM
trap 'copy_run_logs; exit 130' INT
# seed 42 is untagged; SEED!=42 tags _s<seed> (stage 2 needs it to find the ckpt)
SEED="${SEED:-42}"
SEED_TAG=""; [[ "$SEED" != "42" ]] && SEED_TAG="_s${SEED}"
# separate warmup for the aux LR groups; 0 = warm with the body. tag _aw<N>.
AUX_LR_WARMUP_EPOCHS="${AUX_LR_WARMUP_EPOCHS:-0}"; AW_TAG=""; [[ "$AUX_LR_WARMUP_EPOCHS" != "0" ]] && AW_TAG="_aw${AUX_LR_WARMUP_EPOCHS}"
# _pdm tag (cached maps; _pdmgen when this run generates missing ones). the
# in-model ViT tags don't apply then, blank them below.
PDM_TAG=""; PDF_TAG=""
if [[ "$USE_DEPTHMAPS" == "1" && -n "$USE_DEPTH" ]]; then
    if [[ "$USE_AVAILABLE_DEPTHMAPS" == "1" ]]; then PDM_TAG="_pdm"; else PDM_TAG="_pdmgen"; fi
    if [[ "$CACHE_DEPTHFEATS" == "1" ]]; then
        if [[ "$USE_AVAILABLE_DEPTHFEATS" == "1" ]]; then PDF_TAG="_pdf${_DFSUF}"; else PDF_TAG="_pdf${_DFSUF}gen"; fi
    fi
    # with USE_DEPTHFEATS=1 the path_1 features still come from DEPTH_CHECKPOINT,
    # so a non-default supplier keeps its tag; ViT-L is untagged
    if [[ "$USE_DEPTHFEATS" != "1" || "$DEPTH_CHECKPOINT" == "depth_anything_v2_vitl.pth" ]]; then
        DMODEL_TAG=""
    fi
    PATH1_TAG=""
    # mode 3 fuses cached maps at native image_size, so tag _drfull instead of _dr<N>
    if [[ "$USE_DEPTH" == "3" ]]; then DR_TAG="_drfull"; else DR_TAG=""; fi
    [[ "$CACHE_DEPTHFEATS" != "1" ]] && DIN_TAG="" # the _pdf cache is keyed by dinoInSize, keep its tag
fi

# computed once, identical for train (where to save) and inference (what to load)
MODEL_NAME="GECO2_FSCD147_${EXP_TAG}${CROP_TAG}${PDM_TAG}${PDF_TAG}${DMODEL_TAG}${DIN_TAG}${DR_TAG}${PATH1_TAG}${DCONV_TAG}${LCUES_TAG}${OINIT_TAG}${MCONV_TAG}${DFC_TAG}${DNORM_TAG}${SEPCUES_TAG}${SFR_TAG}${KERNEL_TAG}${FFMBN_TAG}${IDINIT_TAG}${PLG_TAG}${HIRES_TAG}${DENS_TAG}${PROBE_TAG}${AW_TAG}${SEED_TAG}_${REGIME_TAG}"
# the checkpoint stage 1 saves and stage 2 loads (train script appends _<epochs>)
CKPT_PATH="${MODELS}/${MODEL_NAME}_${EPOCHS}.pth"

# pretrained init, default GECO2FSCD.pth (the paper's FSC147-final weights); new
# depth params stay random-init via strict=False. overrides:
#   PRETRAINED_INIT=/d/hpc/home/er52565/GECO2/CNTQG_multitrain_ca44.pth (shared prior)
#   PRETRAINED_INIT="" ALLOW_FROM_SCRATCH=1 (from scratch)
PRETRAINED_INIT="${PRETRAINED_INIT-/d/hpc/home/er52565/GECO2/GECO2FSCD.pth}"
# geco2_pretrained: symlink the released ckpt under this run's MODEL_NAME (the
# inference script rebuilds the path from --model_path/--model_name/--ckpt_epochs)
if [[ "${PRETRAINED_EVAL:-0}" == "1" ]]; then
    if [[ ! -f "$PRETRAINED_INIT" ]]; then
        echo "[error] geco2_pretrained: PRETRAINED_INIT not found: $PRETRAINED_INIT"
        exit 1
    fi
    ln -sf "$PRETRAINED_INIT" "$CKPT_PATH"
    echo "[setup] geco2_pretrained: evaluating PRETRAINED_INIT as-is -> $CKPT_PATH -> $PRETRAINED_INIT"
fi

scontrol update JobId="$SLURM_JOB_ID" JobName="GECO2-trinf-H100[${RUN_STAGES}:FSCD147_${EXP_TAG}${CROP_TAG}${PDM_TAG}${PDF_TAG}${DMODEL_TAG}${DIN_TAG}${DR_TAG}${PATH1_TAG}${DCONV_TAG}${LCUES_TAG}${OINIT_TAG}${MCONV_TAG}${DFC_TAG}${DNORM_TAG}${SEPCUES_TAG}${SFR_TAG}${KERNEL_TAG}${FFMBN_TAG}${IDINIT_TAG}${PLG_TAG}${HIRES_TAG}${DENS_TAG}${PROBE_TAG}${AW_TAG}${SEED_TAG}_${REGIME_TAG}_${EPOCHS}ep_${INFER_TYPE}_${TEST_SPLIT}]" 2>/dev/null || true

# arch flags, must match between the two stages; image_size=1024 per the GeCo2 paper
BACKBONE=SAM
REDUCTION=16
IMAGE_SIZE=1024
NUM_ENC_LAYERS=3
EMB_DIM=256
NUM_HEADS=8
# counter.py forces kernel_dim to 1 anyway; 1 matches the official train.sh
KERNEL_DIM="${KERNEL_DIM:-1}"
NUM_OBJECTS=3

# stage 1 hyper-params (official GeCo2 train.sh recipe: batch 4, lr 1e-4, wd 1e-5,
# aux 0.3, tiling_p 0.5, dropout 0.1, grad clip 0.1, 200 ep). our LR schedule is
# WarmupThenPlateau + spike/plateau early stop, not the paper's --lr_drop 50.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${BATCH_SIZE:-4}}"
NUM_WORKERS="${NUM_WORKERS:-12}"
DROPOUT="${DROPOUT:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"
AUX_WEIGHT="${AUX_WEIGHT:-0.3}"
TRAIN_TILING_P="${TRAIN_TILING_P:-0.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
# body LR 2e-5, not the paper's 1e-4: 1e-4 degrades the converged body. same
# recipe for every variant incl. geco2.
LR="${LR:-2e-5}"
# patience 3 = LR cuts at ~stall 4 and 8, both before the hard stop at 10
REDUCE_LR_PATIENCE="${REDUCE_LR_PATIENCE:-3}"
REDUCE_LR_FACTOR="${REDUCE_LR_FACTOR:-0.5}"
# linear LR warmup; FSCD 3ep@0.1, IOC/MCAC 5ep@0.05
LR_WARMUP_EPOCHS="${LR_WARMUP_EPOCHS:-3}"
LR_WARMUP_START_FACTOR="${LR_WARMUP_START_FACTOR:-0.1}"
SPIKE_PATIENCE="${SPIKE_PATIENCE:-2}"
SPIKE_RATIO="${SPIKE_RATIO:-2.0}"
# stop after this many epochs with no val_select/NAE improvement; 0 = disabled
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-10}"

# stage 2 (inference) settings
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-16}"
INFER_TILING_P=0.0
# REPORT_MAX_GT: space-separated GT thresholds for extra stratified rows; "" disables
REPORT_MAX_GT="${REPORT_MAX_GT:-1500}"
# tiled thresholds: SCORE_ABS_THR = absolute box_v cutoff, SCORE_REL_THR =
# optional relative cap (0=off), EDGE_MARGIN in original px
TILE_SIZE=256
TILE_OVERLAP=64
TILE_BATCH_SIZE=8
NMS_IOU=0.5
SCORE_ABS_THR=0.1
SCORE_REL_THR=0.0
EDGE_MARGIN=8
WHOLE_IMAGE_PASS=1
# COCO dump: __AUTO__ = per-(split,type) path; set a path to force, "" to skip
COCO_JSON_OUT_ENV="${COCO_JSON_OUT-__AUTO__}"

# eval splits: default = primary split (arg 7) plus val (Table 1 has both
# halves); override with a space-separated EVAL_SPLITS
if [[ -n "${EVAL_SPLITS:-}" ]]; then
    read -r -a EVAL_SPLITS_ARR <<< "$EVAL_SPLITS"
else
    EVAL_SPLITS_ARR=("$TEST_SPLIT")
    case "$TEST_SPLIT" in val|val_coco) : ;; *) EVAL_SPLITS_ARR+=("val") ;; esac
fi
for _s in "${EVAL_SPLITS_ARR[@]}"; do
    case "$_s" in
        test|test_coco|val|val_coco) : ;;
        *) echo "[error] EVAL_SPLITS has invalid split '$_s'. Accepted: test test_coco val val_coco"; exit 1 ;;
    esac
done

# stage 3: score the COCO dump with eval_bboxes.py into *_cocoeval.txt
# (AP/AP50/AP75, 0-100). 0 = skip.
COCO_EVAL="${COCO_EVAL:-1}"

# stage 0 (geco2_pretrained only): UPSTREAM_EVAL=1 also runs the original
# inference.py + eval_bboxes.py over val+test
UPSTREAM_EVAL="${UPSTREAM_EVAL:-1}"

# COCO-matcher visuals (stage 3): COCO_VIS=1 colors GT/pred boxes by the COCOeval
# matching. off by default.
COCO_VIS="${COCO_VIS:-0}"
COCO_VIS_IOU_THR="${COCO_VIS_IOU_THR:-0.5}"
COCO_VIS_EVERY="${COCO_VIS_EVERY:-1}"
COCO_VIS_MAX="${COCO_VIS_MAX:-0}"

mkdir -p "$PROJECT/logs" "$MODELS"
echo "[setup] Outputs will land in $MODELS (created if needed)."

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=$(( 10000 + SLURM_JOB_ID % 20000 ))
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1

module load Anaconda3/2023.07-2
module load CUDA/12.1.1
source /d/hpc/home/er52565/.bashrc
module load Anaconda3 && eval "$(conda shell.bash hook)" && conda activate cnt2

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PYTHONPATH"

# pre-warm the depth checkpoint into the shared HF cache (nodes are offline);
# a failure here only warns, the model build retries
if [[ -n "$USE_DEPTH" ]]; then
    echo "[setup] pre-warming depth checkpoint (DEPTH_CHECKPOINT=${DEPTH_CHECKPOINT})..."
    python - <<'PYWARM' || echo "[setup][warn] depth prewarm failed (offline node + cold cache?); the model build will retry."
import os
from Depther.infer_depth import _DEPTH_CONFIGS, _hf_download_with_retries, _load_dinov2_arch
ckpt = os.environ.get("DEPTH_CHECKPOINT", "depth_anything_v2_vitl.pth")
cfg = _DEPTH_CONFIGS[ckpt]
print("[setup] depth ckpt cached:", _hf_download_with_retries(repo_id=cfg["repo_id"], filename=ckpt))
if cfg.get("head", "sdt") == "sdt":
    _load_dinov2_arch(cfg["hub_name"])
    print("[setup] dinov2 hub arch cached:", cfg["hub_name"])
PYWARM
fi

echo "[setup] host=$(hostname) date=$(date -Iseconds)"
echo "[setup] run_stages=${RUN_STAGES} experiment=${EXP_TAG} regime=${REGIME_TAG} epochs=${EPOCHS} train_scale=${SCALE} infer=${INFER_TYPE} eval_splits=[${EVAL_SPLITS_ARR[*]}] coco_eval=${COCO_EVAL}"
echo "[setup] model_name=${MODEL_NAME} -> ckpt ${CKPT_PATH}"
echo "[setup] crop_p=${CROP_P} crop_px=[${CROP_MIN_PX},${CROP_MAX_PX}] use_density=${USE_DENSITY}${DENS_TAG:+ (tag ${DENS_TAG})}"

# Dump the effective config, one grep-able line per knob (<unset> = not set here).
echo "[config-dump] git_commit=$(git -C "$PROJECT" rev-parse --short HEAD 2>/dev/null || echo n/a)$(git -C "$PROJECT" diff --quiet 2>/dev/null || echo +dirty) config_file=${CONFIG_FILE:-<none>}"
for _v in \
    RUN_STAGES EXPERIMENT EXP_TAG REGIME REGIME_TAG TEST_SPLIT EPOCHS SCALE INFER_TYPE VISUALS_EVERY SWEEP PROBE_ARG \
    RESULTS_ROOT RUN_SUBDIR MODELS CKPT_JOB MODEL_NAME CKPT_PATH PRETRAINED_INIT ALLOW_FROM_SCRATCH \
    BACKBONE REDUCTION IMAGE_SIZE NUM_ENC_LAYERS EMB_DIM NUM_HEADS KERNEL_DIM NUM_OBJECTS \
    USE_DEPTH DEPTH_CHECKPOINT DINO_INPUT_SIZE DEPTH_TARGET_SIZE DEPTH_SOURCE USE_DEPTHFEATS CACHE_DEPTHFEATS DEPTH_PCA_CHANNELS \
    USE_DEPTHMAPS USE_AVAILABLE_DEPTHMAPS USE_AVAILABLE_DEPTHFEATS DEPTHMAPS_DIR DEPTHFEATS_DIR \
    DEPTH_ADAPT DEPTH_CUES DEPTH_ADAPT_INIT DEPTH_ADAPT_MASKED_CONV DEPTH_FEAT_CHANNELS DEPTH_FEAT_NORM DEPTH_FEAT_NORM_GROUPS \
    DEPTH_KERNEL_SIZE DEPTH_FUSE_IDENTITY_INIT DEPTH_HIRES_FUSION DEPTH_HIRES_NORM DEPTH_FUSE_LR \
    SEP_HIERA_INPUT SEP_HIERA_FULLRES SEP_HIERA_PER_LEVEL_GATE FFM_NORM \
    LR WEIGHT_DECAY DROPOUT MAX_GRAD_NORM AUX_WEIGHT TRAIN_TILING_P TRAIN_BATCH_SIZE NUM_WORKERS SEED \
    LR_WARMUP_EPOCHS LR_WARMUP_START_FACTOR AUX_LR_WARMUP_EPOCHS \
    REDUCE_LR_PATIENCE REDUCE_LR_FACTOR PLATEAU_PATIENCE SPIKE_PATIENCE SPIKE_RATIO SELECT_RMSE_WEIGHT GOOD_SELECT GOOD_PLATEAU_PATIENCE \
    PROBE_ADAPTIVE PROBE_ONLY PROBE_EPOCHS PROBE_PLATEAU_PATIENCE PROBE_LR_PATIENCE \
    USE_DENSITY DENSITY_LOSS_TYPE DENSITY_HEAD_TYPE DENSITY_WEIGHT DENSITY_ABS_COUNT_WEIGHT DENSITY_LR \
    DENSITY_ADAPTIVE_SIGMA DENSITY_SIGMA DENSITY_SIGMA_MIN DENSITY_SIGMA_MAX DENSITY_SIGMA_K DENSITY_SIGMA_BETA DENSITY_GUIDED \
    WHOLE_SWEEP INFER_BATCH_SIZE INFER_TILING_P REPORT_MAX_GT EVAL_SPLITS UPSTREAM_EVAL \
    COCO_EVAL COCO_VIS COCO_VIS_EVERY COCO_VIS_IOU_THR COCO_VIS_MAX COCO_JSON_OUT_ENV \
    CROP_P CROP_MIN_PX CROP_MAX_PX TILE_SIZE TILE_OVERLAP TILE_BATCH_SIZE NMS_IOU SCORE_ABS_THR SCORE_REL_THR EDGE_MARGIN WHOLE_IMAGE_PASS \
    SWEEP_IMAGES SWEEP_ABS SWEEP_NMS SWEEP_EDGE; do
    printf '[config-dump] %s=%s\n' "$_v" "${!_v-<unset>}"
done
echo "[config-dump] extra_train_args=[${EXTRA_ARGS[*]}]"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# recompile Deformable-DETR CUDA ops for this GPU arch
echo "[setup] Recompiling Deformable-DETR CUDA ops on $(hostname)..."
cd "$PROJECT/Deformable-DETR/models/ops"
python setup.py build install 2>&1 | tail -10
cd "$PROJECT"
echo "[setup] Deformable-DETR ops compiled."

# shared flag arrays
DEPTH_FLAGS=()
if [[ -n "$USE_DEPTH" ]]; then
    DEPTH_FLAGS+=(--use_depth "$USE_DEPTH" --depth_feat_channels "$DEPTH_FEAT_CHANNELS")
    if [[ -n "$DEPTH_KERNEL_SIZE" ]]; then
        DEPTH_FLAGS+=(--depth_kernel_size "$DEPTH_KERNEL_SIZE")
    fi
    DEPTH_FLAGS+=(--depth_fuse_identity_init "$DEPTH_FUSE_IDENTITY_INIT" \
                  --depth_feat_norm "$DEPTH_FEAT_NORM" \
                  --depth_feat_norm_groups "$DEPTH_FEAT_NORM_GROUPS" \
                  --depth_target_size "$DEPTH_TARGET_SIZE" \
                  --depth_source "$DEPTH_SOURCE" \
                  --depth_adapt "$DEPTH_ADAPT" \
                  --depth_cues "$DEPTH_CUES" \
                  --depth_adapt_init "$DEPTH_ADAPT_INIT" \
                  --depth_adapt_masked_conv "$DEPTH_ADAPT_MASKED_CONV" \
                  --sep_hiera_input "$SEP_HIERA_INPUT" \
                  --sep_hiera_fullres "$SEP_HIERA_FULLRES" \
                  --sep_hiera_per_level_gate "$SEP_HIERA_PER_LEVEL_GATE" \
                  --ffm_norm "$FFM_NORM" \
                  --vis_every "$VIS_EVERY" \
                  --dino_input_size "$DINO_INPUT_SIZE" \
                  --depth_hires_fusion "$DEPTH_HIRES_FUSION" \
                  --depth_hires_norm "$DEPTH_HIRES_NORM")
    # cached depth maps: depth from disk as the 4th channel, in-model ViT skipped
    if [[ "$USE_DEPTHMAPS" == "1" ]]; then
        DEPTH_FLAGS+=(--depthmaps_dir "$DEPTHMAPS_DIR")
        [[ "$USE_AVAILABLE_DEPTHMAPS" == "1" ]] && DEPTH_FLAGS+=(--use_available_depthmaps)
        if [[ "$CACHE_DEPTHFEATS" == "1" ]]; then
            DEPTH_FLAGS+=(--depthfeats_dir "$DEPTHFEATS_DIR" --decoder_feat_channels_PCA "$DEPTH_PCA_CHANNELS")
            [[ "$USE_AVAILABLE_DEPTHFEATS" == "1" ]] && DEPTH_FLAGS+=(--use_available_depthfeats)
            echo "[setup] cached decoder feats ON ($([[ "$DEPTH_PCA_CHANNELS" == "-1" ]] && echo "RAW path_1, PCA off" || echo "PCA-${DEPTH_PCA_CHANNELS}")): dir=$DEPTHFEATS_DIR use_available=$USE_AVAILABLE_DEPTHFEATS (tag ${PDF_TAG})"
        fi
        echo "[setup] Precomputed depth maps ON: dir=$DEPTHMAPS_DIR use_available=$USE_AVAILABLE_DEPTHMAPS (in-model depth ViT skipped)"
    fi
fi

# stage-1 density flags (train) vs stage-2 density flags (load + rebuild)
TRAIN_DENSITY_FLAGS=()
INFER_DENSITY_FLAGS=()
if [[ "$USE_DENSITY" != "0" ]]; then
    TRAIN_DENSITY_FLAGS+=(--use_density 1 --density_weight "$DENSITY_WEIGHT" --density_loss_type "$DENSITY_LOSS_TYPE" \
                    --density_abs_count_weight "$DENSITY_ABS_COUNT_WEIGHT" --density_lr "$DENSITY_LR" \
                    --density_head_type "$DENSITY_HEAD_TYPE" --density_adaptive_sigma "$DENSITY_ADAPTIVE_SIGMA" \
                    --density_sigma "$DENSITY_SIGMA" --density_sigma_k "$DENSITY_SIGMA_K" \
                    --density_sigma_beta "$DENSITY_SIGMA_BETA" --density_sigma_min "$DENSITY_SIGMA_MIN" \
                    --density_sigma_max "$DENSITY_SIGMA_MAX" --density_guided "$DENSITY_GUIDED")
    INFER_DENSITY_FLAGS+=(--use_density 1 --density_loss_type "$DENSITY_LOSS_TYPE" \
                    --density_head_type "$DENSITY_HEAD_TYPE" --density_guided "$DENSITY_GUIDED")
    echo "[setup] Density head ON: loss=${DENSITY_LOSS_TYPE} head=${DENSITY_HEAD_TYPE} weight=${DENSITY_WEIGHT} abs_count_w=${DENSITY_ABS_COUNT_WEIGHT} density_lr=${DENSITY_LR} adaptive_sigma=${DENSITY_ADAPTIVE_SIGMA} guided=${DENSITY_GUIDED} (count = density.sum())"
fi

INIT_FLAGS=()
if [[ -n "$PRETRAINED_INIT" ]]; then
    if [[ -f "$PRETRAINED_INIT" ]]; then
        INIT_FLAGS+=(--init_from_pretrained "$PRETRAINED_INIT")
        echo "[setup] Initializing CNT from pretrained: $PRETRAINED_INIT"
    else
        echo "[error] PRETRAINED_INIT points to a missing file: $PRETRAINED_INIT" >&2
        echo "[error] not falling back to a from-scratch init, that collapses some" >&2
        echo "[error] variants (e.g. the geco2 RGB baseline)." >&2
        exit 1
    fi
else
    # always init from the pretrained geco2 checkpoint; from-scratch runs need ALLOW_FROM_SCRATCH=1
    if [[ "${ALLOW_FROM_SCRATCH:-0}" == "1" ]]; then
        echo "[setup] PRETRAINED_INIT empty + ALLOW_FROM_SCRATCH=1 -- deliberate FROM-SCRATCH run."
    else
        echo "[error] PRETRAINED_INIT is empty. runs always init from the pretrained" >&2
        echo "[error] GeCo2 checkpoint (FSCD147: GECO2FSCD.pth, IOCfish:" >&2
        echo "[error] CNTQG_multitrain_ca44.pth). for a from-scratch run submit with" >&2
        echo "[error] PRETRAINED_INIT=\"\" ALLOW_FROM_SCRATCH=1." >&2
        exit 1
    fi
fi

# stage-2 visuals + stratified-report flags (identical across whole/tiled passes)
VISUAL_FLAGS=()
if [[ "$VISUALS_EVERY" == "0" ]]; then
    VISUAL_FLAGS+=(--no_visuals)
else
    VISUAL_FLAGS+=(--visuals_every "$VISUALS_EVERY")
fi
REPORT_FLAGS=()
if [[ -n "$REPORT_MAX_GT" ]]; then
    # unquoted on purpose, nargs='+' list
    REPORT_FLAGS+=(--report_max_gt $REPORT_MAX_GT)
fi

# stage 0 (geco2_pretrained only): run the original inference.py + eval_bboxes.py
# on the released weights to anchor the paper numbers. a failure here doesn't
# block stages 2/3, it only sets the job exit code.
UPSTREAM_FAIL=0
if [[ "${PRETRAINED_EVAL:-0}" == "1" && "$UPSTREAM_EVAL" != "0" ]]; then
    UPSTREAM_MODEL_PATH="$(dirname "$PRETRAINED_INIT")"
    UPSTREAM_CKPT_NAME="$(basename "$PRETRAINED_INIT" .pth)"
    stage_start "STAGE 0 upstream paper-parity eval"
    echo "==================== STAGE 0a: upstream inference.py (GPU) ===================="
    echo "[upstream] weights=${UPSTREAM_MODEL_PATH}/${UPSTREAM_CKPT_NAME}.pth -> $PROJECT/geco2_{val,test}.json"
    srun --unbuffered --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" python inference.py \
        --data_path "$DATA" \
        --model_path "$UPSTREAM_MODEL_PATH" \
        --model_name "$UPSTREAM_CKPT_NAME" \
        --image_size "$IMAGE_SIZE" \
        --num_objects "$NUM_OBJECTS" \
        --kernel_dim "$KERNEL_DIM" \
        --batch_size "$INFER_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS"
    UPSTREAM_RC=$?
    if [[ $UPSTREAM_RC -ne 0 ]]; then
        echo "[upstream][error] inference.py exited with code $UPSTREAM_RC -- skipping the upstream eval_bboxes.py"
        UPSTREAM_FAIL=1
    else
        echo "==================== STAGE 0b: upstream eval_bboxes.py (AP/AP50, CPU) ===================="
        python eval_bboxes.py --data_path "$DATA" || UPSTREAM_FAIL=1
    fi
    stage_end "STAGE 0 upstream paper-parity eval"
fi

# stage 1: train (skipped for run-stage=inference)
if [[ "$RUN_STAGES" == "inference" ]]; then
    # inference-only: the checkpoint these knobs name must already exist on disk
    echo "==================== STAGE 1: TRAIN skipped (run-stage=inference) ===================="
    if [[ ! -f "$CKPT_PATH" ]]; then
        # the checkpoint lives in the training job's folder; search the RESULTS_ROOT
        # run subfolders and symlink it under this run's name. CKPT_JOB=<jobid|abs dir>
        # pins the source when several runs trained the same name.
        CKPT_FILE="${MODEL_NAME}_${EPOCHS}.pth"
        CKPT_HITS=()
        if [[ -n "${CKPT_JOB:-}" ]]; then
            CKPT_DIR="$CKPT_JOB"
            [[ "$CKPT_DIR" != /* ]] && CKPT_DIR="${RESULTS_ROOT}/${CKPT_JOB}"
            if [[ ! -d "$CKPT_DIR" && "$CKPT_JOB" != /* ]]; then
                # bare job id: resolve to the single <jobid>_<experiment> match
                _cj=( "${RESULTS_ROOT}/${CKPT_JOB}"_* )
                [[ ${#_cj[@]} -eq 1 && -d "${_cj[0]}" ]] && CKPT_DIR="${_cj[0]}"
            fi
            [[ -f "${CKPT_DIR}/${CKPT_FILE}" ]] && CKPT_HITS+=("${CKPT_DIR}/${CKPT_FILE}")
        else
            # only real files count (! -L): earlier inference-only jobs leave
            # symlinks that would make a second re-eval die as ambiguous
            [[ -f "${RESULTS_ROOT}/${CKPT_FILE}" && ! -L "${RESULTS_ROOT}/${CKPT_FILE}" ]] && CKPT_HITS+=("${RESULTS_ROOT}/${CKPT_FILE}")
            for _c in "${RESULTS_ROOT}"/*/"${CKPT_FILE}"; do
                [[ -f "$_c" && ! -L "$_c" && "$_c" != "$CKPT_PATH" ]] && CKPT_HITS+=("$_c")
            done
        fi
        if [[ ${#CKPT_HITS[@]} -eq 1 ]]; then
            ln -sf "${CKPT_HITS[0]}" "$CKPT_PATH"
            echo "[infer] linked existing checkpoint: ${CKPT_HITS[0]} -> $CKPT_PATH"
            # link the training run's *_metrics.txt too, the parse tools expect it
            # as a sibling of the results txt
            _mt="$(dirname "${CKPT_HITS[0]}")/${MODEL_NAME}_metrics.txt"
            [[ -f "$_mt" ]] && ln -sf "$_mt" "${MODELS}/${MODEL_NAME}_metrics.txt"
        elif [[ ${#CKPT_HITS[@]} -gt 1 ]]; then
            echo "[error] run-stage=inference: ${CKPT_FILE} exists in several run folders -- pin one with CKPT_JOB=<jobid|abs dir>:"
            printf '[error]   %s\n' "${CKPT_HITS[@]}"
            exit 1
        elif [[ -n "${CKPT_JOB:-}" ]]; then
            echo "[error] run-stage=inference: no checkpoint ${CKPT_FILE} in CKPT_JOB dir ${CKPT_DIR} (only that dir is searched when CKPT_JOB is set)."
            exit 1
        else
            echo "[error] run-stage=inference but no checkpoint ${CKPT_FILE} in ${MODELS}, ${RESULTS_ROOT} (legacy flat) or any ${RESULTS_ROOT}/*/ run folder."
            echo "[error] Train it first (run-stage train|both), or check the knobs/epochs match the saved name."
            exit 1
        fi
    fi
    echo "[infer] using existing checkpoint: $CKPT_PATH"
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[infer][warn] ignoring train-only passthrough args in inference mode: ${EXTRA_ARGS[*]}"
else
    stage_start "STAGE 1 train"
    echo "==================== STAGE 1: TRAIN (${EXP_TAG}, ${REGIME_TAG}, ${EPOCHS} ep) ===================="
    echo "[train] script=${TRAIN_SCRIPT} batch=${TRAIN_BATCH_SIZE} lr=${LR} wd=${WEIGHT_DECAY} warmup=${LR_WARMUP_EPOCHS}ep@${LR_WARMUP_START_FACTOR} dropout=${DROPOUT} grad_clip=${MAX_GRAD_NORM} aux_w=${AUX_WEIGHT} spike=${SPIKE_PATIENCE}/${SPIKE_RATIO} tiling_p=${TRAIN_TILING_P} crop_p=${CROP_P}"

    srun --unbuffered --cpu-bind=none --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" python "$TRAIN_SCRIPT" \
        --training \
        --model_name "$MODEL_NAME" \
        --model_path "$MODELS" \
        --data_path "$DATA" \
        --backbone "$BACKBONE" \
        --reduction "$REDUCTION" \
        --image_size "$IMAGE_SIZE" \
        --num_enc_layers "$NUM_ENC_LAYERS" \
        --emb_dim "$EMB_DIM" \
        --num_heads "$NUM_HEADS" \
        --kernel_dim "$KERNEL_DIM" \
        --num_objects "$NUM_OBJECTS" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --seed "$SEED" \
        --batch_size "$TRAIN_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --dropout "$DROPOUT" \
        --max_grad_norm "$MAX_GRAD_NORM" \
        --aux_weight "$AUX_WEIGHT" \
        --tiling_p "$TRAIN_TILING_P" \
        --crop_p "$CROP_P" \
        --crop_min_px "$CROP_MIN_PX" \
        --crop_max_px "$CROP_MAX_PX" \
        --reduce_lr_patience "$REDUCE_LR_PATIENCE" \
        --reduce_lr_factor "$REDUCE_LR_FACTOR" \
        --lr_warmup_epochs "$LR_WARMUP_EPOCHS" \
        --lr_warmup_start_factor "$LR_WARMUP_START_FACTOR" \
        --aux_lr_warmup_epochs "$AUX_LR_WARMUP_EPOCHS" \
        --spike_patience "$SPIKE_PATIENCE" \
        --spike_ratio "$SPIKE_RATIO" \
        --plateau_patience "$PLATEAU_PATIENCE" \
        --good_select "$GOOD_SELECT" \
        --good_plateau_patience "$GOOD_PLATEAU_PATIENCE" \
        --probe_epochs "$PROBE_EPOCHS" \
        --probe_adaptive "$PROBE_ADAPTIVE" \
        --probe_only "$PROBE_ONLY" \
        --probe_plateau_patience "$PROBE_PLATEAU_PATIENCE" \
        --probe_lr_patience "$PROBE_LR_PATIENCE" \
        --depth_fuse_lr "$DEPTH_FUSE_LR" \
        --select_rmse_weight "$SELECT_RMSE_WEIGHT" \
        "${DEPTH_FLAGS[@]}" \
        "${TRAIN_DENSITY_FLAGS[@]}" \
        "${MAX_OBJECTS_FLAGS[@]}" \
        "${INIT_FLAGS[@]}" \
        --pre_norm \
        "${EXTRA_ARGS[@]}"
    TRAIN_RC=$?
    echo "[train] training srun exited with code ${TRAIN_RC}"
    stage_end "STAGE 1 train"

    # the best ckpt is saved on every val improvement, so a time-limit SIGTERM still
    # leaves a valid .pth; gate stage 2 on the file, not the exit code
    if [[ ! -f "$CKPT_PATH" ]]; then
        echo "[error] Expected checkpoint not found after training: $CKPT_PATH"
        echo "[error] Training likely failed before the first checkpoint was written -- skipping inference."
        exit "${TRAIN_RC:-1}"
    fi
    echo "[train] checkpoint present: $CKPT_PATH"
fi

# train-only: stop before inference
if [[ "$RUN_STAGES" == "train" ]]; then
    echo "[done] run-stage=train -> training complete, inference skipped."
    exit 0
fi

# infer_type=none also means no inference
if [[ ${#INFER_TYPES[@]} -eq 0 ]]; then
    if [[ "$RUN_STAGES" == "inference" ]]; then
        echo "[error] run-stage=inference but infer_type=none -- nothing to do."
        exit 1
    fi
    echo "[done] infer_type=none -> training complete, inference skipped."
    exit 0
fi

# stage 2: inference, once per requested type
run_inference() {
    local split="$1"
    local infer_type="$2"
    local infer_script results_suffix split_uc
    case "$infer_type" in
        whole) infer_script="training/inference_whole_on_FSCD147_dataset.py" ;;
        tiled) infer_script="training/inference_tiled_on_FSCD147_dataset.py" ;;
    esac
    # per-split suffix: test -> _TEST_<type>, val -> _VAL_<type>
    split_uc=$(echo "$split" | tr '[:lower:]' '[:upper:]')
    results_suffix="_${split_uc}_${infer_type}"

    # whole-only: WHOLE_SWEEP=1 calibrates box_v threshold + NMS IoU on val
    local whole_flags=()
    if [[ "$infer_type" == "whole" && "${WHOLE_SWEEP:-0}" != "0" ]]; then
        whole_flags+=(--whole_sweep 1)
    fi

    # tiled-only flags (incl. the optional val sweep)
    local tiled_flags=()
    if [[ "$infer_type" == "tiled" ]]; then
        tiled_flags+=(
            --tile_size "$TILE_SIZE"
            --tile_overlap "$TILE_OVERLAP"
            --tile_batch_size "$TILE_BATCH_SIZE"
            --nms_iou "$NMS_IOU"
            --score_abs_thr "$SCORE_ABS_THR"
            --score_rel_thr "$SCORE_REL_THR"
            --edge_margin "$EDGE_MARGIN"
            --whole_image_pass "$WHOLE_IMAGE_PASS"
        )
        if [[ "$SWEEP" == "1" || "$SWEEP" == "2" ]]; then
            local sweep_flag
            [[ "$SWEEP" == "2" ]] && sweep_flag="--sweep_then_test" || sweep_flag="--sweep"
            tiled_flags+=("$sweep_flag" --max_images "${SWEEP_IMAGES:-150}")
            [[ -n "$SWEEP_ABS"  ]] && tiled_flags+=(--sweep_score_abs_thr "$SWEEP_ABS")
            [[ -n "$SWEEP_NMS"  ]] && tiled_flags+=(--sweep_nms_iou "$SWEEP_NMS")
            [[ -n "$SWEEP_EDGE" ]] && tiled_flags+=(--sweep_edge_margin "$SWEEP_EDGE")
        fi
    fi

    # COCO dump path, per (split,type)
    local coco_flags=() coco_out
    if [[ "$COCO_JSON_OUT_ENV" == "__AUTO__" ]]; then
        coco_out="${MODELS}/${MODEL_NAME}${results_suffix}_coco_preds.json"
    else
        coco_out="$COCO_JSON_OUT_ENV"   # may be "" to skip
    fi
    [[ -n "$coco_out" ]] && coco_flags+=(--coco_json_out "$coco_out")

    stage_start "STAGE 2 inference ${infer_type}/${split}"
    echo "-------------------- inference: type=${infer_type} split=${split} vis_every=${VISUALS_EVERY} sweep=${SWEEP} --------------------"
    echo "[infer] script=${infer_script} -> ${MODELS}/${MODEL_NAME}${results_suffix}_results.txt"
    if [[ "$infer_type" == "tiled" ]]; then
        echo "[infer] tiled thresholds: score_abs_thr=${SCORE_ABS_THR} score_rel_thr(cap)=${SCORE_REL_THR} nms_iou=${NMS_IOU} edge_margin_px=${EDGE_MARGIN}"
        [[ "$SWEEP" == "1" || "$SWEEP" == "2" ]] && echo "[infer] SWEEP mode=${SWEEP} on first ${SWEEP_IMAGES:-150} VAL images (abs=${SWEEP_ABS:-default} nms=${SWEEP_NMS:-default} edge=${SWEEP_EDGE:-default})"
    fi
    [[ -n "$coco_out" ]] && echo "[infer] COCO predictions dump -> ${coco_out}"
    [[ -n "$REPORT_MAX_GT" ]] && echo "[infer] extra stratified report rows for GT<{${REPORT_MAX_GT// /, }}"

    srun --unbuffered --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" python "$infer_script" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODELS" \
        --data_path "$DATA" \
        --backbone "$BACKBONE" \
        --reduction "$REDUCTION" \
        --image_size "$IMAGE_SIZE" \
        --num_enc_layers "$NUM_ENC_LAYERS" \
        --emb_dim "$EMB_DIM" \
        --num_heads "$NUM_HEADS" \
        --kernel_dim "$KERNEL_DIM" \
        --num_objects "$NUM_OBJECTS" \
        --batch_size "$INFER_BATCH_SIZE" \
        --tiling_p "$INFER_TILING_P" \
        --ckpt_epochs "$EPOCHS" \
        --test_split "$split" \
        --results_suffix "$results_suffix" \
        "${whole_flags[@]}" \
        "${tiled_flags[@]}" \
        "${VISUAL_FLAGS[@]}" \
        "${REPORT_FLAGS[@]}" \
        "${coco_flags[@]}" \
        "${DEPTH_FLAGS[@]}" \
        "${INFER_DENSITY_FLAGS[@]}"
    local rc=$?
    echo "[infer] type=${infer_type} srun exited with code ${rc}"
    stage_end "STAGE 2 inference ${infer_type}/${split}"

    # stage 3: official COCO AP via eval_bboxes.py into *_cocoeval.txt (the AP in
    # *_results.txt is the 0-1 greedy one); skipped when there's no dump
    if [[ "$COCO_EVAL" != "0" && -n "$coco_out" && -f "$coco_out" ]]; then
        local cocoeval_txt="${MODELS}/${MODEL_NAME}${results_suffix}_cocoeval.txt"
        stage_start "STAGE 3 cocoeval ${infer_type}/${split}"
        echo "[cocoeval] OFFICIAL COCO AP on split=${split}: scoring ${coco_out} -> ${cocoeval_txt}"
        # optional COCO-matcher visuals (TP/FP/FN by the official AP matching)
        local coco_vis_flags=()
        if [[ "$COCO_VIS" == "1" ]]; then
            local cocovis_dir="${MODELS}/${MODEL_NAME}${results_suffix}_cocovis"
            coco_vis_flags=(--vis_matches --vis_iou_thr "$COCO_VIS_IOU_THR" \
                            --vis_every "$COCO_VIS_EVERY" --vis_max_images "$COCO_VIS_MAX" \
                            --vis_dir "$cocovis_dir")
            echo "[cocoeval] COCO-matcher visuals ON (IoU=${COCO_VIS_IOU_THR}) -> ${cocovis_dir}"
        fi
        python eval_bboxes.py --data_path "$DATA" --pred_json "$coco_out" --split "$split" \
            "${coco_vis_flags[@]}" 2>&1 | tee "$cocoeval_txt"
        local eval_rc=${PIPESTATUS[0]}
        echo "[cocoeval] eval_bboxes.py exited with code ${eval_rc} (AP/AP50 are 0-100, the paper scale)"
        [[ $eval_rc -ne 0 ]] && rc=$eval_rc
        stage_end "STAGE 3 cocoeval ${infer_type}/${split}"
    elif [[ "$COCO_EVAL" != "0" ]]; then
        echo "[cocoeval] skipped on split=${split} (no COCO dump at '${coco_out}' -- sweep-only or dump disabled)"
    fi
    return $rc
}

INFER_FAIL=0
for split in "${EVAL_SPLITS_ARR[@]}"; do
    for t in "${INFER_TYPES[@]}"; do
        run_inference "$split" "$t" || INFER_FAIL=1
    done
done

echo "==================== DONE ===================="
echo "[done] checkpoint: $CKPT_PATH"
echo "[done] splits evaluated: [${EVAL_SPLITS_ARR[*]}]  x  inference types: [${INFER_TYPES[*]}]"
echo "[done] count + bbox-quality (MAE / MSE*=RMSE / NAE, P/R/F1, and the 0-1 GREEDY-AP diagnostic):"
echo "         $MODELS/${MODEL_NAME}_TEST_<split>_<type>_results.txt"
echo "[done] OFFICIAL COCO AP/AP50 for paper Table 1 (0-100 scale):"
echo "         $MODELS/${MODEL_NAME}_TEST_<split>_<type>_cocoeval.txt"
echo "[done] -> Table 1: take MAE/RMSE from _results.txt (box-count rows; density off),"
echo "         take AP/AP50 from _cocoeval.txt. Compare against expresults_fscd/_repro_GeCo2_FSCD147."
if [[ "${PRETRAINED_EVAL:-0}" == "1" && "$UPSTREAM_EVAL" != "0" ]]; then
    if [[ "$UPSTREAM_FAIL" == "1" ]]; then
        echo "[done][warn] upstream paper-parity stage FAILED (see STAGE 0 above)"
    else
        echo "[done] upstream paper-parity reproduction (Stage 0): MAE/RMSE + AP/AP50 above in this log;"
        echo "         predictions: $PROJECT/geco2_val.json, $PROJECT/geco2_test.json"
    fi
fi
# surface an inference/eval/upstream failure as the job's exit code
exit $(( INFER_FAIL || UPSTREAM_FAIL ))
