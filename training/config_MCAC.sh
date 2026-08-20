#!/usr/bin/env bash
# config_MCAC.sh - MCAC depth-fusion train+inference knobs. mirrors
# config_IOCfish.sh with MCAC-specific defaults.
# MCAC is few-shot (3 exemplars of the target class), so MODE defaults to 'few'.
# samples are per-(image,class), metrics average MAE/RMSE/NAE/SRE over all
# (image,class) pairs (paper Eqs. 5-6), comparable to LOCA and ABC123.
# whole-only inference, no tiled path.
#
# usage: source then submit (env propagates via sbatch --export=ALL):
#   cd ~/GECO2
#   source training/config_MCAC.sh
#   sbatch training/hpc_H100_train_and_inference_MCAC_dataset.sh \
#          "$RUN_STAGES" "$EXPERIMENT" "$MODE" "$EPOCHS" "$TRAIN_SCALE" "$INFER_TYPE" "$VIS_EVERY" "$SWEEP_MODE" "$PROBE_ARG"
# or let the job source it (still pass the positionals):
#   CONFIG_FILE=$PWD/training/config_MCAC.sh \
#     sbatch training/hpc_H100_train_and_inference_MCAC_dataset.sh both conv_depth_add few 200 whole whole 75
# a 'VAR=x CONFIG_FILE=... sbatch' prefix is overwritten here; derive a config
# instead (config_MCAC_densg.sh).


# fixed in the shell (reference only, plain assignments there; exporting them here
# does nothing). arch flags match the warm-start ckpt (CNTQG_multitrain_ca44.pth)
# and must stay identical between train and inference, changing any of them breaks
# ckpt loading:
#   BACKBONE=SAM  REDUCTION=16  IMAGE_SIZE=1024  NUM_ENC_LAYERS=3  EMB_DIM=256
#   NUM_HEADS=8  KERNEL_DIM=3  NUM_OBJECTS=3
# tiled-protocol constants also exist in the shell but MCAC is whole-only, they
# are never read:
#   TILE_SIZE=512  TILE_OVERLAP=128  TILE_BATCH_SIZE=8  NMS_IOU=0.5  SCORE_ABS_THR=0.1
#   SCORE_REL_THR=0.0  EDGE_MARGIN=8  WHOLE_IMAGE_PASS=1  INFER_TILING_P=0.0


# run selection (positional args, pass on the sbatch line)
RUN_STAGES=both # both | train | inference
EXPERIMENT=conv_depth_add # conv_depth_add | conv_hiera | depth_dim | sep_hiera | ffm | geco2
MODE=few # few | zero (MCAC is few-shot, zero-shot is an ablation)
EPOCHS=200 # epoch budget, the plateau stop usually ends it earlier
TRAIN_SCALE=whole # whole only on MCAC
INFER_TYPE=whole # whole only on MCAC
VIS_EVERY=75 # save a viz every Nth inference image (0=off)
SWEEP_MODE=0 # unused on MCAC
PROBE_ARG= # empty = keep the PROBE_* knobs, noprobe = force them off (for geco2_finetuned)


# depth supply (depth model + depthmap/feature cache)
export DEPTH_CHECKPOINT=depth_anything_v2_vitl.pth # DAv2-Large (tag _vitl), must match at inference
export DINO_INPUT_SIZE=512 # ViT input res, 0 = auto = max(518,DEPTH_TARGET_SIZE); tag _dinoInSize<N>, keys the cache dir
export DEPTH_TARGET_SIZE=256 # depth predicted/fused at this res (finest Hiera FPN grid)
export USE_DEPTHFEATS=1 # 1 = multi-channel path_1 features, 0 = 1-ch scalar disparity
export CACHE_DEPTHFEATS=1 # 1 = read path_1 from disk, also forces USE_DEPTHMAPS=1
export DEPTH_PCA_CHANNELS=8 # PCA-k of the cached features, -1 = raw; keys the cache dir + _pdf<k> tag
export USE_DEPTHMAPS=1 # 1-ch depthmap cache as the image's 4th channel
# MCAC ships no depth maps: the first depth run generates the cache into the
# writable dirs below (_pdmgen), later runs reuse it. pelhanj/MCAC is read-only,
# the cache is keyed by split-prefixed folder id (folder ids repeat across splits).
export USE_AVAILABLE_DEPTHMAPS=0 # 0 = generate missing once (_pdmgen), 1 = require the maps exist (_pdm)
export USE_AVAILABLE_DEPTHFEATS=0 # 0 = build the feature cache if absent, 1 = require it
export DEPTHMAPS_DIR=/d/hpc/home/er52565/GECO2/MCACDataset/depthmaps # 1-ch depthmap cache (writable, not under pelhanj/MCAC)
# DEPTHFEATS_DIR is shell-derived; unset clears stale pins
unset DEPTHFEATS_DIR


# depth adaptation (modes 1/2/5 only)
export DEPTH_ADAPT=conv # conv = _ConvDepthAdapter, linear = plain Conv2d(C_dec,feat,1)
export DEPTH_CUES=learned # learned = adapter learns its own cues, fixed = precomputed Sobel/Laplacian
export DEPTH_ADAPT_INIT=orthogonal # orthogonal | default (tag _oinit)
export DEPTH_ADAPT_MASKED_CONV=1 # valid-pixel renorm so the letterbox pad doesn't bleed into edges
# adapter output width: defaults to the PCA input width (raw -1 falls back to 16).
# evaluated at source time, so overriding DEPTH_PCA_CHANNELS on the command line
# needs a matching DEPTH_FEAT_CHANNELS export. tag _dfc<N>.
export DEPTH_FEAT_CHANNELS=$(( DEPTH_PCA_CHANNELS > 0 ? DEPTH_PCA_CHANNELS : 16 ))
export DEPTH_FEAT_NORM=group # group | none (masked GroupNorm on the depth features)
export DEPTH_FEAT_NORM_GROUPS=0 # 0 = auto num_groups


# depth fusion
# DEPTH_FUSE_LR is set in the training section
export DEPTH_KERNEL_SIZE=1 # fusion conv kernel (modes 1/2 + the ffm concat conv); for a legacy untagged ffm ckpt export it empty (= paper 3x3)
export DEPTH_FUSE_IDENTITY_INIT=1 # depth starts at 0 contribution and grows (tag _idinit)
export DEPTH_HIRES_FUSION=1 # 1 = also inject full-res depth at the RGB input (modes 1/2, tag _hires); backbone runs with grad, batch auto-drops
export DEPTH_HIRES_NORM=1 # masked GroupNorm on that hires input (tag _hiresn)
# sep_hiera (mode 4) only:
export SEP_HIERA_INPUT=replicate # replicate = grayscale to all 3ch, cues = [disparity, Sobel, Laplacian] (_sepcues)
export SEP_HIERA_FULLRES=1 # build the 2nd-Hiera cues/adapter/GN at image_size (tag _sfr)
export SEP_HIERA_PER_LEVEL_GATE=1 # 3 per-level gamma gates (tag _plg, wants identity-init)
# ffm (mode 5) only:
export FFM_NORM=group # group | batch (batch = paper-exact BiSeNet, tag _ffmbn)


# backbone
export UNFREEZE_LAST_HIERA=0 # >0 = train the last N RGB-Hiera stages (tag _ufh<N>)
export BACKBONE_LR=0 # LR for those stages (0 = auto 1e-6 when unfreezing)


# training
export LR=5e-6 # body peak LR, kept low for the warm-started body
export ZS_PROTO_LR=1e-4 # separate higher LR for the zs prototype tokens (0 = follow LR); unused with MODE=few
export DEPTH_FUSE_LR=5e-5 # separate higher LR for the fresh fusion/adapter params (0 = follow LR)
export WEIGHT_DECAY=5e-5 # AdamW weight decay
export DROPOUT=0.1 # transformer dropout (official recipe)
export MAX_GRAD_NORM=0.1 # grad clip (official recipe)
export AUX_WEIGHT=0.3 # aux loss weight, per GECO2 spec
export TRAIN_TILING_P=0.5 # train-time tiling-aug probability (official recipe)
# no run-name tag on these or on LR_WARMUP_*/SPIKE_*; use the job id / [config-dump]
# to tell same-named runs apart (CKPT_JOB=<jobid> pins the ckpt for a re-eval)
export LR_WARMUP_EPOCHS=5 # linear warmup (FSCD 3ep@0.1, IOC/MCAC 5ep@0.05)
export LR_WARMUP_START_FACTOR=0.05 # epoch-1 LR multiplier
export AUX_LR_WARMUP_EPOCHS=10 # extra warmup on the aux LR group (tag _aw<N>)
export SEED=42 # tag _s<seed> when != 42
export NUM_WORKERS=12 # dataloader workers
# export BATCH_SIZE=12 # unset -> shell picks 12, or 6 for grad-through-Hiera; TRAIN_BATCH_SIZE wins if set
export PLATEAU_PATIENCE=12 # stop after this many epochs with no val improvement
export REDUCE_LR_PATIENCE=3 # halve LR after ~patience+1 stalled epochs
export REDUCE_LR_FACTOR=0.5 # LR multiplier per reduce step
export SPIKE_PATIENCE=2 # stop after this many consecutive val-RMSE spikes
export SPIKE_RATIO=2.0 # spike = raw val RMSE > ratio x best-so-far
export SELECT_RMSE_WEIGHT=0.0 # 0.0 = pick checkpoint/LR on pure val MAE, >0 blends in RMSE
export GOOD_SELECT=0 # 0 = select on the val metric, 1 = extra "good" plateau gate
export GOOD_PLATEAU_PATIENCE=2 # patience for the GOOD_SELECT gate
export PRETRAINED_INIT=/d/hpc/home/er52565/GECO2/CNTQG_multitrain_ca44.pth # warm-start ckpt (generalist prior)
export ALLOW_FROM_SCRATCH=0 # 1 only for a from-scratch run


# probe (depth warm-up)
export PROBE_ADAPTIVE=1
export PROBE_EPOCHS=0 # 0 = no fixed probe
export PROBE_PLATEAU_PATIENCE=5 # (adaptive only) val-stall epochs before releasing the body
export PROBE_LR_PATIENCE=2 # (adaptive only) stalled epochs before each depth-fuse LR halving
export PROBE_ONLY=1 # stop training when the probe ends (tag suffix "only")


# density head (on for MCAC, secondary density-integral count)
export USE_DENSITY=1 # 1 = add the DensityHead and report the density-integral count too
export DENSITY_LOSS_TYPE=dmcount # dmcount (DM-Count OT) | mse
export DENSITY_HEAD_TYPE=simple # simple | fpn
export DENSITY_WEIGHT=1.0 # density loss weight
export DENSITY_ABS_COUNT_WEIGHT=0.5 # absolute-count term weight
export DENSITY_LR=1e-4 # density-head LR group
export DENSITY_ADAPTIVE_SIGMA=0 # 1 = per-point sigma from kNN spacing
export DENSITY_SIGMA=8.0 # fixed Gaussian sigma (px) when adaptive is off
export DENSITY_SIGMA_MIN=2.0 # adaptive sigma floor
export DENSITY_SIGMA_MAX=15.0 # adaptive sigma ceiling
export DENSITY_SIGMA_K=3 # kNN k for adaptive sigma
export DENSITY_SIGMA_BETA=0.3 # adaptive sigma spacing scale
export DENSITY_GUIDED=0 # 1 = density-guided detection (tag _densg)
export DENSITY_COUNT_SOURCE=whole # never read on MCAC, whole inference always integrates the full map


# inference / evaluation
export WHOLE_SWEEP=0 # 0 = both splits use GeCo2's relative box_v.max()*0.11, 1 = calibrate an abs threshold on val
export INFER_BATCH_SIZE=2 # whole-inference batch size
export REPORT_MAX_GT= # GT-count strata for stratified metrics (empty = off)
export EVAL_SPLITS= # override which splits get evaluated (empty = test+val)


# tiled training / sweep: MCAC has no tiled path
# export CROP_P=0.5
# export CROP_MIN_PX=384
# export CROP_MAX_PX=640
# export SWEEP_IMAGES=150


# mcac dataset conventions
# frame/occlusion knobs are baked into MCACDataset defaults (center-crop 672 from
# 1080, keep instances with occlusion < 70, exemplars from the shipped
# <split>_eval_bboxes.json auto-discovered under DATA)
export MCAC_DATA_ROOT=/d/hpc/projects/FRI/pelhanj/MCAC # read-only source (img.png, info json, {val,test}_eval_bboxes.json)
