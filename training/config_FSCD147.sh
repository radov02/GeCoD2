#!/usr/bin/env bash
# config_FSCD147.sh - FSCD147 depth-fusion train+inference knobs.
# same layout as config_IOCfish.sh: fixed-in-shell reference block first, then the
# env knobs. run selection is positional, the rest the shell reads as ${VAR:-default}.
# FSCD notes: box-count is reported vs GeCo2 (USE_DENSITY=0, WHOLE_SWEEP=0); the
# init is a converged GeCo2 body, keep LR low and let DEPTH_FUSE_LR carry the
# fresh params. COCO AP runs on test+val. no backbone-unfreeze knobs here.
#
# usage: source then submit (env propagates via sbatch --export=ALL):
#   cd ~/GECO2
#   source training/config_FSCD147.sh
#   sbatch training/hpc_H100_train_and_inference_FSCD147_dataset.sh \
#          "$RUN_STAGES" "$EXPERIMENT" "$REGIME" "$EPOCHS" "$TRAIN_SCALE" "$INFER_TYPE" "$TEST_SPLIT" "$VIS_EVERY" "$SWEEP_MODE" "$PROBE_ARG"
# or let the job source it (still pass the positionals):
#   cd ~/GECO2
#   CONFIG_FILE=$PWD/training/config_FSCD147.sh \
#     sbatch training/hpc_H100_train_and_inference_FSCD147_dataset.sh both conv_depth_add all 200 whole whole test 75
# FSCD's positional order slots TEST_SPLIT in at 6 (before vis).
# env prefixes on a 'CONFIG_FILE=... sbatch' line don't survive this file's exports.


# fixed in the shell (reference only, plain assignments there; exporting them here
# does nothing). arch flags match the warm-start ckpt (GECO2FSCD.pth) and must stay
# identical between train and inference, changing any of them breaks ckpt loading:
#   BACKBONE=SAM  REDUCTION=16  IMAGE_SIZE=1024  NUM_ENC_LAYERS=3  EMB_DIM=256
#   NUM_HEADS=8  NUM_OBJECTS=3
#   (KERNEL_DIM is a real knob below; counter.py forces it to 1, any value loads any ckpt)
# tiled-protocol constants, calibrated by the val sweep rather than hand-tuned:
#   TILE_SIZE=256  TILE_OVERLAP=64  TILE_BATCH_SIZE=8  NMS_IOU=0.5  SCORE_ABS_THR=0.1
#   SCORE_REL_THR=0.0  EDGE_MARGIN=8  WHOLE_IMAGE_PASS=1  INFER_TILING_P=0.0


# run selection (positional args, pass on the sbatch line)
RUN_STAGES=both # both | train | inference
EXPERIMENT=conv_depth_add # conv_depth_add | conv_hiera | depth_dim | sep_hiera | ffm | geco2 | geco2_pretrained (inference-only)
REGIME=all # all | lt1500 (train subset with <1500 objects/img)
EPOCHS=200 # epoch budget, the plateau stop usually ends it earlier
TRAIN_SCALE=whole # whole | tiled
INFER_TYPE=whole # whole | tiled
TEST_SPLIT=test # split of the final pass (FSCD-only positional, slot 6)
VIS_EVERY=75 # save a viz every Nth inference image (0=off)
SWEEP_MODE=0 # tiled-only sweep: 0=off, 1=sweep-only, 2=sweep-then-test
PROBE_ARG= # empty = keep the PROBE_* knobs, noprobe = force them off (for geco2_finetuned)


# depth supply (depth model + depthmap/feature cache)
export DEPTH_CHECKPOINT=depth_anything_v2_vitl.pth # DAv2-Large (tag _vitl), must match at inference
export DINO_INPUT_SIZE=512 # ViT input res, 0 = auto = max(518,DEPTH_TARGET_SIZE); tag _dinoInSize<N>, keys the cache dir
export DEPTH_TARGET_SIZE=256 # depth predicted/fused at this res (finest Hiera FPN grid)
export USE_DEPTHFEATS=1 # 1 = multi-channel path_1 features, 0 = 1-ch scalar disparity
export CACHE_DEPTHFEATS=1 # 1 = read path_1 from disk, also forces USE_DEPTHMAPS=1
export DEPTH_PCA_CHANNELS=8 # PCA-k of the cached features, -1 = raw; keys the cache dir + _pdf<k> tag
export USE_DEPTHMAPS=1 # 1-ch depthmap cache as the image's 4th channel
export USE_AVAILABLE_DEPTHMAPS=1 # 1 = require the maps exist (_pdm), 0 = generate missing once (_pdmgen)
export USE_AVAILABLE_DEPTHFEATS=0 # 0 = build the feature cache if absent, 1 = require it
export DEPTHMAPS_DIR=/d/hpc/home/er52565/GECO2/FSCD147Dataset/depthmaps # 1-ch depthmap cache
# DEPTHFEATS_DIR is shell-derived from DEPTH_PCA_CHANNELS + DINO_INPUT_SIZE; unset
# clears stale pins (pin one by exporting it after sourcing)
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
export DEPTH_HIRES_FUSION=1 # 1 = also inject full-res depth at the RGB input (modes 1/2, tag _hires)
export DEPTH_HIRES_NORM=1 # masked GroupNorm on that hires input (tag _hiresn)
# sep_hiera (mode 4) only:
export SEP_HIERA_INPUT=replicate # replicate = grayscale to all 3ch, cues = [disparity, Sobel, Laplacian] (_sepcues)
export SEP_HIERA_FULLRES=1 # build the 2nd-Hiera cues/adapter/GN at image_size (tag _sfr)
export SEP_HIERA_PER_LEVEL_GATE=1 # 3 per-level gamma gates (tag _plg, wants identity-init)
# ffm (mode 5) only:
export FFM_NORM=group # group | batch (batch = paper-exact BiSeNet, tag _ffmbn)


# model
export KERNEL_DIM=1 # GeCo2 counting-kernel dim (stock value)
# no backbone-unfreeze knobs here, backbone stays frozen


# training
export LR=5e-6 # body peak LR, kept well below DEPTH_FUSE_LR (converged GeCo2 init)
export DEPTH_FUSE_LR=5e-5 # separate higher LR for the fresh fusion/adapter params (0 = follow LR)
export WEIGHT_DECAY=1e-5 # AdamW weight decay (FSCD 1e-5, IOC 5e-5)
export DROPOUT=0.1 # transformer dropout (official recipe)
export MAX_GRAD_NORM=0.1 # grad clip (official recipe)
export AUX_WEIGHT=0.3 # aux loss weight, per GECO2 spec
export TRAIN_TILING_P=0.5 # train-time tiling-aug probability (official recipe)
# no run-name tag on these or on LR_WARMUP_*/SPIKE_*; use the job id / [config-dump]
# to tell same-named runs apart (CKPT_JOB=<jobid> pins the ckpt for a re-eval)
export LR_WARMUP_EPOCHS=3 # linear warmup (FSCD 3ep@0.1, IOC/MCAC 5ep@0.05)
export LR_WARMUP_START_FACTOR=0.1 # epoch-1 LR multiplier
export AUX_LR_WARMUP_EPOCHS=0 # extra warmup on the aux LR group (tag _aw<N>)
export SEED=42 # tag _s<seed> when != 42
export NUM_WORKERS=12 # dataloader workers
# export BATCH_SIZE=4 # unset -> shell uses 4 @1024px; TRAIN_BATCH_SIZE wins if set
export PLATEAU_PATIENCE=10 # stop after this many epochs with no val improvement
export REDUCE_LR_PATIENCE=3 # halve LR after ~patience+1 stalled epochs
export REDUCE_LR_FACTOR=0.5 # LR multiplier per reduce step
export SPIKE_PATIENCE=2 # stop after this many consecutive val-RMSE spikes
export SPIKE_RATIO=2.0 # spike = raw val RMSE > ratio x best-so-far
export SELECT_RMSE_WEIGHT=0.0 # 0.0 = pick checkpoint/LR on pure val MAE, >0 blends in RMSE
export GOOD_SELECT=0 # 0 = select on the val metric, 1 = extra "good" plateau gate
export GOOD_PLATEAU_PATIENCE=2 # patience for the GOOD_SELECT gate
export PRETRAINED_INIT=/d/hpc/home/er52565/GECO2/GECO2FSCD.pth # FSCD warm-start ckpt
export ALLOW_FROM_SCRATCH=0 # 1 only for a from-scratch run


# probe (depth warm-up)
# finetuning the converged GECO2FSCD.pth from epoch 1 degrades it, so the probe
# trains the fresh fusion with the body frozen first
export PROBE_ADAPTIVE=1 # 1 = adaptive probe (_probeA), runs until val stalls PROBE_PLATEAU_PATIENCE epochs
export PROBE_EPOCHS=10 # fixed probe length (min epochs when adaptive)
export PROBE_PLATEAU_PATIENCE=5 # (adaptive only) val-stall epochs before releasing the body
export PROBE_LR_PATIENCE=2 # (adaptive only) stalled epochs before each depth-fuse LR halving
export PROBE_ONLY=1 # stop training when the probe ends (tag suffix "only")


# density head (off for FSCD, headline is box-count vs GeCo2)
export USE_DENSITY=0 # 0 = box-count only, 1 adds the density head
export DENSITY_LOSS_TYPE=dmcount # dmcount | mse (only read when USE_DENSITY=1)
export DENSITY_HEAD_TYPE=simple # simple | fpn
export DENSITY_WEIGHT=1.0
export DENSITY_ABS_COUNT_WEIGHT=0.5
export DENSITY_LR=1e-4
export DENSITY_ADAPTIVE_SIGMA=0
export DENSITY_SIGMA=8.0
export DENSITY_SIGMA_MIN=2.0
export DENSITY_SIGMA_MAX=15.0
export DENSITY_SIGMA_K=3
export DENSITY_SIGMA_BETA=0.3
export DENSITY_GUIDED=0 # 1 = density-guided detection (tag _densg)


# inference / evaluation
export WHOLE_SWEEP=0 # off for FSCD (headline = raw box-count)
export UPSTREAM_EVAL=1 # geco2_pretrained only: 1 = also run stage 0 (original inference.py + eval_bboxes.py)
export INFER_BATCH_SIZE=16 # whole-inference batch size
export REPORT_MAX_GT=1500 # GT-count cap for the extra strata rows (>1500 = paper-excluded outliers)
export EVAL_SPLITS= # override which splits get evaluated (empty = test+val)
# COCO AP (FSCD-only):
export COCO_EVAL=1 # 1 = run COCO AP after counting
export COCO_VIS=0 # 1 = save COCO match visualizations
export COCO_VIS_EVERY=1 # save every Nth COCO-vis image
export COCO_VIS_IOU_THR=0.5 # IoU threshold for the TP/FP overlay
export COCO_VIS_MAX=0 # cap on COCO-vis images (0 = no cap)
export COCO_JSON_OUT=__AUTO__ # __AUTO__ = per-(split,type) path, "" = skip the dump


# tiled training / sweep (only when TRAIN_SCALE/INFER_TYPE=tiled or SWEEP_MODE>0)
# export CROP_P=0.5 # zoom-in crop-aug probability (auto 0 whole / 0.5 tiled)
# export CROP_MIN_PX=192 # min crop side in original px
# export CROP_MAX_PX=320 # max crop side
# export SWEEP_IMAGES=150 # val images the tiled sweep searches over
# export SWEEP_ABS=default # abs box_v grid override
# export SWEEP_NMS=default # NMS-IoU grid override
# export SWEEP_EDGE=default # edge-margin grid override
