#!/usr/bin/env bash
# config_FSCD147_paper.sh - GeCo2 paper training recipe on FSCD-147.
# our finetune of the released GECO2FSCD.pth degrades it; this run applies the
# paper recipe (LR 1e-4, batch 8) on geco2_finetuned (RGB, no depth) to check
# whether the recipe is the reason.
source "$(dirname "${BASH_SOURCE[0]}")/config_FSCD147.sh"

export LR=1e-4 # paper initial LR (ours is 5e-6)
export DEPTH_FUSE_LR=1e-4
export BATCH_SIZE=8 # paper batch size
export TRAIN_BATCH_SIZE=8
export PROBE_ADAPTIVE=0 # no probe, body trains from epoch 1
export PROBE_EPOCHS=0
export PROBE_ONLY=0
export USE_DENSITY=0 # already 0 in config_FSCD147.sh, pinned explicitly
export WEIGHT_DECAY=5e-5 # paper value, config_FSCD147.sh has 1e-5
export PLATEAU_PATIENCE=25 # see config_MCAC_fairbase.sh
export REDUCE_LR_PATIENCE=15 # hold the LR near 1e-4; upstream GeCo2 uses StepLR x0.1 at epoch 50
export SPIKE_PATIENCE=5
