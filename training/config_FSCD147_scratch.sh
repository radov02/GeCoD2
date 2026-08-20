#!/usr/bin/env bash
# config_FSCD147_scratch.sh - FSCD-147 from scratch, paper recipe.
# every reported FSCD number starts from the converged GECO2FSCD.pth; this run
# uses random init instead, so depth trains together with the counter from
# epoch 1. Hiera stays frozen (pretrained SAM 2), only the counter is random.
source "$(dirname "${BASH_SOURCE[0]}")/config_FSCD147_paper.sh"

export PRETRAINED_INIT="" # no warm start
export ALLOW_FROM_SCRATCH=1 # required, else the shell aborts
export DEPTH_FUSE_IDENTITY_INIT=0 # nothing pretrained to protect, depth contributes from epoch 1
export BATCH_SIZE=6 # not the paper's 8: the depth arms run grad-through-Hiera at 1024^2
export TRAIN_BATCH_SIZE=6 # same batch for all scratch arms
export SPIKE_RATIO=3.0 # from-scratch val RMSE is noisy in the first epochs
export SPIKE_PATIENCE=8 # loosen the spike stop so it doesn't false-trigger
