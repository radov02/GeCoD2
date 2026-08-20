#!/usr/bin/env bash
# config_MCAC_scratch.sh - MCAC from scratch, paper recipe.
# warm-start MCAC runs only perturb the already-converged multitrain init; this
# trains the counter from random weights instead (Hiera stays frozen, pretrained
# SAM 2), so depth gets to shape the representation from epoch 1. absolute MAE
# sits far above the warm-start runs, what matters is the gap between the scratch runs.
# the hpc shell expands PRETRAINED_INIT with '-' not ':-', so the empty value survives.
source "$(dirname "${BASH_SOURCE[0]}")/config_MCAC_fairbase.sh"

export PRETRAINED_INIT="" # no warm start
export ALLOW_FROM_SCRATCH=1 # required, else the shell aborts
export DEPTH_FUSE_IDENTITY_INIT=0 # nothing pretrained to protect, depth contributes from epoch 1
export BATCH_SIZE=6 # conv_hiera runs grad-through-Hiera (hires), same batch for both arms
export TRAIN_BATCH_SIZE=6
export SPIKE_RATIO=3.0 # from-scratch val RMSE is noisy in the first epochs
export SPIKE_PATIENCE=8 # loosen the spike stop so it doesn't false-trigger
