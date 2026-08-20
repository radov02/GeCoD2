#!/usr/bin/env bash
# config_MCAC_fairbase.sh - GeCo2 paper recipe in our harness on MCAC (paper
# hyper-params, no density head, body trainable from epoch 1, early stop
# loosened). the published GeCo2 MCAC row is trained for MCAC, ours only adapts
# the generalist CNTQG_multitrain_ca44.pth, so this is a reference point.
source "$(dirname "${BASH_SOURCE[0]}")/config_MCAC.sh"

export USE_DENSITY=0 # paper GeCo2 has no density branch
export DENSITY_GUIDED=0
export PROBE_ADAPTIVE=0 # no probe, the body trains from epoch 1
export PROBE_EPOCHS=0
export PROBE_ONLY=0 # and is never re-frozen
export LR=1e-4 # paper initial LR (ours is 5e-6)
export DEPTH_FUSE_LR=1e-4 # keep the fusion group at the same LR (no depth here anyway)
export BATCH_SIZE=8 # paper batch size (ours: 12 / 6)
export TRAIN_BATCH_SIZE=8
export REDUCE_LR_PATIENCE=15 # hold the LR near 1e-4; upstream GeCo2 uses StepLR x0.1 at epoch 50
export PLATEAU_PATIENCE=25 # paper trains a fixed 200 epochs; 25 keeps a net without cutting early
export SPIKE_PATIENCE=5 # LR 1e-4 on a warm body spikes early, don't stop on it
