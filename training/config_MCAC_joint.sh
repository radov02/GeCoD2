#!/usr/bin/env bash
# config_MCAC_joint.sh - config_MCAC.sh with PROBE_ONLY=0 (joint finetune: the
# probe releases the body after the val stall instead of stopping the run).
# config_MCAC.sh exports PROBE_ONLY=1 and overwrites any env prefix, so pass this
# file as CONFIG_FILE instead of editing the base config back and forth.
source "$(dirname "${BASH_SOURCE[0]}")/config_MCAC.sh"
export PROBE_ONLY=0
