#!/usr/bin/env bash
# config_MCAC_densg_joint.sh - density-guided detection + joint finetune
# (DENSITY_GUIDED=1, PROBE_ONLY=0). config_MCAC.sh exports PROBE_ONLY=1, which
# would otherwise turn a densg depth run into probe-only (probeAonly).
# for densg depth variants; the densg baseline (geco2_finetuned + noprobe) is
# unaffected, noprobe already forces the probe off.
source "$(dirname "${BASH_SOURCE[0]}")/config_MCAC_densg.sh"
export PROBE_ONLY=0
