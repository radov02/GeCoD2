#!/usr/bin/env bash
# config_IOCfish_densg_probeA.sh - config_IOCfish_densg.sh + adaptive probe
# (PROBE_ADAPTIVE=1, tag _probeA). the base config sets PROBE_ADAPTIVE=0 and the
# 9th positional 'probe' only keeps the config knobs, so probeA needs a derived config.
# for densg depth rows only; the geco2_finetuned densg control stays on
# config_IOCfish_densg.sh with '0 noprobe' (nothing to probe there).
source "$(dirname "${BASH_SOURCE[0]}")/config_IOCfish_densg.sh"
export PROBE_ADAPTIVE=1
