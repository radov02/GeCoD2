#!/usr/bin/env bash
# config_MCAC_ddet.sh - density head detached from the trunk (tag _ddet):
# --density_detach 1 stops the DM-Count loss backpropagating into the counter.
# otherwise config_MCAC_densg.sh with the body free to unfreeze.
source "$(dirname "${BASH_SOURCE[0]}")/config_MCAC_densg.sh"

export DENSITY_DETACH=1 # cut the density gradient at the trunk (tag _ddet)
export PROBE_ONLY=0 # let the body unfreeze
