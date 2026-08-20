#!/usr/bin/env bash
# config_MCAC_densg.sh - config_MCAC.sh with density-guided detection on
# (DENSITY_GUIDED=1, tag _densg). pass it as CONFIG_FILE; the base exports
# DENSITY_GUIDED=0, so a 'DENSITY_GUIDED=1 sbatch' prefix gets overwritten.
# densg runs are compared against a geco2_finetuned densg control, not against
# the plain-detection rows.
source "$(dirname "${BASH_SOURCE[0]}")/config_MCAC.sh"
export DENSITY_GUIDED=1
