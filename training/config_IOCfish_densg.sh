#!/usr/bin/env bash
# config_IOCfish_densg.sh - config_IOCfish.sh with density-guided detection on
# (DENSITY_GUIDED=1, tag _densg). pass it as CONFIG_FILE; the base config exports
# DENSITY_GUIDED=0, so an env prefix can't flip it.
# densg runs are compared against a geco2_finetuned densg control, not against
# the plain-detection rows.
source "$(dirname "${BASH_SOURCE[0]}")/config_IOCfish.sh"
export DENSITY_GUIDED=1
