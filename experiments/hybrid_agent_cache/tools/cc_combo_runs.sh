#!/bin/bash
# Lookup-logged replays with patches/combo_63851_plus_61282hold.diff applied:
# PR #63851 (position-ordered free, drop unreachable tail) + PR #61282's
# MambaManager boundary hold, with the held state put back at its own slot.
# Do not edit vllm/ while this runs.
S=$(dirname "$(readlink -f "$0")")
TAG=n8_combo "$S/cc_run.sh" --sessions 8 --max-req 40
TAG=n4_combo "$S/cc_run.sh" --sessions 4 --max-req 40
echo ALL_DONE
