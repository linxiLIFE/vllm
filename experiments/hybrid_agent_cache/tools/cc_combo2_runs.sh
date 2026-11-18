#!/bin/bash
# Lookup-logged replays with PR #66881 (experimental port, GDS) + PR #61282
# (boundary hold and attention-first free order) stacked; see README.
# Do not edit vllm/ while this runs.
S=$(dirname "$(readlink -f "$0")")
GDS="--mamba-frontier-checkpoint --prefix-cache-eviction gds --prefill-cost-coefficients 6.14e-5 2.44e-4"
TAG=n8_66881_61282 EXTRA="$GDS" "$S/cc_run.sh" --sessions 8 --max-req 40
TAG=n4_66881_61282 EXTRA="$GDS" "$S/cc_run.sh" --sessions 4 --max-req 40
echo ALL_DONE
