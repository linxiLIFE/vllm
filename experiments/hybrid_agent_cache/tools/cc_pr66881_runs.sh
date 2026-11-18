#!/bin/bash
# Lookup-logged replays with patches/pr66881_ported_experimental.diff applied:
# frontier checkpoints + GreedyDual-Size eviction. Cost coefficients fitted on
# this 3060 with `vllm bench forward` (results/cctrace/prefill_cost_fit.json):
# D = 6.14e-5 s/token, E = 2.44e-4 s/token/100k.
# Do not edit vllm/ while this runs.
S=$(dirname "$(readlink -f "$0")")
GDS="--mamba-frontier-checkpoint --prefix-cache-eviction gds --prefill-cost-coefficients 6.14e-5 2.44e-4"
TAG=n8_pr66881 EXTRA="$GDS" "$S/cc_run.sh" --sessions 8 --max-req 40
TAG=n4_pr66881 EXTRA="$GDS" "$S/cc_run.sh" --sessions 4 --max-req 40
echo ALL_DONE
