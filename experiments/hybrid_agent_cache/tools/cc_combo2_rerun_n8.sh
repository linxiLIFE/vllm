#!/bin/bash
# Rerun the 8-session 66881+61282 replay after cc_combo2_runs.sh finishes
# (the first attempt hit an aiohttp keep-alive race, fixed in cc_replay.py).
S=$(dirname "$(readlink -f "$0")")
OUT=$S/../results/cctrace
until grep -q ALL_DONE "$OUT/cc_combo2_runs.out"; do sleep 5; done
mv "$OUT/replay_cc_n8_66881_61282.log" "$OUT/replay_cc_n8_66881_61282.aborted.log"
GDS="--mamba-frontier-checkpoint --prefix-cache-eviction gds --prefill-cost-coefficients 6.14e-5 2.44e-4"
TAG=n8_66881_61282 EXTRA="$GDS" "$S/cc_run.sh" --sessions 8 --max-req 40
echo ALL_DONE
