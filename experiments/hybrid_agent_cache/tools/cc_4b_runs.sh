#!/bin/bash
# Qwen3.5-4B on one RTX 3090: 503 blocks x 576 = 281,748 tokens at --gpu-memory-utilization 0.85.
# 2 sessions (~0.57x capacity, control like the 3060 4-session run) and 5 sessions
# (~1.38x, like the 3060 8-session run at 1.44x), each on main and with #61282.
# Needs CUDA_DEVICE_ORDER/CUDA_VISIBLE_DEVICES/PORT from the caller. Do not edit vllm/ while this runs.
S=$(dirname "$(readlink -f "$0")")
P=$S/../patches/pr61282_rebased_ported.diff
cd "$S/../../.."
export MODEL=${MODEL:-Qwen/Qwen3.5-4B}
E="--gpu-memory-utilization ${GPU_UTIL:-0.85}"
applied=0
trap '[ $applied = 1 ] && git apply -R "$P" && echo "[revert 61282 on exit] ok"' EXIT
TAG=4b_n2_main EXTRA="$E" "$S/cc_run.sh" --sessions 2 --max-req 40
TAG=4b_n5_main EXTRA="$E" "$S/cc_run.sh" --sessions 5 --max-req 40
git apply "$P" && applied=1 && echo "[apply 61282] ok" || { echo "[apply 61282] FAILED"; exit 1; }
TAG=4b_n5_61282 EXTRA="$E" "$S/cc_run.sh" --sessions 5 --max-req 40
TAG=4b_n2_61282 EXTRA="$E" "$S/cc_run.sh" --sessions 2 --max-req 40
git apply -R "$P" && applied=0 && echo "[revert 61282] ok"
echo ALL_DONE
