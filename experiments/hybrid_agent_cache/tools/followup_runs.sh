#!/bin/bash
# Follow-ups to the memory-saving test, all with FP8 KV + bf16 GDN state:
#   1) 8 sessions WITHOUT #61282 (is the fix still needed once capacity fits?)
#   2) 12 sessions with #61282, 3) 12 sessions without #61282.
# Starts from the probe-only tree; applies #61282 only for run 2 and reverts it after.
# Do not edit vllm/ while this runs.
S=$(dirname "$(readlink -f "$0")")
P=$S/../patches
cd "$S/../../.."
BOTH="--kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16"
TAG=n8_main_both EXTRA="$BOTH" "$S/cc_run.sh" --sessions 8 --max-req 40
git apply "$P/pr61282_rebased_ported.diff" && echo "[apply 61282] ok" || { echo "[apply 61282] FAILED"; exit 1; }
TAG=n12_61282_both EXTRA="$BOTH" "$S/cc_run.sh" --sessions 12 --max-req 40
git apply -R "$P/pr61282_rebased_ported.diff" && echo "[revert 61282] ok" || { echo "[revert 61282] FAILED"; exit 1; }
TAG=n12_main_both EXTRA="$BOTH" "$S/cc_run.sh" --sessions 12 --max-req 40
echo ALL_DONE
