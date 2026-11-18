#!/bin/bash
# Memory-saving configs on top of PR #61282 (patches/pr61282_rebased_ported.diff applied):
# 1) GSM8K quality check per config, 2) 8- and 4-session trace replays per config.
# Do not edit vllm/ while this runs.
S=$(dirname "$(readlink -f "$0")")
PORT=${PORT:-8000}
curl -s -o /dev/null localhost:$PORT && { echo "port $PORT already in use; set PORT"; exit 1; }
O=$S/../results/cctrace
cd "$S/../../.."
declare -A FLAGS=(
  [base]=""
  [fp8kv]="--kv-cache-dtype fp8"
  [bf16ssm]="--mamba-ssm-cache-dtype bfloat16"
  [both]="--kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16"
)
for cfg in base fp8kv bf16ssm both; do
  .venv/bin/vllm serve Qwen/Qwen3.5-0.8B --port $PORT --max-model-len 8192 ${FLAGS[$cfg]} \
    > $O/serve_gsm8k_$cfg.log 2>&1 &
  pid=$!
  for i in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 3; done
  .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py --port $PORT --num-shots 5 --temperature 0 --seed 0 \
    --save-results $O/gsm8k_$cfg.json > $O/gsm8k_$cfg.log 2>&1
  echo "[gsm8k_$cfg] exit $?"
  kill -INT $pid; wait $pid 2>/dev/null
done
for cfg in fp8kv bf16ssm both; do
  TAG=n8_61282_$cfg EXTRA="${FLAGS[$cfg]}" "$S/cc_run.sh" --sessions 8 --max-req 40
  TAG=n4_61282_$cfg EXTRA="${FLAGS[$cfg]}" "$S/cc_run.sh" --sessions 4 --max-req 40
done
echo ALL_DONE
