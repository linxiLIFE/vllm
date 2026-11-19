#!/bin/bash
# Usage: TAG=name EXTRA="serve flags" cc_run.sh [replay flags...]
S=$(dirname "$(readlink -f "$0")")
PORT=${PORT:-8000}
curl -s -o /dev/null localhost:$PORT && { echo "port $PORT already in use; set PORT"; exit 1; }
O=${OUT:-$S/../results/cctrace}; mkdir -p "$O"
TRACES=${TRACES:-${DATA:-$HOME/study/data}/cc-traces-weka-062126/traces.jsonl}
cd "$S/../../.."
tag=cc_${TAG:-default}
VLLM_CACHE_PROBE_EVICT_SUPERSEDED=${DEMOTE:-0} VLLM_CACHE_PROBE=$O/probe_$tag.jsonl \
  .venv/bin/vllm serve ${MODEL:-Qwen/Qwen3.5-0.8B} --port $PORT --max-model-len 131072 \
  --enable-prompt-tokens-details --kv-cache-metrics --kv-cache-metrics-sample 1.0 ${EXTRA:-} \
  > $O/serve_$tag.log 2>&1 &
pid=$!
for i in $(seq 1 120); do
  curl -sf localhost:$PORT/health >/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 3
done
curl -sf localhost:$PORT/health >/dev/null || { echo "[$tag] server failed"; tail -5 $O/serve_$tag.log; kill $pid 2>/dev/null; wait $pid 2>/dev/null; exit 1; }
curl -s localhost:$PORT/metrics > $O/m_${tag}_before.txt
.venv/bin/python -I $S/cc_replay.py "$TRACES" $O/res_$tag.json --url http://localhost:$PORT --model "${MODEL:-Qwen/Qwen3.5-0.8B}" "$@" > $O/replay_$tag.log 2>&1
echo "[$tag] replay exit $?"
curl -s localhost:$PORT/metrics > $O/m_${tag}_after.txt
kill -INT $pid; wait $pid 2>/dev/null
echo "[$tag] done"
