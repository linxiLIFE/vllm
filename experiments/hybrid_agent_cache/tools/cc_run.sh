#!/bin/bash
# Usage: TAG=name EXTRA="serve flags" cc_run.sh [replay flags...]
S=$(dirname "$(readlink -f "$0")")
O=${OUT:-$S/../results/cctrace}; mkdir -p "$O"
TRACES=${TRACES:-/home/linxi/study/data/cc-traces-weka-062126/traces.jsonl}
cd /home/linxi/study/vllm
tag=cc_${TAG:-default}
VLLM_CACHE_PROBE_EVICT_SUPERSEDED=${DEMOTE:-0} VLLM_CACHE_PROBE=$O/probe_$tag.jsonl \
  .venv/bin/vllm serve Qwen/Qwen3.5-0.8B --port 8000 --max-model-len 131072 \
  --enable-prompt-tokens-details --kv-cache-metrics --kv-cache-metrics-sample 1.0 ${EXTRA:-} \
  > $O/serve_$tag.log 2>&1 &
pid=$!
for i in $(seq 1 120); do
  curl -sf localhost:8000/health >/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 3
done
curl -sf localhost:8000/health >/dev/null || { echo "[$tag] server failed"; tail -5 $O/serve_$tag.log; exit 1; }
curl -s localhost:8000/metrics > $O/m_${tag}_before.txt
.venv/bin/python -I $S/cc_replay.py "$TRACES" $O/res_$tag.json "$@" > $O/replay_$tag.log 2>&1
echo "[$tag] replay exit $?"
curl -s localhost:8000/metrics > $O/m_${tag}_after.txt
kill -INT $pid; wait $pid 2>/dev/null
echo "[$tag] done"
