#!/bin/bash
# Usage: sweep.sh <blocks>...   Restarts vLLM per block count and runs the multi-turn load.
S=$(dirname "$(readlink -f "$0")")
PORT=${PORT:-8000}
curl -s -o /dev/null localhost:$PORT && { echo "port $PORT already in use; set PORT"; exit 1; }
O=${OUT:-$S/../results/synthetic}; mkdir -p "$O"
cd "$S/../../.."
sed "s#@DATA@#${DATA:-$HOME/study/data}#" "$S/generate_multi_turn.json" > "$O/generate_multi_turn.json"
for nb in "$@"; do
  tag=nb$nb${TAG:-}
  VLLM_CACHE_PROBE_EVICT_SUPERSEDED=${DEMOTE:-0} VLLM_CACHE_PROBE=${PROBE:+$O/probe_$tag.jsonl} .venv/bin/vllm serve Qwen/Qwen3.5-0.8B --port $PORT --max-model-len 8192 --gpu-memory-utilization ${GPU_UTIL:-0.85} --max-num-seqs 16 \
    --num-gpu-blocks-override $nb --kv-cache-metrics --kv-cache-metrics-sample 1.0 ${EXTRA:-} \
    > $O/serve_$tag.log 2>&1 &
  pid=$!
  for i in $(seq 1 120); do
    curl -sf localhost:$PORT/health >/dev/null && break
    kill -0 $pid 2>/dev/null || break
    sleep 3
  done
  if ! curl -sf localhost:$PORT/health >/dev/null; then
    echo "[$tag] server failed to start"; tail -5 $O/serve_$tag.log; kill $pid 2>/dev/null; wait $pid; continue
  fi
  curl -s localhost:$PORT/metrics > $O/m_${tag}_before.txt
  .venv/bin/python benchmarks/multi_turn/benchmark_serving_multi_turn.py -m Qwen/Qwen3.5-0.8B \
    -u http://localhost:$PORT -i $O/generate_multi_turn.json -p 4 -k 8 --seed 0 \
    --stats-json-output $O/stats_$tag.json > $O/bench_$tag.log 2>&1
  echo "[$tag] bench exit $?"
  curl -s localhost:$PORT/metrics > $O/m_${tag}_after.txt
  kill -INT $pid; wait $pid 2>/dev/null
  for i in $(seq 1 30); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r ps -o user= -p | grep -qx "$USER" || break; sleep 2; done
done
echo sweep done
