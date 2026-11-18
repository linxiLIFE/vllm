#!/bin/bash
# Start vllm serve with extra flags, report block size / capacity / backend, stop.
# Usage: probe_startup.sh <label> [serve flags...]
label=$1; shift
S=$(dirname "$(readlink -f "$0")")
PORT=${PORT:-8000}
curl -s -o /dev/null localhost:$PORT && { echo "port $PORT already in use; set PORT"; exit 1; }
L=$S/../results/cctrace/startup_$label.log
cd "$S/../../.."
.venv/bin/vllm serve Qwen/Qwen3.5-0.8B --port $PORT --max-model-len 131072 "$@" > "$L" 2>&1 &
pid=$!
for i in $(seq 1 120); do
  curl -sf localhost:$PORT/health >/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 3
done
echo "== $label: $*"
grep -E "attention backend|Setting attention block size|Padding mamba page|GPU KV cache size|Extended KV cache to|kv_cache_dtype|mamba_ssm_cache_dtype|ValueError|Error:" "$L" \
  | grep -v "non-default args" | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | cut -c1-200
curl -sf localhost:$PORT/health >/dev/null && echo "started OK" || echo "FAILED to start"
kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null
