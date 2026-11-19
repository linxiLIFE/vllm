#!/bin/bash
# Usage: lane_runner.sh <worktree-name> <gpu> <port>
# Runs jobs from ~/study/tmp/lanes/<lane>.q one by one; append lines to queue more.
#   replay line:  TAG SESSIONS [serve flags...]      -> cc_run.sh in the lane's worktree
#   gsm8k line:   GSM8K NAME [serve flags...]        -> gsm8k_eval.py against a fresh server
#   END                                              -> stop
lane=$1 gpu=$2 port=$3
W=$HOME/study/wt/$lane
Q=$HOME/study/tmp/lanes/$lane.q
LOG=$HOME/study/tmp/lanes/$lane.log
R=$HOME/study/vllm/experiments/hybrid_agent_cache/results/cctrace
source $HOME/study/tmp/run_env.sh
export CUDA_VISIBLE_DEVICES=$gpu PORT=$port PYTHONPATH=$W OUT=$R
cd "$W" || exit 1
T=$W/experiments/hybrid_agent_cache/tools/cc_run.sh
MEM="--gpu-memory-utilization ${GPU_UTIL:-0.85}"
n=0
echo "[$(date +%F' '%T)] lane $lane gpu $gpu port $port worktree $W" >> "$LOG"
while true; do
  line=$(sed -n "$((n + 1))p" "$Q" 2>/dev/null)
  if [ -z "$line" ]; then sleep 30; continue; fi
  n=$((n + 1))
  set -- $line
  [ "$1" = END ] && { echo "[$(date +%F' '%T)] END" >> "$LOG"; break; }
  echo "[$(date +%F' '%T)] start: $line" >> "$LOG"
  if [ "$1" = GSM8K ]; then
    name=$2; shift 2
    .venv/bin/vllm serve "$MODEL" --port "$PORT" --max-model-len 8192 $MEM "$@" > "$R/serve_gsm8k_4b_$name.log" 2>&1 &
    pid=$!
    for i in $(seq 1 200); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 3; done
    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py --port "$PORT" --num-shots 5 --temperature 0 --seed 0 \
      --save-results "$R/gsm8k_4b_$name.json" > "$R/gsm8k_4b_$name.log" 2>&1
    echo "[$(date +%F' '%T)] gsm8k $name exit $?" >> "$LOG"
    kill -INT $pid; wait $pid 2>/dev/null
  else
    tag=$1 sessions=$2; shift 2
    TAG=$tag EXTRA="$MEM $*" "$T" --sessions "$sessions" --max-req 40 >> "$LOG" 2>&1
  fi
  echo "[$(date +%F' '%T)] end: $line" >> "$LOG"
done
