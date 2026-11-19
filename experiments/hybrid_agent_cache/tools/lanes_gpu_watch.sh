#!/bin/bash
# Logs, once a minute, every compute process on GPUs 2, 3 and 6 with its owner.
LOG=$HOME/study/vllm/experiments/hybrid_agent_cache/results/cctrace/gpu_watch_4b.log
while true; do
  ts=$(date +%F' '%T)
  apps=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader)
  nvidia-smi --query-gpu=index,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader,nounits |
  while IFS=, read idx bus util mem; do
    idx=${idx// /}; bus=${bus// /}
    case $idx in 2|3|6) ;; *) continue;; esac
    procs=$(echo "$apps" | awk -F', ' -v b="$bus" '$1==b{print $2}' | xargs -r ps -o user=,pid= -p | tr -s ' \n' ' ')
    echo "$ts gpu$idx util=${util// /}% mem=${mem// /}MiB procs:$procs"
  done >> "$LOG"
  sleep 60
done
