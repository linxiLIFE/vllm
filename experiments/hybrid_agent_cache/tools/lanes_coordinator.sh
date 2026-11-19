#!/bin/bash
# Waits for the 4B prerequisites, derives the high-pressure session count and the
# GDS prefill-cost coefficients, writes the lane queues and starts GPU 2 and GPU 3.
L=$HOME/study/tmp/lanes
R=$HOME/study/vllm/experiments/hybrid_agent_cache/results/cctrace
PY=$HOME/study/vllm/.venv/bin/python
LOG=$L/coordinator.log
log() { echo "[$(date +%F' '%T)] $*" >> "$LOG"; }

log "waiting for prerequisites"
until grep -q PROBES_DONE $HOME/study/tmp/prereq/probes.out && grep -q FIT_DONE $HOME/study/tmp/prereq/fit.out; do sleep 30; done
log "prerequisites done"

cap() { grep -o "GPU KV cache size: [0-9,]* tokens" "$1" | tail -1 | grep -o "[0-9,]*" | tr -d ,; }
CAP_BOTH=$(cap $R/startup_4b_both.log)
log "capacity both=$CAP_BOTH fp8kv=$(cap $R/startup_4b_fp8kv.log) bf16ssm=$(cap $R/startup_4b_bf16ssm.log)"

# Session count whose total context / capacity is closest to 1.38 (the 5-session default ratio).
NHI=$($PY -I -c "
import sys
sizes=[94720,64704,64128,65984,97984,109824,110528,93440,99520,92928,105600,69504]
cap=int(sys.argv[1]); best=min(range(2,13), key=lambda n: abs(sum(sizes[:n])/cap-1.38))
print(best)" "$CAP_BOTH")
log "high-pressure sessions with fp8+bf16: $NHI"

# Fit t = q * (D + E * (kv - q/2) / 1e5) on the 4B forward benchmark.
COEF=$($PY -I -c "
import json, sys
d = json.load(open(sys.argv[1]))
X, Y = [], []
for c in d:
    r = c['requests'][0]; q, kv = r['query_len'], r['kv_len']
    X.append((q, q * (kv - q / 2) / 1e5)); Y.append(c['median_us'] / 1e6)
sxx = sum(a*a for a, _ in X); sxy = sum(a*b for a, b in X); syy = sum(b*b for _, b in X)
sxt = sum(a*t for (a, _), t in zip(X, Y)); syt = sum(b*t for (_, b), t in zip(X, Y))
det = sxx*syy - sxy*sxy
D = (sxt*syy - syt*sxy) / det; E = (syt*sxx - sxt*sxy) / det
print(f'{D:.3e} {E:.3e}')" $R/prefill_cost_fit_4b_3090.json 2>>"$LOG")
log "prefill cost fit (D E): ${COEF:-FAILED}"

BOTH="--kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16"
FP8="--kv-cache-dtype fp8"
BF16="--mamba-ssm-cache-dtype bfloat16"

# GPU 6 (#61282 lane, already running bf16 jobs): append the rest.
cat >> $L/p61282.q <<EOF
4b_n5_61282_fp8kv 5 $FP8
4b_n2_61282_fp8kv 2 $FP8
4b_n5_61282_both 5 $BOTH
4b_n2_61282_both 2 $BOTH
4b_n${NHI}_61282_both $NHI $BOTH
4b_n5_61282_r2 5
END
EOF

# GPU 2 (main lane): memory saving without #61282, GSM8K, repeat.
cat > $L/main.q <<EOF
GSM8K fp8kv $FP8
4b_n5_main_both 5 $BOTH
4b_n2_main_both 2 $BOTH
4b_n${NHI}_main_both $NHI $BOTH
GSM8K base
GSM8K bf16ssm $BF16
GSM8K both $BOTH
4b_n5_main_r2 5
END
EOF

# GPU 3: the other upstream PRs, one worktree after another.
printf "4b_n5_63851 5\n4b_n2_63851 2\nEND\n" > $L/p63851.q
printf "4b_n5_63851hold 5\n4b_n2_63851hold 2\nEND\n" > $L/c63851.q
if [ -n "$COEF" ]; then
  GDS="--mamba-frontier-checkpoint --prefix-cache-eviction gds --prefill-cost-coefficients $COEF"
  printf "4b_n5_66881 5 $GDS\n4b_n2_66881 2 $GDS\nEND\n" > $L/p66881.q
  printf "4b_n5_66881_61282 5 $GDS\n4b_n2_66881_61282 2 $GDS\nEND\n" > $L/c66881.q
else
  echo END > $L/p66881.q; echo END > $L/c66881.q; log "66881 lanes skipped: no fit"
fi

cd $L
setsid nohup ./lane_runner.sh main 2 18002 > main.runner.out 2>&1 < /dev/null &
setsid nohup bash -c 'for l in p63851 c63851 p66881 c66881; do ./lane_runner.sh $l 3 18003; done; echo GPU3_CHAIN_DONE' > gpu3.runner.out 2>&1 < /dev/null &
log "started GPU 2 (main) and GPU 3 (63851 -> 63851hold -> 66881 -> 66881+61282)"
