#!/bin/bash
# One row per 4B replay: hit, ceiling, attention-only, A, B (evicted / truncated), preemptions,
# wall, TTFT p50/p90, and other users' processes seen on the GPU during the run.
cd $HOME/study/vllm/experiments/hybrid_agent_cache
PY=$HOME/study/vllm/.venv/bin/python
T=$HOME/study/data/cc-traces-weka-062126/traces.jsonl
R=results/cctrace
W=$R/gpu_watch_4b.log
L=$HOME/study/tmp/lanes
printf "tag|hit|ceil576|attn_only|A|B|B_evicted|B_trunc|preempt|wall_s|ttft_p50|ttft_p90|gpu|foreign_procs\n"
for res in $R/res_cc_4b_*.json; do
  tag=$(basename $res .json); tag=${tag#res_cc_}
  a=$($PY -I tools/cc_analyze.py $T $res 2>/dev/null)
  hit=$(echo "$a" | grep -o "hit [0-9.]*%" | head -1 | cut -d' ' -f2)
  ceil=$(echo "$a" | grep -o "floored to 576: [0-9.]*%" | grep -o "[0-9.]*%")
  wall=$(echo "$a" | grep -o "wall=[0-9]*s" | grep -o "[0-9]*")
  p50=$(echo "$a" | grep -o "p50 [0-9.]*" | cut -d' ' -f2); p90=$(echo "$a" | grep -o "p90 [0-9.]*" | head -1 | cut -d' ' -f2)
  d=$($PY -I tools/lookup_decompose.py $T $res $R/probe_cc_$tag.jsonl.lookups 2>/dev/null)
  ao=$(echo "$d" | grep "attention-only hit" | grep -o "[0-9.]*%")
  A=$(echo "$d" | grep "attention blocks missing" | grep -o "[0-9.]*%")
  B=$(echo "$d" | grep "attention present, GDN missing" | grep -o "[0-9.]*%")
  rb=$($PY -I tools/residual_b.py $T $res $R/probe_cc_$tag.jsonl.lookups 2>/dev/null)
  be=$(echo "$rb" | grep -E "^  ckpt_evicted" | grep -o "[0-9.]*%")
  bt=$(echo "$rb" | grep -E "^  attn_truncated" | grep -o "[0-9.]*%")
  pa=$(grep -E "^vllm:num_preemptions_total" $R/m_cc_${tag}_after.txt | awk '{print $2}')
  pb=$(grep -E "^vllm:num_preemptions_total" $R/m_cc_${tag}_before.txt | awk '{print $2}')
  pre=$(awk -v a="$pa" -v b="$pb" 'BEGIN{printf "%d", a-b}')
  # GPU and time window from the lane logs (the first 4 runs predate the lanes: GPU 6).
  line=$(grep -h -A0 "start: $tag " $L/*.log 2>/dev/null | tail -1)
  lane=$(grep -l "start: $tag " $L/*.log 2>/dev/null | tail -1)
  gpu=$(head -1 "$lane" 2>/dev/null | grep -o "gpu [0-9]" | cut -d' ' -f2); gpu=${gpu:-6}
  t0=$(echo "$line" | grep -o "^\[[^]]*\]" | tr -d '[]')
  t1=$(grep -h "end: $tag " $L/*.log 2>/dev/null | tail -1 | grep -o "^\[[^]]*\]" | tr -d '[]')
  foreign="n/a"
  if [ -n "$t0" ] && [ -n "$t1" ]; then
    foreign=$(awk -v s="$t0" -v e="$t1" -v g="gpu$gpu" '($1" "$2)>=s && ($1" "$2)<=e && $3==g' $W |
      grep -o "procs:.*" | tr ' ' '\n' | grep -E "^[a-z]+\+?$" | grep -v -E "^(procs|${USER:0:7}\+|$USER)$" | sort -u | tr '\n' ',')
    foreign=${foreign:-none}
  fi
  echo "$tag|$hit|$ceil|$ao|$A|$B|${be:-0%}|${bt:-0%}|$pre|$wall|$p50|$p90|$gpu|$foreign"
done
echo
echo "GSM8K:"
for f in $R/gsm8k_4b_*.log; do echo "$(basename $f .log): $(grep -i -E "accuracy" $f | tail -1)"; done
