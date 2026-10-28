# Real Qwen2.5 W4A16 dispatch experiment

Measured on RTX 3060 SM86 with the real
Qwen2.5-1.5B-Instruct-GPTQ-Int4 checkpoint, FP16, TP=1, group128.
The experiment replaces the Humming linear method inside the real vLLM model
for all 28 transformer layers. It shares the original weight/scale allocation;
there is no model reload, repack or weight copy when M changes.
This is a process-local experimental integration, not a registered backend.

## Policy and fusion

| Projection | Custom path | Humming path |
| --- | --- | --- |
| QKV | M <= 48 | M >= 49 |
| O | M <= 24 | M >= 25 |
| Gate/up | M <= 8 | M >= 9 |
| Down | M <= 8 | M >= 9 |

The policy uses actual flattened input rows, including graph padding.
Unsupported configurations retain Humming. Fixed custom configurations come
from the checked-in layer-0 tuning tables; all model layers use those same
configurations. The QKV custom epilogue adds bias using Humming's existing
permuted allocation. Humming already fuses bias; this avoids introducing a
separate add when replacing its kernel. QKV and gate/up are already merged
projections in the original model. SiLU/multiply and downstream normalization
remain vLLM's existing operators.

## Dense batch 1–52 sweep

Every integer decode batch from 1 through 52 was measured with real model
weights, KV length exactly 128 and identical fusion boundaries. Each backend
runs two rounds, reversing both backend order and batch order. Each case has
30 replays per sample and five samples per round: 300 timed forwards per
backend/batch, plus warmup. These are transformer-forward times, excluding
the final logits projection, scheduler and sampler.

| Batch | Humming ms | Hybrid ms | Humming / hybrid |
| --- | ---: | ---: | ---: |
| 1 | 3.231 | 3.125 | 1.034 |
| 4 | 3.604 | 3.220 | 1.119 |
| 8 | 3.597 | 3.465 | 1.038 |
| 9 | 3.935 | 3.690 | 1.066 |
| 16 | 4.280 | 4.048 | 1.057 |
| 24 | 5.703 | 5.616 | 1.016 |
| 25 | 5.731 | 5.692 | 1.007 |
| 32 | 6.038 | 6.021 | 1.003 |
| 48 | 8.348 | 8.284 | 1.008 |
| 49 | 9.962 | 10.124 | 0.984 |
| 52 | 10.188 | 10.151 | 1.004 |

Summary values are medians of the two round medians. The plot connects every
measured point without smoothing. Shading shows the range of round medians
or per-round latency ratios, not a confidence interval. Small-batch timing
remains variable: per-run sample-median spreads reach 12.7–15.5%, and batch-2
speedup ranges from 0.992 to 1.221 across rounds. Avoid treating the central
curve as a precise fixed speedup.

The MLP/O/QKV dashed boundaries at 9/25/49 are configured dispatch choices
from the earlier kernel experiments, not newly established model crossovers.
Both paths use Humming exclusively at batch 49–52; their measured ratios
of 0.984–1.004 illustrate separate-run variability. Both curves also have
steps at batches 17 and 33, so a latency step alone does not identify a
custom/Humming switch.

All 208 case runs pass finite-output, exact graph/eager and deterministic
graph-replay checks. Both hybrid sweeps additionally pass 4,928 first-use
layer/batch comparisons against Humming at relative-L2 < 0.002.

Reproduce with `model_batch_sweep.py` as documented in
[README.md](README.md#real-model-dispatch-experiment). Local artifacts are in
`/home/linxi/.local/state/vllm-qwen-int4/gemv-experiment/model-batch-sweep/`:
`batch-sweep.png`, `batch-sweep.svg`, `summary.csv`, and four complete
`round*-*.json` records with logs, validation counts and command metadata.

## Earlier sparse transformer forward

Initial independent runs, in microseconds. Decode has KV length 128 with no
jitter. Each case runs the real model forward in a CUDA graph, 50 replays
per sample and seven samples after the benchmark's warmup. The final logits
projection, sampling, scheduling and detokenization are excluded. L2 is not
flushed between layers. The generation benchmark below includes those costs.

| Case | Humming | Hybrid | Humming / hybrid |
| --- | ---: | ---: | ---: |
| Decode M=1 | 3356.7 | 3074.6 | 1.092 |
| Decode M=8 | 3708.9 | 3270.1 | 1.134 |
| Decode M=9 | 3929.1 | 3574.8 | 1.099 |
| Decode M=24 | 5689.3 | 5594.6 | 1.017 |
| Decode M=25 | 5726.2 | 5668.4 | 1.010 |
| Decode M=48 | 8350.7 | 8267.8 | 1.010 |
| Decode M=49 | 10019.3 | 10081.3 | 0.994 |
| Decode M=64 | 10484.7 | 10537.0 | 0.995 |
| Prefill 128 | 17643.0 | 17440.3 | 1.012 |
| Prefill 512 | 66351.6 | 65607.2 | 1.011 |

Small-M sample spreads reach 12.24% in the initial hybrid M=1 run and 5.90%
in the baseline M=1 run. Large-M and prefill differences around 1% should
not be interpreted as an improvement: these execute identical Humming paths
once every projection exceeds its cutoff.

A second campaign reverses backend order (Humming first, then hybrid), uses
150 replays per sample and seven samples, and confirms the small-M benefit:

| Decode M | Humming us | Hybrid us | Humming / hybrid |
| --- | ---: | ---: | ---: |
| 1 | 3326.0 | 3013.1 | 1.104 |
| 8 | 3635.2 | 3352.6 | 1.084 |
| 24 | 5674.0 | 5572.6 | 1.018 |
| 48 | 8310.8 | 8247.3 | 1.008 |
| 64 | 10603.5 | 10611.2 | 0.999 |

M=1 spreads remain 2.86%/9.01% (baseline/hybrid), and M=8 spreads
5.69%/1.30%. The measurements support a small-batch gain, not a precise
fixed percentage across conditions.

## Bias fusion ablation

Same custom kernels and selections, with either a separate QKV bias add or
bias fused into the output store; 150 replays/sample and seven samples.
Separate bias tensors are unpermuted once outside timing.

| Decode M | Separate add us | Fused bias us | Separate / fused |
| --- | ---: | ---: | ---: |
| 1 | 3062.3 | 3013.6 | 1.016 |
| 8 | 3324.3 | 3259.2 | 1.020 |

Profiler traces confirm 28 fewer GPU activities per transformer forward:
339 to 311 at M=1 and 395 to 367 at M=8. At M=1, all 28 separate
`CUDAFunctor_add<Half>` calls disappear. Profiled kernel durations are not
used as latency measurements. Timing sample spreads still reach 8.51% at
M=1 and 6.99% at M=8; the 1.6–2.0% measured benefit is indicative, while
the removal of 28 launches is directly verified.

## Actual generation

Ordinary generation with `--generation-logprobs 0`, greedy sampling, prefix
caching off, full decode graphs and eager prefill. Times include the final
logits projection, scheduling, sampling and detokenization. Each batch/length
has one warmup and three measured repetitions; medians are in milliseconds.

| Batch | Tokens/request | Humming | Hybrid | Humming / hybrid |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 18.95 | 22.02 | 0.861 |
| 1 | 65 | 360.24 | 343.19 | 1.050 |
| 8 | 1 | 20.70 | 22.96 | 0.901 |
| 8 | 65 | 380.12 | 366.38 | 1.038 |
| 32 | 1 | 62.87 | 62.95 | 0.999 |
| 32 | 65 | 553.09 | 553.15 | 1.000 |

The complete 65-token generations improve by about 5% and 4% at batch 1
and 8, while batch 32 is unchanged. One-token generation regresses at small
batch: the experimental Python dispatch and eager execution still cost host
time. The 65-minus-1 estimate is 5.33 to 5.02 ms/decode step at batch 1 and
5.62 to 5.37 ms at batch 8. This estimate is sensitive to first-token overhead.

Token comparisons repeat the logprob-enabled finding below: all one-token
outputs and all batch-8/32 sequences match, while batch-1 diverges.

## Generation with logprobs

`LLM.generate`, greedy sampling, prefix caching off, full decode CUDA graphs,
eager prefill, top-5 logprobs and detokenization enabled. Four fixed prompts
cover English, Chinese arithmetic and Python. Each batch/length has one warmup
and three timed generations. Values are medians in milliseconds.

| Batch | Tokens/request | Humming | Hybrid |
| --- | ---: | ---: | ---: |
| 1 | 1 | 18.55 | 24.10 |
| 1 | 65 | 359.79 | 351.71 |
| 8 | 1 | 21.32 | 23.14 |
| 8 | 65 | 383.89 | 372.64 |
| 32 | 1 | 63.97 | 63.50 |
| 32 | 65 | 566.79 | 568.05 |

Subtracting each batch's one-token time from its 65-token time and dividing
by 64 estimates average decode step cost. It includes scheduler, sampling
and host overhead and is not a per-token profiler measurement.

| Batch | Humming ms/step | Hybrid ms/step | Humming / hybrid |
| --- | ---: | ---: | ---: |
| 1 | 5.332 | 5.119 | 1.042 |
| 8 | 5.665 | 5.461 | 1.037 |
| 32 | 7.857 | 7.883 | 0.997 |

These are small gains from three repetitions, not statistically established
serving-throughput gains. Prefill and short-generation overhead can offset
the faster decode kernels.

## Correctness and limits

- 504 first-use layer/batch comparisons in the forward run and 672 in the
  generation run pass finite-output and relative-L2 < 0.002 gates against
  Humming using the same actual layer inputs and parameters.
  The separate-bias ablation also passes these gates.
- All one-token outputs match. For 65-token outputs, all eight batch-8 and
  all 32 batch-32 sequences match exactly. Batch-1 diverges at token 35.
  Before that divergence, Humming's two leading logprobs are -1.25908 and
  -1.27470; hybrid's are -1.27615 and -1.26053 for those same candidates.
  Small accumulation differences flip two close candidates and subsequent
  text differs. This does not establish task accuracy equivalence.
- Forward graph/eager comparisons are exact for decode and prefill128.
  Prefill512 relative RMS is about 0.001 for both arms, even though the
  hybrid policy uses Humming exclusively at this size.
- The integration uses Python/ctypes for custom launches, with an opaque
  PyTorch custom op. Graph replay avoids this launch overhead; eager
  execution can have different performance.

## Artifacts and reproduction

Use the commands in [README.md](README.md#real-model-dispatch-experiment).
Local raw results and dispatch records are under
`/home/linxi/.local/state/vllm-qwen-int4/gemv-experiment/model-hybrid/`:
`forward.json`, `baseline-forward.json`, `stats.json`, `generation.json`,
`baseline-generation.json`, `generation-stats.json`, and metadata files.
Confirmation runs use `confirm-*.json`. Metadata records command arguments,
source hashes, GPU, PyTorch/CUDA versions and repository commit. Humming uses
the pinned `humming-main-package` from the existing experiment.
Ordinary generation is recorded in `plain-baseline-generation.json` and
`plain-hybrid-generation.json`. `fused-profile/` and `unfused-profile/`
contain kernel summaries and compressed traces for the fusion ablation.

This experiment and report were prepared with AI assistance.
