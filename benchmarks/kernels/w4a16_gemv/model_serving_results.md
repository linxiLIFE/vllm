# Real streaming service comparison

Qwen2.5-1.5B-Instruct-GPTQ-Int4 on one RTX 3060, FP16, TP=1.
The same real checkpoint is served over loopback HTTP with either original
Humming or the experimental automatic dispatch in all 28 transformer layers.
This measures the actual streaming service, including final logits, sampling,
detokenization, scheduler, frontend and HTTP overhead.

## Workload and controls

- Maximum concurrency: 1, 2, 4, 8, 16 and 32. This is a concurrency cap, not
  a promise of a fixed GEMM batch on every decode step.
- Fixed input lengths 128 and 512; fixed output length 128, greedy sampling,
  no logprobs, seeded random prompts and ignore-EOS enabled.
- Two fresh-server rounds per backend. Backend order is Humming then hybrid
  in round 0 and hybrid then Humming in round 1; case order is reversed too.
- Each case has `max(32, concurrency * 8)` measured requests after
  `max(4, concurrency)` warmup requests and the benchmark's sanity request.
  Each backend/load point has 64–512 measured requests across both rounds.
- Prefix caching and torch.compile disabled; full decode CUDA graphs with
  capture sizes 1/2/4/8/16/24/32/48/64; eager prefill; FlashAttention 2.
- Identical fusion boundaries, including fused bias, and shared Humming weight
  layouts. The policy remains custom QKV through M=48, O through M=24 and
  gate/up/down through M=8; all other shapes retain Humming.
- Identical scheduler: max 64 sequences, 512 batched tokens, max context 2048,
  GPU memory utilization 0.5. Requests are replenished immediately subject
  to the client's concurrency cap, not an open-loop arrival trace.

## Measured results

All 48 load/round/backend runs complete successfully: 4,352 measured requests
and 557,056 output tokens, with zero failures and exactly 128 output tokens
per request. Hybrid worker shutdown records confirm 3,388 and 3,780 custom
eager/capture calls in the two rounds. Graph replays are not counted there.
The local server port is released after the final run.

Paired values below are **Humming / automatic dispatch**. TTFT and TPOT are
pooled P50 in milliseconds; throughput is aggregate output tokens/s.

### Input 128, output 128

| Concurrency | TTFT ms | TPOT ms/token | Output tokens/s | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 31.31 / 31.23 | 5.246 / 4.970 | 186.3 / 194.4 | +4.36% |
| 2 | 52.60 / 54.76 | 5.324 / 5.243 | 349.8 / 354.7 | +1.40% |
| 4 | 85.79 / 90.79 | 5.198 / 5.385 | 673.7 / 659.8 | -2.06% |
| 8 | 137.92 / 139.41 | 6.057 / 5.810 | 1131.4 / 1159.4 | +2.48% |
| 16 | 184.21 / 178.78 | 7.413 / 7.323 | 1840.5 / 1866.2 | +1.40% |
| 32 | 204.78 / 209.35 | 11.110 / 11.367 | 2529.0 / 2469.3 | -2.36% |

### Input 512, output 128

| Concurrency | TTFT ms | TPOT ms/token | Output tokens/s | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 79.28 / 81.02 | 5.238 / 5.098 | 172.2 / 179.2 | +4.02% |
| 2 | 85.53 / 87.45 | 5.917 / 5.568 | 305.6 / 316.3 | +3.50% |
| 4 | 200.97 / 212.00 | 6.260 / 6.168 | 516.3 / 513.4 | -0.55% |
| 8 | 261.47 / 274.21 | 8.320 / 8.185 | 783.2 / 775.2 | -1.02% |
| 16 | 284.15 / 284.81 | 13.286 / 13.439 | 1041.0 / 1038.3 | -0.26% |
| 32 | 310.15 / 319.40 | 25.666 / 26.197 | 1147.5 / 1131.9 | -1.36% |

### Input 128 tail latencies

Paired values are again Humming / automatic dispatch, in milliseconds.
Full tails for both input lengths, ITL and end-to-end latency are in the CSV.

| Concurrency | TTFT P95 | TTFT P99 | TPOT P95 | TPOT P99 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32.96 / 32.38 | 33.33 / 33.15 | 5.39 / 5.10 | 5.44 / 5.12 |
| 2 | 57.19 / 57.67 | 57.91 / 58.98 | 5.62 / 5.36 | 5.63 / 5.37 |
| 4 | 94.13 / 102.47 | 94.56 / 104.42 | 5.65 / 5.87 | 5.65 / 5.89 |
| 8 | 162.77 / 168.95 | 164.27 / 175.88 | 6.98 / 6.51 | 7.41 / 6.72 |
| 16 | 238.08 / 245.75 | 302.15 / 312.89 | 8.31 / 8.26 | 8.64 / 8.68 |
| 32 | 381.83 / 398.07 | 554.08 / 605.16 | 12.00 / 12.29 | 12.41 / 12.67 |

### Interpretation

The clearest benefit is single-concurrency output throughput: +4.36% for
128-token inputs and +4.02% for 512-token inputs. It improves in each
individual round too, but the gains vary: +2.98%/+5.76% for short inputs
and +1.79%/+6.24% for long inputs. Short-input TPOT P50 drops about 5.3%.
Single-request decode-rate estimates are about 191 versus 201 tokens/s for
short inputs; aggregate output throughput is lower because it includes TTFT.

There is no uniform TTFT improvement. Several higher-concurrency workloads
regress, and some small apparent gains reverse across rounds. For example,
short-input concurrency 4 throughput changes -7.78% and +3.40% in the two
rounds, averaging to -2.06% by the reported throughput aggregation. Thus the
fixed-batch transformer-forward advantage does not establish a general
serving advantage. This is a pilot on one desktop GPU, not evidence that
every concurrency below the kernel crossover improves in production.

## Metrics

TTFT is measured from the client sending a request to its first streamed text
chunk; it excludes waiting for the client's concurrency semaphore. TPOT is
computed for each request as `(latency - TTFT) / (output tokens - 1)`.
ITL measures individual streamed-chunk gaps. Latency percentiles pool the
per-request records from both rounds; they are not averages of percentiles.

Output tokens/s is aggregate output throughput for the complete measured
workload, including prefill and ramp/tail effects. It is reported as the
median of the two round throughputs. `1000 / TPOT_ms` is a separate per-request
decode-rate estimate and is included in the CSV, not confused with aggregate
throughput. Input tokens are not counted in output throughput.

The CSV contains P50/P95/P99 TTFT, TPOT, ITL and end-to-end latency, request/s,
output tokens/s, per-round ranges and relative differences. The plot shades
the range of round medians/throughputs, not confidence intervals. Two rounds
and these small request counts do not establish production P99 SLAs.

## Reproduction and artifacts

Use `model_serving_benchmark.py` as documented in
[README.md](README.md#streaming-http-service-metrics). The worker subclass
installs dispatch before model loading inside the real engine process; no
API or protocol behavior is changed. Each server is stopped after its arm.
The experiment does not leave a serving process running.

Local artifacts are under
`/home/linxi/.local/state/vllm-qwen-int4/gemv-experiment/model-serving/`:
`summary.csv`, `summary.json`, `metadata.json`, `serving-metrics.png`, SVG,
`commands.json`, all per-request JSON results, server/client logs and hybrid
worker dispatch counters. These counters cover eager/capture calls, not
CUDA graph replay counts.

Numerical kernel checks and the known greedy-token divergence are documented
in [model_dispatch_results.md](model_dispatch_results.md#correctness-and-limits).
This serving benchmark measures performance, not task accuracy equivalence.

This experiment and report were prepared with AI assistance.
