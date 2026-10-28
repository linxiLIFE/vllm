# Investigation of serving variability

The evidence does not support blaming prefill alone. Fixed-model GPU forward
still favors the custom kernels, while fixed unrelated GPU work also exhibits
large timing variation on this shared RTX 3060 / WSL2 / WDDM environment.
The experimental dispatch additionally adds measurable eager CPU overhead.
These findings explain why a small kernel saving does not guarantee a stable
end-to-end serving percentage; they do not identify the source of every old
slow round retrospectively.

## 1. Alternate within the same loaded model

`model_forward_abba.py` loads real weights once, then switches the linear
method between original Humming and hybrid before each exact-batch capture.
Each shape runs Humming/hybrid/hybrid/Humming twice, 60 replays per sample,
five samples per arm. Weight pointers, input shapes and KV initialization
remain shared. Values below are medians of the four arm medians, in ms.

| Shape | Humming | Hybrid | Humming / hybrid |
| --- | ---: | ---: | ---: |
| Decode M=4, KV=128 | 3.538 | 3.195 | 1.107 |
| Decode M=4, KV=512 | 4.028 | 3.699 | 1.089 |
| Prefill 128 | 17.567 | 17.591 | 0.999 |
| Prefill 512 | 66.472 | 66.422 | 1.001 |

Thus the fixed-batch GPU benefit remains. Large prefill uses Humming in both
paths and its GPU duration differs by less than 0.2%. This benchmark excludes
the logits projection, sampling, host scheduler and HTTP frontend.

## 2. Instrument serving steps and remove request refill

`model_serving_diagnose.py` uses concurrency 4, input 128, output 256. Each
server compares four-request waves with no refill until all four complete,
and immediate semaphore-based refill. Three repeats reverse protocol order.
Five fresh servers run Humming/hybrid/wrapped-Humming/hybrid/Humming.
The wrapped-Humming control performs dispatch checks but always calls Humming.

Asynchronous CUDA events bracket model forward, execution and sampling;
host timestamps and batch/prefill/Graph metadata are recorded without
per-step synchronization. Records resolve after the load stops. These are
instrumented diagnostic runs, not replacements for the uninstrumented
serving benchmark. There are 480 measured requests and 122,880 output tokens.

Every measured window has 1,028 steps: eight with prefill, 1,016 pure
four-request decode steps, and four other decode steps. All pure four-request
decode steps use full CUDA graphs. Waves and refill have the same counts.
Removing refill does not eliminate the variation; it is not chiefly a change
between eager execution and Graph replay for this workload.

Representative values are medians across each server's six window medians:

| Server arm | Forward GPU ms | Logits/sampling GPU ms | Step interval ms | Middle ITL ms |
| --- | ---: | ---: | ---: | ---: |
| Humming, first | 3.629 | 1.431 | 5.108 | 5.151 |
| Hybrid, first | 3.365 | 1.430 | 4.963 | 4.986 |
| Wrapped Humming | 3.658 | 1.431 | 5.170 | 5.193 |
| Hybrid, second | 3.373 | 1.430 | 4.929 | 4.952 |
| Humming, last | 3.760 | 1.431 | 5.168 | 5.197 |

The custom forward saves about 0.25–0.4 ms, while the unchanged
logits/sampling phase costs about 1.43 ms at its median. Consequently a
9–11% transformer-forward speedup must be diluted in complete inference.
The service middle-ITL measurement uses tokens 33–224 to exclude startup
and the request tail.

In the last two server arms, the median gap on the main CUDA stream from
one sampling end to the next execution start is about 0.002 ms, also about
0.002 ms at P99. The compute stream is generally supplied with queued work;
large CPU starvation gaps are not the principal steady-decode explanation
in these diagnostic runs. Host durations overlap queued GPU work and must
not be added directly to GPU durations.

### Dispatch-wrapper overhead is real, but separate

Eager prefill's forward host time is about 12.1–12.3 ms for bare Humming,
15.7 ms for wrapped Humming, and 16.3–16.8 ms for hybrid. The wrapped control
uses identical Humming kernels, isolating additional Python dispatch/layout
checks and accounting work. That overhead is outside the GPU-only forward
benchmark. GPU prefill spans remain much closer because CPU submission
overlaps GPU execution; the host difference must not be interpreted as an
equal added TTFT. Static checks could be cached in a later optimization.

## 3. Fixed logits GEMM reproduces the device-side variation

`sampler_jitter.py` removes the model, scheduler, HTTP, prefill and dispatch.
It captures just `FP16[4,1536] @ FP16[1536,151936]`, matching the logits
projection shape. Allocation and initialization occur before timing; fixed
all-one inputs allow an exact output check of 1536 for every output element.
Each mode warms for one second and measures 1,000 identical Graph replays.
Both queued replay and synchronization after every replay are repeated twice.

Without a profiler, CUDA event brackets measure:

| Mode | P50 ms | P90 ms | P99 ms |
| --- | ---: | ---: | ---: |
| Queued, repeat 0 | 1.384 | 2.003 | 2.444 |
| Synchronize each, repeat 0 | 1.384 | 2.000 | 2.223 |
| Queued, repeat 1 | 1.384 | 2.005 | 2.265 |
| Synchronize each, repeat 1 | 1.384 | 2.003 | 2.230 |

An independent Nsight Systems trace contains 6,599 executions of exactly one
cuBLAS kernel and one launch configuration: grid `(1,4748,1)`, block
`(128,1,1)`, 54 registers/thread. Its kernel-activity durations have P50
1.274 ms, P90 1.840 ms and P99 2.100 ms; 1,343 executions exceed 1.6 ms.
This confirms variation in the kernel activity itself, not merely HTTP
chunk timing or a change in selected operator/launch grid. Activity times
and CUDA event brackets have different boundaries and are not subtracted
from one another.

The same untouched sampling phase in serving has median GPU span about
1.43 ms but P90 about 2.2 ms and P99 about 2.9 ms. Such variation is larger
than the custom forward's few-tenths-of-a-millisecond saving. A small number
of sequential server rounds is insufficient to assume a precise serving gain.

## 4. Environment and attribution limits

The machine reports WSL2 (`6.18.54.1-microsoft-standard-WSL2`) and WDDM.
During measured windows, 200-ms NVML samples show SM clocks 1935–1957 MHz
and memory clock 7301 MHz. This does not support a large sustained clock
change as the sole explanation, although it cannot exclude faster changes.

A Windows GPU-engine counter snapshot taken while the isolated benchmark
process ran shows other workloads on physical GPU 0: `msedgewebview2`
processes at 12% and 2% 3D utilization, and `csrss` at 1%. Therefore the GPU
was not established to be exclusive to the benchmark. Concurrent graphics
work and WDDM scheduling are plausible contributors, not proven per-spike
causes. No host process was stopped for an interference-removal ablation.
Eight follow-up Windows counter samples during the isolated GEMM confirm
concurrent CUDA-engine activity of 95–99% and `msedgewebview2` 3D-engine
activity of 11–16% on physical GPU 0. These are separate engine utilization
counters and must not be added into a single utilization percentage.
Nsight reports GPU context-switch tracing unavailable on this system, so
the trace cannot attribute each stretched kernel to a particular host process.

Earlier fixed-batch measurements were not uniformly stable either: batch-2
round speedup ratios ranged from 0.992 to 1.221. The old serving concurrency-4
throughput changes of -7.78% and +3.40% therefore cannot establish a genuine
kernel regression. Conversely, the new controls do not prove that all old
variation came from one Windows process. Accurate reporting must retain both
the observed model benefit and these environment/measurement limits.

## Artifacts and reproduction

Local root:
`/home/linxi/.local/state/vllm-qwen-int4/gemv-experiment/model-stability/`.

- `forward-abba/results.json`: same-model alternating GPU forward samples.
- `serving-graceful/`: five server/client logs, request records, step-event
  records, clocks/power telemetry, exact commands and `step-summary.csv`/JSON.
- `serving/`: initial diagnostic attempt. Its default abort shutdown lost
  event exports; its request records are retained, but it is excluded from
  the step analysis. The corrected runner uses `--shutdown-timeout 15`.
- `sampler-jitter-unprofiled.json`: fixed-GEMM measurements without profiler.
- `sampler-jitter.nsys-rep`, SQLite, JSON: independent CUDA/kernel trace.
- `sampler-context.log` is represented by the retained tool log at
  `/tmp/sampler-context.log`; it records unavailable context-switch tracing.
- `gpu-jitter-evidence.png`/SVG and `gpu-jitter-summary.json`: compact evidence.
- Windows GPU-engine and process snapshots are retained in `windows-gpu-*.json`.

Use `model_forward_abba.py` with the usual real-weight forward arguments;
`model_serving_diagnose.py` takes the model, Humming root and output directory.
Run `analyze_model_stability.py --input-dir <serving-graceful>` to summarize.
`sampler_jitter.py --output <file.json>` runs the isolated control. Its optional
Nsight capture range is CUDA profiler API, CUDA Graph node trace, CPU sampling
disabled. All experiment servers are stopped after measurement.

This investigation and report were prepared with AI assistance.
