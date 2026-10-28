# Experimental small-batch W4A16 kernels

This experiment compares CUDA variants with Humming and vLLM Marlin using real
Qwen2.5-1.5B-Instruct-GPTQ-Int4 attention and MLP weights. It is a
standalone benchmark, not a registered vLLM quantization backend.

## Synthetic DeepSeek V4.1 projection shapes

`shape_scaling.py` generates random packed INT4 weights without loading a
checkpoint. It compares the existing fused-bias CUDA entries in
`model_dispatch.cu` against Humming with fused bias, sharing the same transformed
weight, scale and bias pointers. `--bias-modes none fused` also measures the
bias-free case. `separate` is an optional custom GEMM plus separate bias-add
ablation; Humming still fuses bias in that arm.

Shapes come from the official DeepSeek-V4.1-Flash inference implementation at
revision `314113f8a7dadb388215396ba5fbee2c610a04d6`. They include Q low-rank
projections, KV, grouped output projection, expert gate/up and down, and indexer
Q. `v41_o_a_group` measures one of eight independent groups. Expert gate/up
stacks two projections for a single expert; its M is the tokens assigned to
that expert. The four Qwen shapes provide a synthetic small-model control.
Native V4.1 FP8/FP4 precision, MoE routing, grouped scheduling, attention and
communication are outside this experiment. The official projections are
bias-free; nonzero bias is an explicit operator ablation.

Tune both families, then fix their configurations for independent confirmation:

```bash
LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
  benchmarks/kernels/w4a16_gemv/shape_scaling.py \
  --humming-root /path/to/humming-package --cupti-root /path/to/cupti-package \
  --output-dir /tmp/v41-shape-tune --batches 1 8 32 64 \
  --rounds 1 --repeats 50 --warmup-ms 100
LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
  benchmarks/kernels/w4a16_gemv/shape_scaling.py \
  --humming-root /path/to/humming-package --cupti-root /path/to/cupti-package \
  --output-dir /tmp/v41-shape-confirm --batches 1 8 32 64 \
  --selection /tmp/v41-shape-tune/selection.json \
  --seed 1 --rounds 2 --repeats 150 --warmup-ms 100
.venv/bin/python benchmarks/kernels/w4a16_gemv/shape_scaling_report.py \
  --input-dir /tmp/v41-shape-confirm
```

The plotting command needs matplotlib. The benchmark saves raw CUPTI samples,
round medians, selected configurations, independent decoded-weight reference
checks and source hashes. Every case checks correctness before timing, and
confirmation reverses implementation order between rounds. Cold-L2 flush,
initial weight transformation, allocations and compilation are excluded.
Reported bandwidth uses ideal bytes for one read of each input/parameter and
one output write; it is not measured DRAM traffic. The plotting command produces
`report.md`, `speedup.png` and `speedup.svg` from saved results.
See the [V4.1 shape results](results/v41-shape-scaling/README.md) for the measured
speedups, regressions and separate-bias ablation.

### Wider tiling and register prefetch

`wide_gemm.cu` extends the fused kernel family with 32/64-column output tiles,
8/16/32/64-row reuse, and 1/2/4 K warps. Shorter K loops control register
pressure; selected variants prefetch one 128-element group's packed weight
words into registers before consuming them. Weight dequantization remains
inside GEMM, using the existing Humming allocation and fused bias epilogue.

The harness builds this cubin incrementally with the CMake `wide_gemm` target.
`--build-dir` shares a build across runs; `--variants` restricts tuning to a
candidate list. `--previous-selection` adds the original custom configuration
as a control. It does not alter the fixed winner supplied by `--selection`.
`--profile-only` runs reference checks without CUPTI timing, allowing Nsight
Compute to collect counters without competing CUPTI subscribers.

Reproduce the fixed three-arm confirmation on SM86:

```bash
LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
  benchmarks/kernels/w4a16_gemv/shape_scaling.py \
  --humming-root /path/to/humming-package --cupti-root /path/to/cupti-package \
  --output-dir /tmp/v41-opt-confirm --build-dir /tmp/v41-opt-build \
  --batches 1 8 32 64 --bias-modes fused --seed 1 \
  --selection benchmarks/kernels/w4a16_gemv/results/v41-shape-optimized/selection.json \
  --previous-selection benchmarks/kernels/w4a16_gemv/results/v41-shape-scaling/selection.json \
  --rounds 2 --repeats 150 --warmup-ms 100
```

These are standalone kernel choices. The Qwen real-model dispatch below does
not automatically acquire the new configurations or support DeepSeek's native
FP8/FP4 checkpoint. See the [optimization results](results/v41-shape-optimized/README.md)
for controlled old/new/Humming comparisons and limitations.

## Real-model dispatch experiment

`model_dispatch.py` installs a process-local dispatch in vLLM's
`HummingLinearMethod.apply` and loads the real checkpoint with
`--quantization humming --load-format auto`. It shares Humming's transformed
parameters across all 28 layers: custom kernels handle QKV through M=48, O
through M=24, and gate/up and down through M=8. Larger and unsupported shapes
use the original Humming method. The QKV epilogue consumes Humming's permuted
bias directly; `--unfused-bias` provides a separate-add ablation. Humming
already fuses bias, so this preserves that feature in the custom path.

```bash
LIBRARY_PATH=/usr/lib/wsl/lib PYTHONPATH=/path/to/humming-package \
  .venv/bin/python benchmarks/kernels/w4a16_gemv/model_dispatch.py \
  --backend hybrid --dispatch-build /tmp/model-dispatch/build \
  --dispatch-stats /tmp/model-dispatch/stats.json --validate-layers \
  --model /path/to/Qwen2.5-1.5B-Instruct-GPTQ-Int4 \
  --quantization humming --load-format auto --dtype float16 \
  --max-model-len 1024 --max-num-seqs 64 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.5 --kv-len-jitter 0 \
  --case decode1=1*1@128 --case decode8=8*1@128 \
  --case decode48=48*1@128 --case decode49=49*1@128 \
  --case prefill=128 --iters 50 --samples 7 \
  --output-json /tmp/model-dispatch/forward.json
```

Repeat with `--backend humming` and separate output paths for the baseline.
The forward benchmark captures the real transformer in CUDA graphs; it excludes
the final logits projection, sampling and scheduler overhead and does not
flush L2 between layers. Use
`--generation --generation-batches 1 8 32` instead of `--case` arguments to
time `LLM.generate` with greedy 1- and 65-token outputs, top-5 logprobs,
one warmup and three repetitions. Generation uses full decode graphs and eager
prefill, with prefix caching disabled. Its times include host scheduling,
sampling, logprobs and detokenization. Difference the 65- and 1-token times
to estimate average decode cost. Four fixed English/Chinese/code prompts
provide token/logprob comparisons, not a task-accuracy evaluation.
Use `--generation-logprobs 0` for ordinary generation without logprobs.
See [real-model results](model_dispatch_results.md) for measurements and limits.

For a dense real-model decode sweep, `model_batch_sweep.py` measures every
integer batch from 1 through 52 by default. Two rounds reverse both backend
order and batch order. Both paths retain fused bias and the same remaining
model operators. Each hybrid run checks all eligible layers against Humming.

```bash
LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
  benchmarks/kernels/w4a16_gemv/model_batch_sweep.py \
  --model /path/to/Qwen2.5-1.5B-Instruct-GPTQ-Int4 \
  --humming-root /path/to/humming-package \
  --output-dir /tmp/model-batch-sweep \
  --max-batch 52 --rounds 2 --iters 30 --samples 5
```

The plotting step requires matplotlib. Outputs include per-run JSON, logs,
dispatch validation records, metadata, `summary.csv`, `batch-sweep.png` and
`batch-sweep.svg`. Shading shows the range of round medians, not a confidence
interval. Dashed lines mark configured projection dispatch boundaries from
the earlier kernel experiments; they are not newly inferred model crossovers.
Append `--plot-only` with the same output directory and sweep settings to
regenerate plots without using the GPU.

`--validate-layers` checks custom outputs against Humming on each first
layer/batch invocation outside capture, requiring finite values and relative
L2 below 0.002. Dispatch stats count eager/capture calls, not graph replays.
Metadata records command, source hashes, GPU and library versions.

### Streaming HTTP service metrics

`model_serving_benchmark.py` starts isolated local vLLM servers and invokes
`vllm bench serve` against their streaming completions endpoint. The hybrid
server loads `model_serving_worker.Worker`, which installs the same dispatch
inside the engine process before loading the model. Humming and hybrid retain
the same fused bias, FlashAttention backend, decode graphs and scheduler
settings. Prefix caching and torch.compile are disabled in both.

```bash
LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
  benchmarks/kernels/w4a16_gemv/model_serving_benchmark.py \
  --model /path/to/Qwen2.5-1.5B-Instruct-GPTQ-Int4 \
  --humming-root /path/to/humming-package \
  --output-dir /tmp/model-serving
```

Defaults: concurrency 1/2/4/8/16/32, input lengths 128/512, fixed output
length 128, two backend-order-reversed rounds, seeded random prompts, greedy
sampling, no logprobs and forced output length. Each case has
`max(32, concurrency * 8)` measured requests after warmup. This is a closed-loop
concurrency test over loopback HTTP, not an open-loop production arrival trace.
TTFT starts when the client sends the request, excluding time waiting for its
concurrency semaphore. TPOT is `(request latency - TTFT) / (output tokens - 1)`.
Output throughput includes prefill and the complete measured workload duration.

Once finished, append `--summarize-only` to the same command to produce
`summary.csv`, `summary.json`, `metadata.json`, `serving-metrics.png` and SVG
(requires matplotlib). Latency percentiles pool requests from both rounds;
throughput is the median of round throughputs. Shading shows the range of
round medians for latency and of round throughputs; it is not a confidence interval.
Per-request timings, texts, errors, server/client logs and exact commands are
retained. P95/P99 from this small sample are exploratory, not production SLAs.
Servers are stopped after each arm. Dispatch counts are saved on worker shutdown.
See [streaming service results](model_serving_results.md) for measured TTFT,
TPOT, output throughput and tail latencies, including regressions.

### Diagnosing serving variability

[Stability investigation](model_stability_results.md) compares same-model
ABBA forward timing, fixed request waves versus refill, wrapped-Humming
controls, asynchronous per-step timings and a fixed standalone logits GEMM.
`model_forward_abba.py --output-dir ...` accepts the usual real-weight
`vllm bench forward` arguments. `model_serving_diagnose.py` accepts `--model`,
`--humming-root` and `--output-dir`; it runs a fixed concurrency-4 investigation
and uses a 15-second shutdown timeout to export diagnostic events safely.
`analyze_model_stability.py --input-dir ...` summarizes those saved records.
`sampler_jitter.py --output ...` reproduces GPU timing variation without a
model, HTTP, prefill or custom dispatch. Its optional Nsight capture uses
CUDA profiler API and CUDA Graph node tracing.

Step diagnostics are opt-in through `W4A16_STEP_DIAGNOSTICS`; ordinary
serving benchmarks do not record these extra events. `wrapped` is a diagnostic
worker mode that retains dispatch checks while always calling Humming.
Instrumented serving numbers should not replace the ordinary benchmark.

The [checked-in measurement archive](results/README.md) contains current
latency tables, fixed configurations, correctness records and plots, including
the optimized QKV crossover at M=49 and the existing O/gate-up/down boundaries.

## Modifying Humming's input staging

`humming_direct.py` and its accompanying patch build an isolated copy of the
pinned Humming package. On the SM86 FP16/INT4 group128 path, mode 1 sends
packed weights directly to their consumer registers; mode 2 also loads
activations and group scales directly and removes empty mainloop pipeline
waits/refill barriers. Output reduction and epilogue shared memory remain.
Compact stage fields retain one unused int4 anchor. The original package is
not modified. The harness checks that the applied patch matches its source.

```bash
.venv/bin/python benchmarks/kernels/w4a16_gemv/humming_direct.py \
  --source /path/to/pinned-humming-package \
  --destination /tmp/humming-direct-package
```

Use `shared_layout_benchmark.py --humming-direct` with the copied
`--humming-root` and the usual model/CUPTI/source arguments. Tune with
`--phase tune`, then confirm with `--phase compare --selection ...`.
Both original and modified paths sweep default/stage2/no-Stream-K configs.
Confirmation retains matched original-config controls as well as the best
configuration of each family. Registered kernel handles are fixed before
timing, so changing the Python preparation mode cannot switch a timed kernel.

Independent M=8 cold L2 confirmation on RTX 3060, in microseconds:

| Projection | Our custom | Original Humming | Weight-direct Humming | All-input-direct Humming |
| --- | --- | --- | --- | --- |
| QKV | 9.22 | 9.68 | 11.53 | 9.92 |
| O | 7.65 | 7.94 | 8.42 | 8.25 |
| Gate/up | 50.42 | 53.33 | 56.50 | 53.68 |
| Down | 28.16 | 28.41 | 31.69 | 28.71 |

This rewrite does not yield a universal improvement. O benefits at some warm
points: M=9 improves from 7.07 to 6.27 us, and M=16 from 7.21 to 6.53 us
(essentially tied with our 6.54 us). Several large-M all-direct cases regress
by over 20%. These are results of this rewrite, not a bound on future tuning.

QKV M=8 SASS confirms identical static HMMA counts and removal of input
LDGSTS/LDSM/DEPBAR instructions in all-direct mode. Nsight shows essentially
unchanged DRAM bytes and achieved occupancy (16.65%) in the three Humming
paths; all-direct increases registers/thread from 92 to 126. Global-memory
dependency stalls increase in both direct variants. Freeing shared capacity
alone does not increase the work available to run concurrently.

44 shapes (four projections, M=1/2/4/8/9/16/17/32/48/49/64) use separately
fixed configurations, CUPTI graph timing with two reverse-order 150-sample
rounds, and cold/warm caches. All 880 selected-path reference checks passed
the fixed tolerances. Local artifacts are under
`/home/linxi/.local/state/vllm-qwen-int4/gemv-experiment/`:
`humming-direct-tune/`, `humming-direct-compare/` and `humming-direct-profile/`.
They include raw samples, matched-config comparisons, source snapshots,
correctness records, SASS and Nsight exports. Profile latency is not used
as benchmark latency.

The current target is to beat tuned Humming at M=1/2/4/8 on all four
projections under both cache conditions, while sharing Humming's single
weight and scale layout. `shared_layout.cu` is the current candidate;
`shared_layout_benchmark.py` compares it with Humming using the same parameter
pointers and validates alternating small/large M. Earlier custom-layout
experiments below remain historical comparisons. Down's current measured
advantage is still too small to call robust.

## M=1 contract

- M=1, K divisible by 128, group size 128, symmetric INT4 without act order.
- FP16 activations, scales and output; FP16 dequantized weights, FP32 accumulation.
- Custom weights are contiguous `[N, K/8]` uint32 words. Each word stores eight
  successive K values, least significant nibble first; integer value is nibble−8.
  Scales are contiguous `[N, K/128]`.
- No bias, padding, noncontiguous tensors or multi-token input support.
- Independent-output variants reduce within each warp, without shared memory,
  block barriers, atomics or cross-block reduction. Split-K experiments use
  FP32 global partials and a second kernel, or shared partials and a block barrier.

The loader checks the checkpoint's zero points and group indices before
transposing its packed words and scales. Packing and model loading are outside
the measured operation.

## Run

Requires the repository's CUDA environment, CMake, Ninja, nvcc, CUDA Python
bindings, FlashInfer and safetensors. The harness builds a cubin incrementally
with CMake/Ninja and launches it through the CUDA driver on PyTorch's stream.
This avoids requiring nvcc to match the CUDA version used to build PyTorch.

```bash
.venv/bin/python benchmarks/kernels/w4a16_gemv/benchmark.py \
  --model /absolute/path/to/Qwen2.5-1.5B-Instruct-GPTQ-Int4 \
  --output-dir /tmp/w4a16-gemv

.venv/bin/python benchmarks/kernels/w4a16_gemv/benchmark.py \
  --model /absolute/path/to/Qwen2.5-1.5B-Instruct-GPTQ-Int4 \
  --output-dir /tmp/w4a16-gemv-check --layer-index 14 --check-only
```

The checkpoint directory must contain `model.safetensors`. The default model
path selects the local cached snapshot used for the experiment. Outputs include
`metadata.json`, `correctness.json` and, when timing, `results.csv`.

Correctness uses zero, one and three seeded random inputs against FP32 matmul
of the same FP16 dequantized weights, rounded to FP16. Every variant must pass
relative L2 error below 0.001 and elementwise `atol=0.002, rtol=0.02`.

Timing uses FlashInfer's CUPTI helper with CUDA graphs; without CUPTI the helper
falls back to graph/event timing. Both cold rotating arguments and warm buffers
are measured. Two rounds reverse variant order, each with 100 ms warmup and 60
samples by default. Allocation, compilation, packing, and host dispatch are
excluded. Effective GB/s uses minimum bytes
`K*N*(0.5 + 2/128) + 2*(K+N)`; it is not measured DRAM bandwidth.

For Nsight Compute, run under `ncu --profile-from-start off` with selected
`--profile` variants. The harness enables profiling only around the selected
kernel calls. Profile latency is not used as benchmark latency.

## Initial measured example

RTX 3060 12 GB, SM86, PyTorch 2.13.0+cu132, nvcc 12.9, local repository commit
`b8e9001bf` (see run metadata for the full commit), layer 0, M=1. These are local
graph/event medians, not serving latency. Both Marlin atomic reduction settings
were tested; the table uses the faster setting per shape and cache condition.
Other Marlin launch configurations were not exhaustively tuned.

| Projection | Cache | Marlin µs | Custom µs | Speedup |
| --- | --- | ---: | ---: | ---: |
| QKV, K=1536 N=2048 | Cold | 9.0112 | 6.9632 | 1.29× |
| QKV, K=1536 N=2048 | Warm | 7.2704 | 6.3488 | 1.15× |
| O, K=1536 N=1536 | Cold | 7.6800 | 5.7344 | 1.34× |
| O, K=1536 N=1536 | Warm | 5.7344 | 4.9152 | 1.17× |

The winner is `gemv_vector_w4_o2`: four warps per block, two output channels per
warp, paired half dequantization and 16-byte activation loads.

QKV cold-cache iteration history:

| Change | Median µs |
| --- | ---: |
| Independent outputs, scalar dequantization | 10.6496 |
| Paired half dequantization | 7.5776 |
| Reuse activation across two output channels | 7.1680 |
| Load eight FP16 activations with one 16-byte access | 6.9632 |

Removing cross-block reduction alone did not beat Marlin. Dequantization and
activation instruction costs mattered as well.

Nsight Compute on QKV measured 96 versus 36 registers/thread, about 17% versus
67% achieved occupancy and about 24% versus 70% scheduler issue activity for
Marlin and the final GEMV. Both had zero register spills. The narrower-load
two-output variant had 456,704 theoretical global L2 sectors versus an ideal
161,792; the vector variant matched 161,792. DRAM reads remained about 1.63 MB
for both custom variants, so this is a reduction in load instruction/transaction
overhead, not a comparable reduction in DRAM traffic.

These results motivate testing M=2–8 and integrating an M=1 dispatch into vLLM.
They do not establish end-to-end decode speedup, quality on generated text,
performance on other GPUs, or superiority over all possible Marlin tuning.

## Second optimization experiment

The harness also supports `--variants` to select named candidates. The second
experiment tested FP32 dequantization followed by FP16 rounding, scale broadcast,
8/16-thread output reductions, fixed-K specialization, register prefetch and
independent accumulators. The prefetch variants require `K/8 >= Threads`;
`k1536` variants require K=1536.

The winning change is `gemv_spread_t32`. It pairs nibbles 0/4, 1/5, 2/6 and 3/7
from the original packed word, allowing `(q & 0x000f000f) | 0x64006400` to
construct a half2. This removes the shifts needed to reposition adjacent
nibbles. It preserves FP16 dequantization and FP32 accumulation, while changing
the accumulation order. No new offline weight layout is needed.

Contemporary comparison on the same GPU, layer 0, two rounds of 100 samples:

| Projection | Cache | Previous vector GEMV µs | Spread-pair GEMV µs | Latency reduction |
| --- | --- | ---: | ---: | ---: |
| QKV | Cold | 7.0656 | 6.6560 | 5.8% |
| QKV | Warm | 6.4512 | 5.5296 | 14.3% |
| O | Cold | 5.9392 | 5.6320 | 5.2% |
| O | Warm | 5.1200 | 4.4032 | 14.0% |

```bash
.venv/bin/python benchmarks/kernels/w4a16_gemv/benchmark.py \
  --output-dir /tmp/w4a16-spread --repeats 100 \
  --variants marlin_atomic0 marlin_atomic1 gemv_vector_w4_o2 gemv_spread_t32
```

Nsight on QKV measured 909,312 versus 747,520 executed warp instructions, with
LOP3 instructions dropping from 197,632 to 99,328. Registers remained at 36,
DRAM reads remained about 1.63 MB and both variants had zero spills. Scheduler
issue activity fell from about 71% to 66% despite the lower latency: fewer
issued instructions can be beneficial. Issue activity is not an optimization
objective by itself.

The negative experiments are useful as well:

- FP32 dequantization and scale broadcast increased executed instructions and
  did not improve latency. The FP32 candidate explicitly retains the intermediate
  FP16 rounding to keep the weight contract comparable.
- 16/8 threads per output reduced QKV launch size from 256 to 128/64 blocks and
  achieved occupancy from about 65% to 35%/20% for the two-output variants. Fewer
  reduction instructions did not offset the reduced concurrency.
- Source-level prefetch did not beat the spread-pair variant. SASS placed the
  main next-iteration loads after the current FMAs, defeating the intended
  overlap. This does not rule out other software pipeline designs.
- Independent accumulators increased register and instruction costs without a
  consistent win. K=1536 specialization also failed to beat the generic winner
  consistently.

## Applying the GEMV to MLP

Select `--layers gate_up_proj down_proj` to test fused gate/up weights
(K=1536, N=17920) and down weights (K=8960, N=1536). The default layers remain
QKV/O. Automatic variant selection skips K=1536 specializations for down.

After scanning 4/8 warps, 1/2/4 outputs per warp and alternative thread group
sizes, the best tested configurations were four outputs per warp for gate/up
and two for down, both using spread-pair dequantization and four warps/block.
The table compares against the faster of Marlin's two atomic settings in a
confirmation run with two rounds of 150 samples:

| Projection | Cache | Marlin µs | Custom µs | Latency reduction | Custom effective GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Gate/up | Cold | 47.3088 | 44.3392 | 6.3% | 321.0 |
| Gate/up | Warm | 47.4112 | 44.3392 | 6.5% | 321.0 |
| Down | Cold | 25.7024 | 25.4976 | 0.8% | 279.1 |
| Down | Warm | 25.4976 | 25.1904 | 1.2% | 282.5 |

Warm here means reused buffers; these MLP weights exceed L2 capacity and do not
become fully cache-resident. The roughly 1% down difference does not establish
a compelling replacement for Marlin.

Nsight measured gate/up DRAM throughput at about 83% for Marlin and 90% for the
custom kernel, with approximately equal read bytes. Down measured about 78%
versus 80%. Four outputs per warp reduced down's grid to 96 blocks and achieved
occupancy to 27%, compared with 192 blocks and 53% for two outputs, making it
slower. The shapes benefit from different configurations.

```bash
.venv/bin/python benchmarks/kernels/w4a16_gemv/benchmark.py \
  --output-dir /tmp/w4a16-mlp --layers gate_up_proj down_proj --repeats 150 \
  --variants marlin_atomic0 marlin_atomic1 gemv_spread_t32 gemv_spread_w4_o4
```

These measurements exclude SiLU, elementwise gate multiplication, bias and the
rest of the model. They are not measurements of a fused MLP or full decode.

## Down split-K experiment

Two split-K approaches partition K at quantization-group boundaries, including
uneven group counts. Each partial sum is FP32 and the output is rounded to
FP16 only after combining all partials:

- `gemv_split*` launches a two-dimensional grid and writes `[Splits, N]` global
  partials, followed by `gemv_reduce_split`. The benchmark includes both kernels;
  scratch allocation is outside timing and no scratch clearing is required.
- `gemv_local*` assigns K partitions to warps in the same block, writes small
  shared-memory partials and combines them after a block barrier. These variants
  require N divisible by `Warps*Outputs` so that every block reaches the barrier.
  The launcher enforces this contract.

The sweep tested global splits 2/4/8/16, 1/2 outputs and 4/8 warps; local splits
2/4/8 with 1/2 outputs and 2/4 logical output warps. Neither approach clearly
beat the previous two-output GEMV on K=8960, N=1536. A four-way, one-output global
split measured about 26.1 µs including reduction versus about 25.0 µs unsplit.
The best local four-way one-output variant roughly tied at 25.0 µs.

Nsight measured achieved occupancy rising from about 62% unsplit to 88% for
global partials and 92% for the best local split, but DRAM throughput only rose
from about 79% to 81%. More occupancy did not translate into a substantial
bandwidth increase. Global reduction adds another kernel; local reduction adds
shared-memory operations and a block barrier. These costs offset the small
benefit. This experiment does not support reduced concurrency as the sole
explanation of Down performance.

```bash
.venv/bin/python benchmarks/kernels/w4a16_gemv/benchmark.py \
  --output-dir /tmp/down-split --layers down_proj --repeats 200 \
  --variants marlin_atomic0 marlin_atomic1 gemv_spread_t32 \
  gemv_spread_w4_o1 gemv_split4_w4_o1 gemv_local4_w4_o1
```

## Batched comparison with current Humming

`batched.cu` extends independent-output GEMV to M=1/2/4/8/16/32. Each warp
computes a tile of 1/2/4/8 input rows and 1/2 output columns, reusing unpacked
weights across the input rows. This is one tiled kernel, not repeated host calls
to the M=1 kernel. Row tiles may span multiple blocks when M exceeds eight.
The benchmark also retains the original M=1 candidate for comparison.

`batch_benchmark.py` scans row tiles, output columns, 4/8 warps and selected
16-thread groups. Marlin scans both atomic settings; Humming scans its default,
alternative stage count and disabled Stream-K. Configurations are selected
using cold-cache tuning, then fixed for a separate comparison under both cache
conditions. Default library configurations remain in the raw comparison data.
Invalid optional configurations are recorded and excluded; a missing valid
family or failed selected configuration aborts the run.

The Humming checkout used in this experiment is upstream main at
[`81402fe03040e1233afeea88f3420db7ba01302d`](https://github.com/vllm-project/humming/commit/81402fe03040e1233afeea88f3420db7ba01302d),
installed separately as `0.1.18.post6+g81402fe03`. The latest tagged release
checked at experiment start was v0.1.18; the preexisting environment's 0.1.17
was not replaced. Specify both the isolated package root and source checkout:

```bash
LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
  benchmarks/kernels/w4a16_gemv/batch_benchmark.py \
  --humming-root /absolute/path/to/humming-package \
  --humming-source /absolute/path/to/humming-source \
  --output-dir /tmp/batch-tune --phase tune --repeats 40

LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
  benchmarks/kernels/w4a16_gemv/batch_benchmark.py \
  --humming-root /absolute/path/to/humming-package \
  --humming-source /absolute/path/to/humming-source \
  --output-dir /tmp/batch-compare --phase compare --repeats 150 \
  --selection /tmp/batch-tune/selection.json
```

On WSL, `LIBRARY_PATH` lets Humming's device-info extension link the driver
library. Builds use CMake/Ninja and the driver API as in the M=1 harness.
`--phase check --layer-index 14 --selection ...` checks other layers without
timing; `--phase profile` enables Nsight profiling around selected calls only.
`--resume` resumes completed shape cases in an interrupted tuning run.

The batched accuracy gate is common to all families: relative L2 below 0.001
and elementwise `rtol=0.02, atol=max(0.002, 0.002*reference_RMS)`. It records
maximum absolute error, RMS-normalized maximum error and whether the original
fixed `atol=0.002, rtol=0.02` check passed. RMS scaling avoids overinterpreting
relative error at near-zero outputs; relative L2 is not relaxed. Marlin's
atomic setting may reduce FP16 partials even with FP32 MMA accumulation, so
bitwise equality with the FP32 reference is not expected. Failed optional
tuning settings and the original failed experiments remain in the artifacts.
Only configurations passing the original fixed elementwise tolerance may be
selected. All final selected paths passed that original check on layers 0, 14
and 27: 1,080 comparisons over four projections and six M values. Wider-gate
default measurements are retained for transparency, not promoted to winners.

Measured cold-cache custom/Marlin/Humming main latency in microseconds:

| M | QKV | O | Gate/up | Down |
| --- | --- | --- | --- | --- |
| 1 | 6.66 / 9.01 / 8.09 | 5.63 / 7.68 / 6.76 | 44.44 / 47.00 / 48.54 | 25.60 / 26.32 / 23.96 |
| 2 | 8.19 / 9.06 / 8.09 | 7.17 / 7.78 / 6.76 | 53.15 / 47.21 / 48.64 | 33.79 / 26.01 / 23.96 |
| 4 | 10.55 / 9.11 / 8.09 | 8.19 / 7.78 / 6.76 | 72.60 / 47.82 / 49.05 | 37.27 / 26.21 / 24.06 |
| 8 | 16.90 / 9.22 / 8.19 | 13.41 / 7.99 / 6.86 | 123.80 / 48.44 / 49.87 | 61.54 / 26.73 / 24.37 |
| 16 | 29.49 / 11.16 / 10.34 | 22.68 / 9.01 / 7.68 | 254.62 / 54.37 / 52.94 | 116.79 / 29.90 / 27.03 |
| 32 | 55.30 / 17.61 / 14.75 | 42.39 / 16.08 / 9.83 | 563.81 / 80.23 / 79.97 | 230.14 / 46.28 / 43.32 |

The custom path has clear wins at M=1 for QKV/O and gate/up. QKV M=2 is close
to Humming; M>=4 favors mature GEMM paths. Humming wins down even at M=1.
Gate/up favors Marlin at M=2/4/8; both libraries are close at M=32. These are
fixed tuned configurations, not claims of exhaustive library tuning. Default
library configurations and warm-cache results remain in the raw CSV files.

Packing, allocation, compilation and host dispatch are excluded from timing.
All implementations operate on identical quantized weights and FP16 inputs;
the reference uses FP32 matmul of the same FP16 dequantized weights. This is
not an original-model quantization evaluation or a full vLLM decode benchmark.

## Revising the target to M=1/2/4/8 against Humming

A fresh comparison reproduced the old crossover: QKV M=2 is approximately
at parity (8.192 vs 8.090 µs), and M=4 clearly loses. O and Gate/up first lose
at sampled M=2; Down already loses at M=1. Small differences at QKV M=2 do
not establish a robust winner.

The old batched kernel still executes FP32 scalar FMAs for every input row.
QKV profiler data shows 196,608 warp FFMA instructions at M=2 versus 786,432
at M=8, and registers increasing from 44 to 96. Humming's
[`base.py`](https://github.com/vllm-project/humming/blob/81402fe03040e1233afeea88f3420db7ba01302d/humming/tune/base.py)
selects an eight-row Tensor Core tile for M<=8. Its
[`wmma.cuh`](https://github.com/vllm-project/humming/blob/81402fe03040e1233afeea88f3420db7ba01302d/humming/include/humming/mma/wmma.cuh)
dequantizes weight fragments and reuses them across the tile. Consequently,
its small-M latency stays nearly flat.

`tensorcore.cu` instead computes W[16,16] @ X.T[16,8] with
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. The same kernel family
covers M=1/2/4/8, filling unused input fragments with zeros. Weights are
repacked offline into MMA fragment order, retaining the same INT4 values and
FP16 scales. Accumulation and block-local K reduction remain FP32.

The experiments test scalar versus 16-byte weight loads, K partitions,
activation reuse across 16/32/64 output columns per warp, and L1 bypass for
streamed weights. These are configurations of one kernel family. No runtime
weight repacking, global atomics or extra reduction kernel is required.

Independent confirmation on the same RTX 3060, layer 0, two reverse-order
rounds of 150 samples; cold-cache revised custom / Humming latency in µs:

| M | QKV | O | Gate/up | Down |
| --- | --- | --- | --- | --- |
| 1 | 6.55 / 8.09 | 5.63 / 6.76 | 43.72 / 48.33 | 23.24 / 23.86 |
| 2 | 6.66 / 8.09 | 5.73 / 6.76 | 43.93 / 48.64 | 23.45 / 23.86 |
| 4 | 6.86 / 8.19 | 6.04 / 6.86 | 44.34 / 48.85 | 23.65 / 24.06 |
| 8 | 7.68 / 8.29 | 6.35 / 6.86 | 45.36 / 49.66 | 24.27 / 24.27 |

QKV/O use `tensorv_n2_k4`; Gate/up uses `tensorv_n1_k8` for M<=4 and
`tensorv_n2_k4` at M=8. Down uses `tensorw2_n1_k8` for M<=2,
`tensorc2_n1_k16` at M=4 and `tensorc2_n1_k8` at M=8. The `w2` and `c2`
variants reuse activations across two output tiles; `c2` also bypasses L1
for weight loads. All configurations were selected on cold-cache tuning and
then held fixed for cold and warm confirmation. Humming uses the same pinned
commit and tuning range described above; its search is not exhaustive.

QKV/O/Gate-up also beat Humming with warm buffers. Down M=1/2/4 only has a
small measured advantage; M=8 ties under both cache conditions and in both
timing rounds. **The target is not yet fully met.** CUPTI is absent, so these
are FlashInfer CUDA graph/event measurements. The small Down margins need
stronger evidence before being considered robust.

Down profiling measured approximately 83% DRAM throughput for both the wide
custom kernel and Humming. Raising K warps to 16/32 alone made it slower;
reusing activations across two output tiles helped, while four tiles did
not. L1 bypass did not establish a robust M=8 win. The next experiment should
test staged asynchronous loads for the long K=8960 reduction, with bandwidth
and instruction counts as evidence.

Enable these candidates with `--tensorcore`; `--variants` restricts custom
candidate names while retaining library candidates. Use `--batches 1 2 4 8`
for the target range, tune first, and pass its `selection.json` to a separate
`--phase compare` run as in the batched example. The same selected paths
passed the original fixed tolerance on layers 0, 14 and 27: 720 comparisons
across three families, four projections, four M values and five input cases.
The maximum relative L2 error was below 0.000421. Pre-commit checks passed.

Detailed local artifacts are in
`/home/linxi/.local/state/vllm-qwen-int4/gemv-experiment/`: `crossover-report.md`,
`crossover-summary.csv`, `crossover.png`, `tensor-final-selection.json`,
`tensor-final-compare/`, `tensor-final-check14/`, `tensor-final-check27/`, and
`tensor-down-profile.ncu-repz`. They include commands, raw samples, per-round
medians, source hashes, correctness errors, and unsuccessful tuning settings.

## Isolating the weight-layout benefit and packing cost

`repack_benchmark.py` compares the Tensor Core path above with a native-layout
MMA path that reads the original custom `[N,K/8]` words and `[N,K/128]` scales.
Both execute the same MMA and FP32 block-local K reduction. A matched control
uses identical output tiles, warp counts and cache policy; a separate sweep
of 16 native-layout configurations selects a tuned control. Every matched
control produced bitwise-identical output to its repacked counterpart.

Independent cold-cache confirmation at M=8, two reverse-order rounds of
150 samples, with configurations fixed before comparison:

| Projection | Tuned native µs | Repacked µs | Latency reduction |
| --- | ---: | ---: | ---: |
| QKV | 10.75 | 7.68 | 28.6% |
| O | 8.70 | 6.45 | 25.9% |
| Gate/up | 65.95 | 45.36 | 31.2% |
| Down | 41.68 | 24.37 | 41.5% |

Across all M=1/2/4/8, cold-cache reductions versus tuned native MMA range
from 14.6% to 41.5%. This includes the benefit of preparing MMA fragments,
avoiding runtime extraction/gather and enabling packed loads. It does not
measure acceleration of the original CUDA Core GEMV or isolate a single load
instruction. The native-layout tuning search is not exhaustive.

The existing `tensor_pack` helper copies weights to CPU, rearranges with
NumPy and copies the packed result to GPU; scales are rearranged on GPU.
Seven warmed wall-time repetitions, including allocation and synchronization:

| Projection | CPU helper ms | GPU helper ms | GPU packing kernel µs |
| --- | ---: | ---: | ---: |
| QKV | 13.49 | 0.234 | 13.21 |
| O | 9.34 | 0.223 | 9.73 |
| Gate/up | 165.58 | 0.566 | 117.15 |
| Down | 85.02 | 0.308 | 50.59 |

The new `repack_vector` CUDA kernel creates bitwise-identical packed weights
and scales entirely on GPU, so CPU round trips are not required. The GPU
helper column includes allocation, Python/driver launch and synchronization;
the GPU kernel column excludes them and uses preallocated output. CPU helper
cost describes the current NumPy implementation, not a CPU lower bound.
The existing batched comparison still defaults to the CPU helper; the GPU
converter is exposed by this experiment.

Packing must happen once and be amortized over repeated calls. For example,
QKV M=2 saves about 2.15 µs/call: the measured CPU helper requires about
6,272 calls to amortize, versus 109 calls for the GPU helper including host
overhead. These are calls to the same projection, not generated tokens.

```bash
.venv/bin/python benchmarks/kernels/w4a16_gemv/repack_benchmark.py \
  --output-dir /absolute/path/to/repack-compare \
  --selection /absolute/path/to/tensor-final-selection.json \
  --repeats 150 --tune-repeats 30 --pack-repeats 7
```

All 1,360 native/repacked correctness checks on layer 0 passed the original
fixed elementwise gate and relative L2 threshold; worst relative L2 was below
0.000094. Full cold/warm results, matched controls, phase timings, all samples,
and amortization estimates are in the local artifact directory
`/home/linxi/.local/state/vllm-qwen-int4/gemv-experiment/repack-compare/`.

## Current candidate: one parameter layout shared with Humming

`shared_layout.cu` directly consumes Humming's transformed INT4 weight and
FP16 scale tensors. It neither creates a custom layout nor converts weights
during forward. Original custom/Marlin parameter tensors are released after
initial Humming preparation. The benchmark keeps its dense reference on CPU,
so it retains only one quantized weight/scale allocation per projection.

This implementation targets the pinned Humming commit above, SM86 dense
FP16/INT4 group128 without act order or zero points, with matching padded and
logical sizes. It rejects raw/flat/native-dequant/packed-K/INT-scale layouts.
Humming's initial parameter preparation can still use temporary buffers.

The physical INT4 words are `[K/16, N/64, lane, four N tiles]`; the scales
use Humming's per-64-channel permutation. The kernel maps that order directly
to MMA registers. It tests scalar/8-byte/16-byte weight loads, paired/vector
scale loads, activation reuse across output tiles, K warp counts and L1
bypass. K partition sizes 16/32/64/128 improve the balance of work assigned
to warps without changing quantization groups or the stored scale layout.
Forty-five configurations were tested across two CUPTI tuning passes.

Independent confirmation on RTX 3060, layer 0, cold-cache custom / Humming
latency in µs; configurations were fixed before comparison:

| M | QKV | O | Gate/up | Down |
| --- | --- | --- | --- | --- |
| 1 | 8.57 / 9.54 | 6.97 / 7.84 | 48.60 / 52.34 | 27.42 / 27.97 |
| 2 | 8.86 / 9.57 | 7.20 / 7.85 | 49.06 / 52.44 | 27.74 / 28.03 |
| 4 | 9.02 / 9.50 | 7.39 / 7.87 | 49.54 / 52.67 | 27.96 / 28.12 |
| 8 | 9.21 / 9.60 | 7.49 / 7.94 | 50.01 / 53.28 | 28.22 / 28.38 |

Warm-cache measurements also favor the custom path at the four requested M
values. Down M=8 cold only saves about 0.16 µs (0.6%), so this does not
establish a robust win across runs or hardware. Humming tuning covers default,
stage-2 and no-Stream-K settings, not every possible library configuration.

These measurements use CUPTI 13.4 with FlashInfer graph replay, 100 ms warmup,
and two reverse-order rounds of 150 samples. Cold calls flush a separate
2x-L2 buffer before the measured kernel. No rotating weight copies are used.
Initial transformation, allocation, compilation, host dispatch and flushing
are outside recorded kernel durations. The isolated CUPTI package is separate
from the original environment. This protocol differs from the earlier
graph/event fallback with rotating arguments; absolute times should not be
compared directly across those experiments.

The same pointer pair is passed to both implementations. The sequence
M=1/16/2/32/4/16/8/32 checks the custom path at M<=8 and Humming at M=16/32,
with no conversion and unchanged parameter pointers/versions. Across layers
0/14/27, 600 selected-path reference comparisons and 96 alternating calls
passed the original fixed tolerance and relative L2 gate. Maximum relative
L2 was below 0.000392. This remains a standalone prototype, not an integrated
vLLM quantization backend.

```bash
LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
  benchmarks/kernels/w4a16_gemv/shared_layout_benchmark.py \
  --cupti-root /absolute/path/to/cupti-package \
  --humming-root /absolute/path/to/humming-package \
  --humming-source /absolute/path/to/humming-source \
  --output-dir /absolute/path/to/shared-compare \
  --phase compare --repeats 150 \
  --selection /absolute/path/to/shared-step-tune/selection.json
```

For tuning, omit `--selection` and use `--phase tune`. For other-layer checks,
use `--phase check --layer-index 14` or `27` with a separate output directory.
Detailed local artifacts are in
`/home/linxi/.local/state/vllm-qwen-int4/gemv-experiment/shared-final-compare/`,
including `report.md`, `summary.csv`, raw samples, source snapshots, parameter
pointer records and alternating-dispatch checks. `shared-cupti-tune/` and
`shared-step-tune/` preserve the tuning evidence; `shared-check14/` and
`shared-check27/` preserve the other-layer checks.

### QKV row reuse and shared-memory ablation

The forced-custom M=1–32 comparison initially reused the M=8 configuration
above that boundary. A new sweep finds that QKV M=9–16 benefits from eight
K warps instead of sixteen. For M=17–32, `reuse2_t2_k4_s4` uses four K warps;
each warp decodes a weight fragment once and applies it to two 8-row tiles.
The block covers 16 rows, with the same Humming parameter pointers and one
shared FP32 split-K reduction. This adds neither parameter copies nor repacking.

Independent RTX 3060 layer-0 confirmation, cold L2 latency in microseconds:

| M | Previous custom | New custom | Tuned Humming |
| --- | --- | --- | --- |
| 8 | 9.19 | 9.19 | 9.67 |
| 9 | 13.44 | 9.85 | 11.30 |
| 16 | 14.05 | 10.53 | 11.36 |
| 17 | 19.48 | 13.02 | 15.83 |
| 32 | 23.26 | 14.50 | 16.32 |

The new QKV path is faster at every integer M=1–32 in both cold and warm
confirmation measurements; minimum speedups are 1.036x and 1.023x. This is a
QKV result on this GPU and checkpoint, not a claim for every projection.
The M=1 cold baseline had inconsistent round medians and was repeated;
the report retains both raw runs and identifies the repeat used in its tables.

The QKV M=1–8 ablation keeps the output tiling, K partition and MMA work
fixed within each pair. It compares direct weight loads, the exact direct
kernel with equal extra shared memory reserved, immediate `cp.async` staging,
and staging with one-step lookahead. Scales and activations remain direct.
With eight K warps at M=8, direct/reserved/lookahead take 9.37/9.37/10.24 us.
With sixteen K warps, they take 9.15/9.15/13.63 us. SASS confirms added
LDGSTS/LDS/DEPBAR instructions and unchanged HMMA counts; register counts
also change. All variants still use shared memory for output reduction.

Bypassing this weight-staging path helps this implementation, but these
numbers do not isolate shared-memory latency from instruction/synchronization
and register effects. The experiment does not reproduce Humming's full
pipeline or establish that staging explains its entire performance gap.

Use `--max-custom-m 32 --layers qkv_proj` to enable the extended comparison.
Tune with `--variants` and `--phase tune`; fix the resulting `--selection`
for confirmation. `--reference-selection` also measures the previous
configurations in that same run. The `staged_*`, `pipelined_*` and `reserved_*`
variants expose the ablation controls.

Artifacts under
`/home/linxi/.local/state/vllm-qwen-int4/gemv-experiment/`:
`shared-qkv-reuse-tune/`, `shared-qkv-reuse-compare/`,
`shared-qkv-smem-ablation/`, and `shared-qkv-reuse-check14/`.
The comparison contains the complete 32-point tables, plots, configuration
choices and source snapshots. Its 320 selected new/Humming checks and all
320 custom ablation checks passed the fixed tolerances; another 40 selected
checks passed on layer 14 at M=9/16/17/32.
