# DeepSeek V4.1 Flash projection shape simulation

Synthetic symmetric INT4 group128, FP16 activations/dequant/output, FP32 accumulation, TP=1, RTX 3060. No checkpoint weights were downloaded.

Shape definitions follow the [official inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py). Expert gate/up is the concatenation of w1/w3; o_a_group uses the actual block-diagonal group dimensions.

Tuning sweeps five existing custom configurations (K warps, output width and row reuse) and Humming default/stage-2/no-Stream-K variants where applicable. This is a coarse sweep, not an exhaustive search.

Configurations were selected in a separate tuning run. Confirmation uses 2 rounds with reversed implementation order, 100 ms warmup and 150 CUPTI graph samples per implementation/cache/shape/bias case. The reported value is the median of round medians. Allocation, initial weight transform, compilation, and cold-L2 flush are excluded.

Both implementations read the same Humming weight/scale allocation and use the same transformed nonzero bias in the fused-bias comparison. No extra bias-add kernel is included in either main comparison arm.

Correctness: 384 comparisons against independent FP32 GEMM of decoded FP16 weights; relative L2 < 0.001 and elementwise atol=max(0.002, 0.002*reference_RMS), rtol=0.02. Max relative L2: 0.000541768.

## Fused bias, cold L2

Each cell is custom / Humming in microseconds, followed by speedup.

| Projection (K × N) | M=1 | M=8 | M=32 | M=64 |
| --- | ---: | ---: | ---: | ---: |
| qwen_qkv (1536 × 2048) | 9.11 / 9.77 (1.073×) | 9.74 / 9.92 (1.019×) | 15.28 / 16.32 (1.068×) | 28.85 / 25.55 (0.886×) |
| qwen_o (1536 × 1536) | 7.65 / 7.90 (1.033×) | 8.27 / 8.01 (0.968×) | 12.88 / 10.59 (0.823×) | 22.23 / 19.62 (0.883×) |
| qwen_gate_up (1536 × 17920) | 50.34 / 52.72 (1.047×) | 52.92 / 53.60 (1.013×) | 125.54 / 81.50 (0.649×) | 252.60 / 152.24 (0.603×) |
| qwen_down (8960 × 1536) | 27.92 / 28.02 (1.003×) | 29.02 / 28.38 (0.978×) | 57.98 / 43.68 (0.753×) | 114.03 / 81.80 (0.717×) |
| v41_q_a (5120 × 1280) | 15.79 / 16.09 (1.019×) | 17.62 / 16.37 (0.929×) | 26.96 / 26.31 (0.976×) | 49.83 / 45.51 (0.913×) |
| v41_q_b (1280 × 32768) | 75.48 / 74.79 (0.991×) | 76.65 / 76.55 (0.999×) | 194.70 / 120.37 (0.618×) | 399.55 / 233.72 (0.585×) |
| v41_kv (5120 × 512) | 10.16 / 11.16 (1.098×) | 11.18 / 11.33 (1.013×) | 17.89 / 15.77 (0.881×) | 25.47 / 25.09 (0.985×) |
| v41_o_a_group (4096 × 1024) | 12.25 / 12.34 (1.007×) | 13.26 / 12.59 (0.949×) | 21.13 / 21.24 (1.005×) | 35.98 / 33.47 (0.930×) |
| v41_o_b (8192 × 5120) | 70.50 / 70.99 (1.007×) | 70.97 / 71.33 (1.005×) | 210.83 / 120.91 (0.573×) | 407.55 / 222.78 (0.547×) |
| v41_expert_gate_up (5120 × 4608) | 42.71 / 43.52 (1.019×) | 43.26 / 43.71 (1.010×) | 127.24 / 72.42 (0.569×) | 234.80 / 133.30 (0.568×) |
| v41_expert_down (2304 × 5120) | 24.27 / 25.35 (1.045×) | 24.77 / 25.48 (1.029×) | 56.66 / 42.46 (0.749×) | 108.44 / 73.15 (0.674×) |
| v41_index_q_b (1280 × 4096) | 12.86 / 13.95 (1.085×) | 13.40 / 14.04 (1.047×) | 26.65 / 20.73 (0.778×) | 48.21 / 38.18 (0.792×) |

## Scope

The official model uses native FP8/FP4 and bias-free projections. This experiment borrows its matrix shapes and tests our current W4A16 implementation; it is not a measurement of native model inference.

`v41_o_a_group` is one of eight independent group projections; `v41_expert_gate_up` stacks gate/up for one expert. Routed experts are not merged into a dense MLP. M means tokens assigned to this expert when interpreting the expert rows. No routing, grouped-MoE scheduling, communication, attention, KV cache, or nonlinear operations are timed.

GB/s in summary.csv uses ideal bytes: K*N/2 + 2*(K/128)*N + 2*M*(K+N), plus 2*N for bias. TFLOPS = 2*M*N*K / elapsed time. These are useful-work metrics, not measured DRAM traffic.

Full cold/warm, bias/no-bias results, round medians and selected configurations are in summary.csv. Raw samples are in samples.json.
