# DeepSeek V4.1 Flash projection shape simulation

Synthetic symmetric INT4 group128, FP16 activations/dequant/output, FP32 accumulation, TP=1, RTX 3060. No checkpoint weights were downloaded.

Shape definitions follow the [official inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py). Expert gate/up is the concatenation of w1/w3; o_a_group uses the actual block-diagonal group dimensions.

Tuning sweeps custom configurations (K warps, output width, row reuse and optional register prefetch) and Humming default/stage-2/no-Stream-K variants where applicable. This is a coarse sweep, not an exhaustive search.

Configurations were selected in a separate tuning run. Confirmation uses 2 rounds with reversed implementation order, 100 ms warmup and 150 CUPTI graph samples per implementation/cache/shape/bias case. The reported value is the median of round medians. Allocation, initial weight transform, compilation, and cold-L2 flush are excluded.

Both implementations read the same Humming weight/scale allocation and use the same transformed nonzero bias in the fused-bias comparison. No extra bias-add kernel is included in either main comparison arm.

Correctness: 234 comparisons against independent FP32 GEMM of decoded FP16 weights; relative L2 < 0.001 and elementwise atol=max(0.002, 0.002*reference_RMS), rtol=0.02. Max relative L2: 0.000542817.

## Fused bias, cold L2

Each cell is custom / Humming in microseconds, followed by speedup.

| Projection (K × N) | M=1 | M=8 | M=32 | M=64 |
| --- | ---: | ---: | ---: | ---: |
| qwen_qkv (1536 × 2048) | 9.11 / 9.70 (1.065×) | 9.75 / 9.83 (1.009×) | 15.14 / 16.27 (1.074×) | 24.61 / 25.46 (1.035×) |
| qwen_o (1536 × 1536) | 7.70 / 7.94 (1.031×) | 8.24 / 8.01 (0.972×) | 12.89 / 10.66 (0.827×) | 20.68 / 19.42 (0.939×) |
| qwen_gate_up (1536 × 17920) | 50.19 / 52.77 (1.052×) | 52.87 / 53.62 (1.014×) | 85.89 / 81.27 (0.946×) | 150.79 / 151.93 (1.008×) |
| qwen_down (8960 × 1536) | 27.95 / 27.96 (1.001×) | 29.00 / 28.40 (0.979×) | 58.45 / 43.56 (0.745×) | 92.47 / 81.81 (0.885×) |
| v41_q_a (5120 × 1280) | 15.78 / 16.01 (1.015×) | 17.73 / 16.28 (0.918×) | 26.43 / 26.24 (0.993×) | 49.67 / 45.29 (0.912×) |
| v41_q_b (1280 × 32768) | 75.28 / 74.86 (0.994×) | 77.24 / 76.52 (0.991×) | 126.11 / 119.57 (0.948×) | 240.71 / 233.34 (0.969×) |
| v41_kv (5120 × 512) | 10.15 / 11.14 (1.098×) | 11.19 / 11.25 (1.006×) | 17.92 / 15.72 (0.877×) | 25.51 / 25.15 (0.986×) |
| v41_o_a_group (4096 × 1024) | 11.91 / 12.33 (1.036×) | 13.28 / 12.61 (0.949×) | 21.19 / 21.20 (1.001×) | 34.19 / 33.47 (0.979×) |
| v41_o_b (8192 × 5120) | 70.43 / 70.87 (1.006×) | 70.90 / 71.20 (1.004×) | 128.40 / 120.75 (0.940×) | 237.50 / 222.32 (0.936×) |
| v41_expert_gate_up (5120 × 4608) | 42.65 / 43.48 (1.019×) | 43.19 / 43.74 (1.013×) | 83.03 / 72.33 (0.871×) | 151.77 / 132.96 (0.876×) |
| v41_expert_down (2304 × 5120) | 24.24 / 25.34 (1.045×) | 24.81 / 25.46 (1.026×) | 42.23 / 42.39 (1.004×) | 75.29 / 73.03 (0.970×) |
| v41_index_q_b (1280 × 4096) | 12.86 / 14.00 (1.089×) | 13.47 / 14.05 (1.042×) | 21.94 / 20.72 (0.945×) | 42.60 / 37.94 (0.891×) |

## Previous custom versus optimized custom, cold L2

Both custom versions and Humming were measured in the same run.

| Shape | M | Previous (µs) | Optimized (µs) | Previous / optimized |
| --- | ---: | ---: | ---: | ---: |
| qwen_qkv | 1 | 9.11 | 9.11 | 1.000× |
| qwen_qkv | 8 | 9.75 | 9.75 | 1.000× |
| qwen_qkv | 32 | 15.14 | 15.14 | 1.000× |
| qwen_qkv | 64 | 28.91 | 24.61 | 1.175× |
| qwen_o | 1 | 7.70 | 7.70 | 1.000× |
| qwen_o | 8 | 8.24 | 8.24 | 1.000× |
| qwen_o | 32 | 12.89 | 12.89 | 1.000× |
| qwen_o | 64 | 22.57 | 20.68 | 1.091× |
| qwen_gate_up | 1 | 50.19 | 50.19 | 1.000× |
| qwen_gate_up | 8 | 52.87 | 52.87 | 1.000× |
| qwen_gate_up | 32 | 122.67 | 85.89 | 1.428× |
| qwen_gate_up | 64 | 253.45 | 150.79 | 1.681× |
| qwen_down | 1 | 27.95 | 27.95 | 1.000× |
| qwen_down | 8 | 29.00 | 29.00 | 1.000× |
| qwen_down | 32 | 58.45 | 58.45 | 1.000× |
| qwen_down | 64 | 114.18 | 92.47 | 1.235× |
| v41_q_a | 1 | 15.78 | 15.78 | 1.000× |
| v41_q_a | 8 | 17.73 | 17.73 | 1.000× |
| v41_q_a | 32 | 26.43 | 26.43 | 1.000× |
| v41_q_a | 64 | 49.67 | 49.67 | 1.000× |
| v41_q_b | 1 | 75.89 | 75.28 | 1.008× |
| v41_q_b | 8 | 77.24 | 77.24 | 1.000× |
| v41_q_b | 32 | 188.52 | 126.11 | 1.495× |
| v41_q_b | 64 | 390.32 | 240.71 | 1.622× |
| v41_kv | 1 | 10.15 | 10.15 | 1.000× |
| v41_kv | 8 | 11.19 | 11.19 | 1.000× |
| v41_kv | 32 | 17.92 | 17.92 | 1.000× |
| v41_kv | 64 | 25.51 | 25.51 | 1.000× |
| v41_o_a_group | 1 | 12.18 | 11.91 | 1.023× |
| v41_o_a_group | 8 | 13.28 | 13.28 | 1.000× |
| v41_o_a_group | 32 | 21.19 | 21.19 | 1.000× |
| v41_o_a_group | 64 | 35.90 | 34.19 | 1.050× |
| v41_o_b | 1 | 70.43 | 70.43 | 1.000× |
| v41_o_b | 8 | 70.90 | 70.90 | 1.000× |
| v41_o_b | 32 | 210.79 | 128.40 | 1.642× |
| v41_o_b | 64 | 406.03 | 237.50 | 1.710× |
| v41_expert_gate_up | 1 | 42.76 | 42.65 | 1.003× |
| v41_expert_gate_up | 8 | 43.19 | 43.19 | 1.000× |
| v41_expert_gate_up | 32 | 125.97 | 83.03 | 1.517× |
| v41_expert_gate_up | 64 | 234.49 | 151.77 | 1.545× |
| v41_expert_down | 1 | 24.24 | 24.24 | 1.000× |
| v41_expert_down | 8 | 24.82 | 24.81 | 1.000× |
| v41_expert_down | 32 | 57.07 | 42.23 | 1.351× |
| v41_expert_down | 64 | 106.72 | 75.29 | 1.417× |
| v41_index_q_b | 1 | 12.86 | 12.86 | 1.000× |
| v41_index_q_b | 8 | 13.37 | 13.47 | 0.992× |
| v41_index_q_b | 32 | 27.25 | 21.94 | 1.242× |
| v41_index_q_b | 64 | 48.36 | 42.60 | 1.135× |

## Scope

The official model uses native FP8/FP4 and bias-free projections. This experiment borrows its matrix shapes and tests our current W4A16 implementation; it is not a measurement of native model inference.

`v41_o_a_group` is one of eight independent group projections; `v41_expert_gate_up` stacks gate/up for one expert. Routed experts are not merged into a dense MLP. M means tokens assigned to this expert when interpreting the expert rows. No routing, grouped-MoE scheduling, communication, attention, KV cache, or nonlinear operations are timed.

GB/s in summary.csv uses ideal bytes: K*N/2 + 2*(K/128)*N + 2*M*(K+N), plus 2*N for bias. TFLOPS = 2*M*N*K / elapsed time. These are useful-work metrics, not measured DRAM traffic.

Full cold/warm, bias/no-bias results, round medians and selected configurations are in summary.csv. Raw samples are in samples.json.
