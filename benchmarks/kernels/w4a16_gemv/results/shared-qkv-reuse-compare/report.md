# QKV weight reuse and shared-memory ablation

RTX 3060, Qwen2.5-1.5B GPTQ INT4 layer 0 QKV, K=1536 N=2048, group128, FP16 activations/scales/dequantization/output, FP32 accumulation. One shared Humming-layout weight/scale allocation; no runtime repacking.

CUPTI CUDA graphs, cold and warm L2, 100 ms warmup, 150 samples per round, two reversed-order rounds. Allocation, compilation, initial transform, host work, CPU reference and cache flush excluded.

M=1 was repeated separately: the first full run had an unstable selected Humming cold pair (78.867 / 9.535 us). Tables and figures use the complete M=1 repeat from ../shared-qkv-reuse-m1-confirm/ for all arms/caches (selected Humming cold rounds 10.079 / 9.623 us). The original raw run is retained unchanged. No other selected case had >10% separation between round medians.

Optimization: M=1–2 retains K8/S4; M=3–8 retains K16/S8; M=9–16 uses K8/S4 with 8-row blocks; M=17–32 uses K4/S4, with each warp reusing decoded weights for two 8-row tiles and each block covering 16 rows. Configs fixed by a separate sweep before confirmation. The old implementation and the new one are measured in the same confirmation run. Other projections are not claimed optimized.

Ablation: direct global-to-register weight loads vs cp.async global-to-shared, wait, shared-to-register then identical decode/MMA. An immediate-wait variant and a one-step-lookahead variant are both included. Reserved controls launch the exact direct kernel with extra dynamic shared memory equal to staging storage. Scales/activations retain direct loads; output split-K reduction still uses shared memory in every arm.

Limits: this is a controlled change to our kernel, not a faithful recreation of Humming's multi-stage pipeline. The generated register count changes along with the copy path. It tests whether bypassing this staging path helps this implementation; it cannot assign a fraction of the total Humming gap to shared-memory use.

320 selected new/Humming checks and 320 custom ablation checks passed atol=.002, rtol=.02 and relative L2<.001.

## Cold L2 optimization

| M | Old custom us | New custom us | Humming us | Reduction vs old |
| --- | --- | --- | --- | --- |
| 1 | 8.767 | 8.767 | 9.887 | 0.0% |
| 2 | 8.896 | 8.896 | 9.535 | 0.0% |
| 3 | 9.023 | 9.023 | 9.512 | 0.0% |
| 4 | 9.000 | 9.000 | 9.546 | 0.0% |
| 5 | 9.023 | 9.023 | 9.568 | 0.0% |
| 6 | 9.216 | 9.216 | 9.545 | 0.0% |
| 7 | 9.194 | 9.194 | 9.666 | 0.0% |
| 8 | 9.186 | 9.186 | 9.675 | 0.0% |
| 9 | 13.437 | 9.855 | 11.304 | 26.7% |
| 10 | 13.765 | 10.131 | 11.327 | 26.4% |
| 11 | 13.723 | 10.111 | 11.292 | 26.3% |
| 12 | 13.902 | 10.302 | 11.295 | 25.9% |
| 13 | 14.010 | 10.463 | 11.391 | 25.3% |
| 14 | 14.015 | 10.496 | 11.359 | 25.1% |
| 15 | 14.013 | 10.461 | 11.390 | 25.3% |
| 16 | 14.047 | 10.525 | 11.359 | 25.1% |
| 17 | 19.483 | 13.021 | 15.832 | 33.2% |
| 18 | 19.743 | 13.151 | 15.871 | 33.4% |
| 19 | 19.742 | 13.056 | 15.903 | 33.9% |
| 20 | 19.820 | 13.182 | 15.903 | 33.5% |
| 21 | 19.487 | 13.215 | 15.933 | 32.2% |
| 22 | 20.030 | 13.374 | 15.967 | 33.2% |
| 23 | 19.838 | 13.205 | 16.031 | 33.4% |
| 24 | 20.030 | 13.215 | 15.999 | 34.0% |
| 25 | 22.782 | 13.315 | 16.031 | 41.6% |
| 26 | 22.847 | 13.407 | 16.034 | 41.3% |
| 27 | 22.728 | 13.453 | 16.099 | 40.8% |
| 28 | 23.102 | 13.631 | 16.095 | 41.0% |
| 29 | 22.973 | 13.726 | 16.179 | 40.3% |
| 30 | 23.459 | 13.802 | 16.177 | 41.2% |
| 31 | 23.170 | 13.981 | 16.254 | 39.7% |
| 32 | 23.262 | 14.495 | 16.315 | 37.7% |

First Humming win in M=1–32: None.

## Cold L2 ablation

| M | K-warps | Direct us | Reserved us | Immediate staging us | Lookahead staging us | Humming us |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 8.625 | 8.609 | 9.569 | 9.570 | 9.886 |
| 1 | 16 | 8.706 | 8.703 | 12.673 | 12.578 | 9.886 |
| 2 | 8 | 8.888 | 8.874 | 9.733 | 9.730 | 9.665 |
| 2 | 16 | 8.928 | 8.937 | 13.233 | 13.152 | 9.665 |
| 3 | 8 | 8.959 | 8.993 | 9.887 | 9.858 | 9.695 |
| 3 | 16 | 9.023 | 9.014 | 13.282 | 13.346 | 9.695 |
| 4 | 8 | 9.088 | 9.120 | 9.890 | 9.889 | 9.695 |
| 4 | 16 | 9.088 | 9.088 | 13.407 | 13.375 | 9.695 |
| 5 | 8 | 9.245 | 9.218 | 10.018 | 10.049 | 9.697 |
| 5 | 16 | 9.120 | 9.121 | 13.538 | 13.538 | 9.697 |
| 6 | 8 | 9.278 | 9.277 | 10.207 | 10.175 | 9.725 |
| 6 | 16 | 9.245 | 9.212 | 13.663 | 13.723 | 9.725 |
| 7 | 8 | 9.309 | 9.279 | 10.248 | 10.143 | 9.692 |
| 7 | 16 | 9.213 | 9.213 | 13.599 | 13.653 | 9.692 |
| 8 | 8 | 9.374 | 9.373 | 10.240 | 10.237 | 9.695 |
| 8 | 16 | 9.151 | 9.152 | 13.598 | 13.628 | 9.695 |

## Warm L2 optimization

| M | Old custom us | New custom us | Humming us | Reduction vs old |
| --- | --- | --- | --- | --- |
| 1 | 5.440 | 5.440 | 7.071 | 0.0% |
| 2 | 6.015 | 6.015 | 7.135 | 0.0% |
| 3 | 6.111 | 6.111 | 7.071 | 0.0% |
| 4 | 6.047 | 6.047 | 7.104 | 0.0% |
| 5 | 6.176 | 6.176 | 7.167 | 0.0% |
| 6 | 6.336 | 6.336 | 7.135 | 0.0% |
| 7 | 6.401 | 6.401 | 7.139 | 0.0% |
| 8 | 6.368 | 6.368 | 7.136 | 0.0% |
| 9 | 10.240 | 8.255 | 9.771 | 19.4% |
| 10 | 10.590 | 8.477 | 9.791 | 20.0% |
| 11 | 10.271 | 8.572 | 9.826 | 16.5% |
| 12 | 10.559 | 8.810 | 9.791 | 16.6% |
| 13 | 10.175 | 8.604 | 9.823 | 15.4% |
| 14 | 10.044 | 8.859 | 9.853 | 11.8% |
| 15 | 10.207 | 8.927 | 9.951 | 12.5% |
| 16 | 10.971 | 8.891 | 9.916 | 19.0% |
| 17 | 14.751 | 11.707 | 13.534 | 20.6% |
| 18 | 15.103 | 11.711 | 13.535 | 22.5% |
| 19 | 15.071 | 11.806 | 13.568 | 21.7% |
| 20 | 15.039 | 11.838 | 13.566 | 21.3% |
| 21 | 15.006 | 11.807 | 13.628 | 21.3% |
| 22 | 15.169 | 11.872 | 13.663 | 21.7% |
| 23 | 15.255 | 12.119 | 13.663 | 20.6% |
| 24 | 15.560 | 11.959 | 13.663 | 23.1% |
| 25 | 19.294 | 11.934 | 13.695 | 38.1% |
| 26 | 19.226 | 12.387 | 13.699 | 35.6% |
| 27 | 18.853 | 12.291 | 13.731 | 34.8% |
| 28 | 19.531 | 12.415 | 13.784 | 36.4% |
| 29 | 19.546 | 12.667 | 13.796 | 35.2% |
| 30 | 19.428 | 12.627 | 13.796 | 35.0% |
| 31 | 19.486 | 12.991 | 13.887 | 33.3% |
| 32 | 19.650 | 13.603 | 13.918 | 30.8% |

First Humming win in M=1–32: None.

## Warm L2 ablation

| M | K-warps | Direct us | Reserved us | Immediate staging us | Lookahead staging us | Humming us |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 5.440 | 5.409 | 6.242 | 6.242 | 7.106 |
| 1 | 16 | 5.570 | 5.602 | 8.641 | 8.482 | 7.106 |
| 2 | 8 | 5.761 | 5.759 | 6.338 | 6.432 | 7.139 |
| 2 | 16 | 5.983 | 5.952 | 9.183 | 9.022 | 7.139 |
| 3 | 8 | 5.888 | 5.888 | 6.560 | 6.656 | 7.167 |
| 3 | 16 | 6.112 | 6.113 | 9.344 | 9.311 | 7.167 |
| 4 | 8 | 5.794 | 5.794 | 6.881 | 6.818 | 7.172 |
| 4 | 16 | 6.115 | 6.127 | 9.474 | 9.410 | 7.172 |
| 5 | 8 | 5.982 | 5.983 | 6.997 | 6.943 | 7.177 |
| 5 | 16 | 6.174 | 6.176 | 9.691 | 9.627 | 7.177 |
| 6 | 8 | 6.210 | 6.239 | 7.165 | 7.325 | 7.168 |
| 6 | 16 | 6.400 | 6.399 | 9.964 | 9.916 | 7.168 |
| 7 | 8 | 6.080 | 6.080 | 7.392 | 7.336 | 7.135 |
| 7 | 16 | 6.400 | 6.431 | 10.111 | 10.143 | 7.135 |
| 8 | 8 | 6.110 | 6.111 | 7.612 | 7.501 | 7.164 |
| 8 | 16 | 6.397 | 6.397 | 10.235 | 10.331 | 7.164 |
