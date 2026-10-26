# QKV direct-weight-path crossover

RTX 3060; Qwen2.5-1.5B GPTQ INT4 layer0 QKV; K=1536 N=2048; group128; FP16 input/dequantization/output, FP32 accumulation. Same Humming parameter pointers, no runtime repacking. Weights go directly to registers; split-K output reduction still uses shared memory.

Current custom config beyond M=16: reuse2_t2_k4_s4, 4 warps/block, each block covers 16 rows x 32 output channels. The custom kernel is unchanged; only the harness size limit was extended to 64. This measures the current operator, not the limit of all direct-load algorithms.

Humming is tuned per M over default/stage2/no-Stream-K. Configurations were fixed in a separate tuning pass. CUPTI graph replay, 100ms warmup, two reverse-order rounds of 150 samples; boundary M=48–51 independently repeated with 300 samples/round. Allocation, initial transform, compilation, CPU work and cache flush excluded.

First confirmed loss is M=49 under both cold and warm L2. Every integer through M=64 was measured across the previous M=1–32 and new M=33–64 campaigns; M=49–64 all lose. The initial coarse M=49 cold comparison was near parity, so the exact boundary is measurement-dependent; the clean boundary repeat shows 5.8% slower cold and 9.7% slower warm at M=49.

The full M=33–64 run had unstable Humming cold rounds at M=49 (78.328 / 25.326 us). Figures/tables use the complete independent M=48–51 repeat for both kernels and both caches, with the original raw run retained. M=49 repeat cold rounds: custom 26.942 / 27.037 us; Humming 25.508 / 25.503 us.

360 selected-path correctness checks passed atol=.002, rtol=.02 and relative L2<.001 in the extension and boundary confirmation.

The full extension run imported Humming from the pinned source checkout; tuning and boundary confirmation used the isolated installed package. All 151 Python/CUDA/C++ source files in the package were verified byte-identical to the source checkout at commit 81402fe03040e1233afeea88f3420db7ba01302d.

| M | Cold custom us | Cold Humming us | Warm custom us | Warm Humming us |
| --- | --- | --- | --- | --- |
| 1 | 8.767 | 9.887 | 5.440 | 7.071 |
| 2 | 8.896 | 9.535 | 6.015 | 7.135 |
| 3 | 9.023 | 9.512 | 6.111 | 7.071 |
| 4 | 9.000 | 9.546 | 6.047 | 7.104 |
| 5 | 9.023 | 9.568 | 6.176 | 7.167 |
| 6 | 9.216 | 9.545 | 6.336 | 7.135 |
| 7 | 9.194 | 9.666 | 6.401 | 7.139 |
| 8 | 9.186 | 9.675 | 6.368 | 7.136 |
| 9 | 9.855 | 11.304 | 8.255 | 9.771 |
| 10 | 10.131 | 11.327 | 8.477 | 9.791 |
| 11 | 10.111 | 11.292 | 8.572 | 9.826 |
| 12 | 10.302 | 11.295 | 8.810 | 9.791 |
| 13 | 10.463 | 11.391 | 8.604 | 9.823 |
| 14 | 10.496 | 11.359 | 8.859 | 9.853 |
| 15 | 10.461 | 11.390 | 8.927 | 9.951 |
| 16 | 10.525 | 11.359 | 8.891 | 9.916 |
| 17 | 13.021 | 15.832 | 11.707 | 13.534 |
| 18 | 13.151 | 15.871 | 11.711 | 13.535 |
| 19 | 13.056 | 15.903 | 11.806 | 13.568 |
| 20 | 13.182 | 15.903 | 11.838 | 13.566 |
| 21 | 13.215 | 15.933 | 11.807 | 13.628 |
| 22 | 13.374 | 15.967 | 11.872 | 13.663 |
| 23 | 13.205 | 16.031 | 12.119 | 13.663 |
| 24 | 13.215 | 15.999 | 11.959 | 13.663 |
| 25 | 13.315 | 16.031 | 11.934 | 13.695 |
| 26 | 13.407 | 16.034 | 12.387 | 13.699 |
| 27 | 13.453 | 16.099 | 12.291 | 13.731 |
| 28 | 13.631 | 16.095 | 12.415 | 13.784 |
| 29 | 13.726 | 16.179 | 12.667 | 13.796 |
| 30 | 13.802 | 16.177 | 12.627 | 13.796 |
| 31 | 13.981 | 16.254 | 12.991 | 13.887 |
| 32 | 14.495 | 16.315 | 13.603 | 13.918 |
| 33 | 17.885 | 24.169 | 16.977 | 21.180 |
| 34 | 18.639 | 23.485 | 17.888 | 21.214 |
| 35 | 18.543 | 23.485 | 17.981 | 21.279 |
| 36 | 18.703 | 23.454 | 17.630 | 21.246 |
| 37 | 18.974 | 23.550 | 18.303 | 21.310 |
| 38 | 18.743 | 23.517 | 17.464 | 21.342 |
| 39 | 19.262 | 23.582 | 18.043 | 21.374 |
| 40 | 19.770 | 23.582 | 18.846 | 21.374 |
| 41 | 20.030 | 23.359 | 18.934 | 21.199 |
| 42 | 19.959 | 23.386 | 18.968 | 21.217 |
| 43 | 19.807 | 23.374 | 18.945 | 21.279 |
| 44 | 19.967 | 23.391 | 19.102 | 21.341 |
| 45 | 20.245 | 23.490 | 19.140 | 21.340 |
| 46 | 20.453 | 23.486 | 19.780 | 21.406 |
| 47 | 20.798 | 23.500 | 19.908 | 21.446 |
| 48 | 21.064 | 23.710 | 19.951 | 21.712 |
| 49 | 26.989 | 25.508 | 25.598 | 23.327 |
| 50 | 27.197 | 25.566 | 25.502 | 23.263 |
| 51 | 26.462 | 25.405 | 24.978 | 23.199 |
| 52 | 26.916 | 25.369 | 25.565 | 23.168 |
| 53 | 27.357 | 25.340 | 25.919 | 23.172 |
| 54 | 27.945 | 25.378 | 26.788 | 23.261 |
| 55 | 27.359 | 25.534 | 26.078 | 23.422 |
| 56 | 26.996 | 25.503 | 25.662 | 23.326 |
| 57 | 28.415 | 25.587 | 27.070 | 23.330 |
| 58 | 27.645 | 25.565 | 26.237 | 23.361 |
| 59 | 28.671 | 25.630 | 27.260 | 23.391 |
| 60 | 28.089 | 25.567 | 26.893 | 23.388 |
| 61 | 29.179 | 25.599 | 27.678 | 23.294 |
| 62 | 29.401 | 25.579 | 28.032 | 23.326 |
| 63 | 29.437 | 25.652 | 27.920 | 23.358 |
| 64 | 29.597 | 25.662 | 28.286 | 23.389 |
