// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "model_dispatch.cu"

#define WIDE_ENTRY(R, T, K, S)                                              \
  extern "C" __global__ void model_wide_r##R##_t##T##_k##K##_s##S(          \
      const half* x, const uint32_t* weight, const half* scales, half* out, \
      const half* bias, int m, int k, int n) {                              \
    __shared__ float scratch[R * T * K * 128];                              \
    shared_body<T, 1, K, false, true, S, true, R>(                          \
        x, weight, scales, out, m, k, n, scratch, nullptr, bias);           \
  }

#define WIDE_STEPS(R, T, K) \
  WIDE_ENTRY(R, T, K, 4)    \
  WIDE_ENTRY(R, T, K, 8)

#define WIDE_TILES(R, K) \
  WIDE_STEPS(R, 2, K)    \
  WIDE_STEPS(R, 4, K)

#define WIDE_ROWS(R) \
  WIDE_TILES(R, 1)   \
  WIDE_TILES(R, 2)   \
  WIDE_TILES(R, 4)

WIDE_ROWS(1)
WIDE_ROWS(2)
WIDE_ROWS(4)
WIDE_TILES(8, 1)
WIDE_TILES(8, 2)
WIDE_STEPS(8, 2, 4)

#define SHORT_STEPS(R, T, K) \
  WIDE_ENTRY(R, T, K, 1)     \
  WIDE_ENTRY(R, T, K, 2)

SHORT_STEPS(2, 4, 1)
SHORT_STEPS(2, 4, 2)
SHORT_STEPS(4, 4, 1)
SHORT_STEPS(4, 4, 2)
SHORT_STEPS(4, 4, 4)
SHORT_STEPS(8, 4, 1)
SHORT_STEPS(8, 4, 2)
SHORT_STEPS(8, 2, 2)

#define PREFETCH_ENTRY(R, T, K, S)                                          \
  extern "C" __global__ void model_prefetch_r##R##_t##T##_k##K##_s##S(      \
      const half* x, const uint32_t* weight, const half* scales, half* out, \
      const half* bias, int m, int k, int n) {                              \
    __shared__ float scratch[R * T * K * 128];                              \
    shared_body<T, 1, K, false, true, S, true, R, 0, true>(                 \
        x, weight, scales, out, m, k, n, scratch, nullptr, bias);           \
  }

PREFETCH_ENTRY(4, 4, 1, 8)
PREFETCH_ENTRY(4, 4, 4, 8)
