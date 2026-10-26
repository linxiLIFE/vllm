// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <int Rows, int Outputs, int Warps, int Threads>
__device__ __forceinline__ void batch_body(const half* __restrict__ x,
                                           const uint32_t* __restrict__ weights,
                                           const half* __restrict__ scales,
                                           half* __restrict__ y, int m, int k,
                                           int n) {
  const int lane = threadIdx.x % Threads;
  const int column =
      (blockIdx.x * Warps * 32 / Threads + threadIdx.x / Threads) * Outputs;
  if (column >= n) return;
  const int first_row = blockIdx.y * Rows;
  const int words = k / 8;
  const int groups = k / 128;
  float accum[Rows][Outputs] = {};
  for (int word = lane; word < words; word += Threads) {
    uint32_t packed[Outputs];
    half scale[Outputs];
#pragma unroll
    for (int o = 0; o < Outputs; ++o) {
      packed[o] = column + o < n ? weights[(column + o) * words + word] : 0;
      scale[o] = column + o < n ? scales[(column + o) * groups + word / 16]
                                : __float2half(0.0f);
    }
    int4 activation[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      activation[r] = first_row + r < m
                          ? *reinterpret_cast<const int4*>(
                                x + (first_row + r) * k + word * 8)
                          : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 a[Rows];
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
        const half* h = reinterpret_cast<const half*>(&activation[r]);
        a[r] = make_float2(__half2float(h[j]), __half2float(h[j + 4]));
      }
#pragma unroll
      for (int o = 0; o < Outputs; ++o) {
        const uint32_t bits =
            ((packed[o] >> (4 * j)) & 0x000f000fu) | 0x64006400u;
        const half2 encoded = *reinterpret_cast<const half2*>(&bits);
        const half2 integer = __hsub2(encoded, __float2half2_rn(1032.0f));
        const float2 b =
            __half22float2(__hmul2(integer, __half2half2(scale[o])));
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
          accum[r][o] = fmaf(a[r].x, b.x, accum[r][o]);
          accum[r][o] = fmaf(a[r].y, b.y, accum[r][o]);
        }
      }
    }
  }
  const unsigned mask = __activemask();
#pragma unroll
  for (int r = 0; r < Rows; ++r) {
#pragma unroll
    for (int o = 0; o < Outputs; ++o) {
#pragma unroll
      for (int offset = Threads / 2; offset > 0; offset /= 2) {
        accum[r][o] += __shfl_down_sync(mask, accum[r][o], offset, Threads);
      }
      if (lane == 0 && first_row + r < m && column + o < n)
        y[(first_row + r) * n + column + o] = __float2half_rn(accum[r][o]);
    }
  }
}

#define BATCH_ENTRY(Rows, Outputs, Warps, Threads)                             \
  extern "C" __global__ void                                                   \
  batch_r##Rows##_o##Outputs##_w##Warps##_t##Threads(                          \
      const half* x, const uint32_t* weights, const half* scales, half* y,     \
      int m, int k, int n) {                                                   \
    batch_body<Rows, Outputs, Warps, Threads>(x, weights, scales, y, m, k, n); \
  }

#define BATCH_ROWS(Rows)      \
  BATCH_ENTRY(Rows, 1, 4, 32) \
  BATCH_ENTRY(Rows, 1, 8, 32) \
  BATCH_ENTRY(Rows, 2, 4, 32) \
  BATCH_ENTRY(Rows, 2, 8, 32)

BATCH_ROWS(1)
BATCH_ROWS(2)
BATCH_ROWS(4)
BATCH_ROWS(8)
BATCH_ENTRY(1, 4, 4, 32)
BATCH_ENTRY(2, 1, 4, 16)
BATCH_ENTRY(4, 1, 4, 16)
BATCH_ENTRY(8, 1, 4, 16)
