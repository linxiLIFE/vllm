// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <int Warps, int Outputs, bool Paired, bool VectorInput,
          int Threads = 32, bool BroadcastScale = false,
          bool FloatDequant = false, int StaticK = 0, bool SpreadPair = false,
          bool Prefetch = false, bool SplitAccum = false, int Splits = 1,
          bool LocalSplit = false>
__device__ __forceinline__ void gemv_body(
    const half* __restrict__ x, const uint32_t* __restrict__ weights,
    const half* __restrict__ scales, half* __restrict__ y, int k, int n,
    float* __restrict__ partial = nullptr) {
  if constexpr (StaticK != 0) k = StaticK;
  const int lane = threadIdx.x % Threads;
  const int logical_thread =
      LocalSplit ? threadIdx.x % (Warps * 32) : threadIdx.x;
  const int row =
      (blockIdx.x * Warps * 32 / Threads + logical_thread / Threads) * Outputs;
  if (row >= n) return;
  const int words = k / 8;
  const int groups = k / 128;
  const int split_id = LocalSplit ? threadIdx.x / (Warps * 32) : blockIdx.y;
  const int first_word = Splits == 1 ? 0 : (groups * split_id / Splits) * 16;
  const int last_word =
      Splits == 1 ? words : (groups * (split_id + 1) / Splits) * 16;
  float accum[Outputs] = {};
  float accum_other[Outputs] = {};
  uint32_t next_packed[Outputs];
  half next_scale[Outputs];
  int4 next_activation;
  if constexpr (Prefetch) {
#pragma unroll
    for (int o = 0; o < Outputs; ++o) {
      next_packed[o] = row + o < n ? weights[(row + o) * words + lane] : 0;
      next_scale[o] = row + o < n ? scales[(row + o) * groups + lane / 16]
                                  : __float2half(0.0f);
    }
    next_activation = *reinterpret_cast<const int4*>(x + lane * 8);
  }

  for (int word = first_word + lane; word < last_word; word += Threads) {
    uint32_t packed[Outputs];
    half scale[Outputs];
    int4 activation;
    if constexpr (Prefetch) {
      activation = next_activation;
#pragma unroll
      for (int o = 0; o < Outputs; ++o) {
        packed[o] = next_packed[o];
        scale[o] = next_scale[o];
      }
      if (word + Threads < words) {
#pragma unroll
        for (int o = 0; o < Outputs; ++o) {
          next_packed[o] =
              row + o < n ? weights[(row + o) * words + word + Threads] : 0;
          next_scale[o] =
              row + o < n ? scales[(row + o) * groups + (word + Threads) / 16]
                          : __float2half(0.0f);
        }
        next_activation =
            *reinterpret_cast<const int4*>(x + (word + Threads) * 8);
      }
    } else {
#pragma unroll
      for (int o = 0; o < Outputs; ++o) {
        packed[o] = row + o < n ? weights[(row + o) * words + word] : 0;
        if constexpr (BroadcastScale) {
          float value = 0.0f;
          if (lane % 16 == 0 && row + o < n)
            value = __half2float(scales[(row + o) * groups + word / 16]);
          value = __shfl_sync(__activemask(), value, (lane / 16) * 16, Threads);
          scale[o] = __float2half_rn(value);
        } else {
          scale[o] = row + o < n ? scales[(row + o) * groups + word / 16]
                                 : __float2half(0.0f);
        }
      }
      if constexpr (VectorInput) {
        activation = *reinterpret_cast<const int4*>(x + word * 8);
      }
    }
    if constexpr (Paired) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        half2 input;
        if constexpr (VectorInput) {
          input = reinterpret_cast<const half2*>(&activation)[j];
        } else {
          input = reinterpret_cast<const half2*>(x + word * 8)[j];
        }
        float2 a;
        if constexpr (SpreadPair) {
          a = make_float2(
              __half2float(reinterpret_cast<const half*>(&activation)[j]),
              __half2float(reinterpret_cast<const half*>(&activation)[j + 4]));
        } else {
          a = __half22float2(input);
        }
#pragma unroll
        for (int o = 0; o < Outputs; ++o) {
          const uint32_t q = packed[o] >> ((SpreadPair ? 4 : 8) * j);
          const uint32_t bits =
              SpreadPair ? 0x64006400u | (q & 0x000f000fu)
                         : 0x64006400u | (q & 0xfu) | ((q & 0xf0u) << 12);
          const half2 encoded = *reinterpret_cast<const half2*>(&bits);
          const half2 integer = __hsub2(encoded, __float2half2_rn(1032.0f));
          const float2 b =
              __half22float2(__hmul2(integer, __half2half2(scale[o])));
          if constexpr (SplitAccum) {
            accum[o] = fmaf(a.x, b.x, accum[o]);
            accum_other[o] = fmaf(a.y, b.y, accum_other[o]);
          } else {
            accum[o] = fmaf(a.x, b.x, accum[o]);
            accum[o] = fmaf(a.y, b.y, accum[o]);
          }
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float a =
            VectorInput
                ? __half2float(reinterpret_cast<const half*>(&activation)[j])
                : __half2float(x[word * 8 + j]);
#pragma unroll
        for (int o = 0; o < Outputs; ++o) {
          const int q = int((packed[o] >> (4 * j)) & 0xfu) - 8;
          half b;
          if constexpr (FloatDequant) {
            b = __float2half_rn(__int2float_rn(q) * __half2float(scale[o]));
          } else {
            b = __hmul(__int2half_rn(q), scale[o]);
          }
          accum[o] = fmaf(a, __half2float(b), accum[o]);
        }
      }
    }
  }
#pragma unroll
  for (int o = 0; o < Outputs; ++o) {
    if constexpr (SplitAccum) accum[o] += accum_other[o];
#pragma unroll
    for (int offset = Threads / 2; offset > 0; offset /= 2) {
      if constexpr (Threads == 32) {
        accum[o] += __shfl_down_sync(0xffffffffu, accum[o], offset);
      } else {
        accum[o] += __shfl_down_sync(__activemask(), accum[o], offset, Threads);
      }
    }
    if (lane == 0 && row + o < n) {
      if constexpr (Splits == 1) {
        y[row + o] = __float2half_rn(accum[o]);
      } else {
        if constexpr (LocalSplit) {
          partial[split_id * Warps * Outputs + logical_thread / 32 * Outputs +
                  o] = accum[o];
        } else {
          partial[split_id * n + row + o] = accum[o];
        }
      }
    }
  }
}

#define GEMV_ENTRY(Name, Warps, Outputs, Paired, VectorInput)                  \
  extern "C" __global__ void Name(const half* x, const uint32_t* weights,      \
                                  const half* scales, half* y, int k, int n) { \
    gemv_body<Warps, Outputs, Paired, VectorInput>(x, weights, scales, y, k,   \
                                                   n);                         \
  }

GEMV_ENTRY(gemv_scalar_w4, 4, 1, false, false)
GEMV_ENTRY(gemv_scalar_w8, 8, 1, false, false)
GEMV_ENTRY(gemv_pair_w4_o1, 4, 1, true, false)
GEMV_ENTRY(gemv_pair_w8_o1, 8, 1, true, false)
GEMV_ENTRY(gemv_pair_w4_o2, 4, 2, true, false)
GEMV_ENTRY(gemv_pair_w8_o2, 8, 2, true, false)
GEMV_ENTRY(gemv_pair_w4_o4, 4, 4, true, false)
GEMV_ENTRY(gemv_pair_w8_o4, 8, 4, true, false)
GEMV_ENTRY(gemv_vector_w4_o2, 4, 2, true, true)
GEMV_ENTRY(gemv_vector_w8_o2, 8, 2, true, true)

#define GEMV_TUNED(Name, Warps, Outputs, Threads, Broadcast, Float, K)         \
  extern "C" __global__ void Name(const half* x, const uint32_t* weights,      \
                                  const half* scales, half* y, int k, int n) { \
    gemv_body<Warps, Outputs, !Float, true, Threads, Broadcast, Float, K>(     \
        x, weights, scales, y, k, n);                                          \
  }

GEMV_TUNED(gemv_t16_w4_o1, 4, 1, 16, false, false, 0)
GEMV_TUNED(gemv_t16_w8_o1, 8, 1, 16, false, false, 0)
GEMV_TUNED(gemv_t16_w4_o2, 4, 2, 16, false, false, 0)
GEMV_TUNED(gemv_t16_w8_o2, 8, 2, 16, false, false, 0)
GEMV_TUNED(gemv_t8_w4_o1, 4, 1, 8, false, false, 0)
GEMV_TUNED(gemv_t8_w8_o1, 8, 1, 8, false, false, 0)
GEMV_TUNED(gemv_t8_w4_o2, 4, 2, 8, false, false, 0)
GEMV_TUNED(gemv_t8_w8_o2, 8, 2, 8, false, false, 0)
GEMV_TUNED(gemv_broadcast_t32, 4, 2, 32, true, false, 0)
GEMV_TUNED(gemv_broadcast_t16, 4, 2, 16, true, false, 0)
GEMV_TUNED(gemv_float_t32, 4, 2, 32, false, true, 0)
GEMV_TUNED(gemv_float_t16, 4, 2, 16, false, true, 0)
GEMV_TUNED(gemv_k1536_t32, 4, 2, 32, false, false, 1536)
GEMV_TUNED(gemv_k1536_t16, 4, 2, 16, false, false, 1536)

#define GEMV_OVERLAP(Name, Warps, Threads, Spread, Prefetch, K)                \
  extern "C" __global__ void Name(const half* x, const uint32_t* weights,      \
                                  const half* scales, half* y, int k, int n) { \
    gemv_body<Warps, 2, true, true, Threads, false, false, K, Spread,          \
              Prefetch>(x, weights, scales, y, k, n);                          \
  }

GEMV_OVERLAP(gemv_spread_t32, 4, 32, true, false, 0)
GEMV_OVERLAP(gemv_spread_t16, 4, 16, true, false, 0)
GEMV_OVERLAP(gemv_spread_t8, 4, 8, true, false, 0)
GEMV_OVERLAP(gemv_prefetch_t32, 4, 32, false, true, 0)
GEMV_OVERLAP(gemv_prefetch_t16, 4, 16, false, true, 0)
GEMV_OVERLAP(gemv_spread_prefetch_t32, 4, 32, true, true, 0)
GEMV_OVERLAP(gemv_spread_prefetch_t16, 4, 16, true, true, 0)
GEMV_OVERLAP(gemv_spread_k1536_t32, 4, 32, true, false, 1536)
GEMV_OVERLAP(gemv_spread_k1536_t16, 4, 16, true, false, 1536)

#define GEMV_ILP(Name, Threads, Spread)                                        \
  extern "C" __global__ void Name(const half* x, const uint32_t* weights,      \
                                  const half* scales, half* y, int k, int n) { \
    gemv_body<4, 2, true, true, Threads, false, false, 0, Spread, false,       \
              true>(x, weights, scales, y, k, n);                              \
  }

GEMV_ILP(gemv_ilp_t32, 32, false)
GEMV_ILP(gemv_ilp_spread_t32, 32, true)
GEMV_ILP(gemv_ilp_spread_t16, 16, true)

#define GEMV_SPREAD_CFG(Name, Warps, Outputs)                                  \
  extern "C" __global__ void Name(const half* x, const uint32_t* weights,      \
                                  const half* scales, half* y, int k, int n) { \
    gemv_body<Warps, Outputs, true, true, 32, false, false, 0, true>(          \
        x, weights, scales, y, k, n);                                          \
  }

GEMV_SPREAD_CFG(gemv_spread_w4_o1, 4, 1)
GEMV_SPREAD_CFG(gemv_spread_w8_o1, 8, 1)
GEMV_SPREAD_CFG(gemv_spread_w8_o2, 8, 2)
GEMV_SPREAD_CFG(gemv_spread_w4_o4, 4, 4)
GEMV_SPREAD_CFG(gemv_spread_w8_o4, 8, 4)

#define GEMV_SPLIT(Name, Warps, Outputs, Splits)                             \
  extern "C" __global__ void Name(const half* x, const uint32_t* weights,    \
                                  const half* scales, half* y, int k, int n, \
                                  float* partial) {                          \
    gemv_body<Warps, Outputs, true, true, 32, false, false, 0, true, false,  \
              false, Splits>(x, weights, scales, y, k, n, partial);          \
  }

GEMV_SPLIT(gemv_split2_w4_o2, 4, 2, 2)
GEMV_SPLIT(gemv_split4_w4_o2, 4, 2, 4)
GEMV_SPLIT(gemv_split8_w4_o2, 4, 2, 8)
GEMV_SPLIT(gemv_split16_w4_o2, 4, 2, 16)
GEMV_SPLIT(gemv_split2_w4_o1, 4, 1, 2)
GEMV_SPLIT(gemv_split4_w4_o1, 4, 1, 4)
GEMV_SPLIT(gemv_split8_w4_o1, 4, 1, 8)
GEMV_SPLIT(gemv_split2_w8_o2, 8, 2, 2)
GEMV_SPLIT(gemv_split4_w8_o2, 8, 2, 4)
GEMV_SPLIT(gemv_split8_w8_o2, 8, 2, 8)

extern "C" __global__ void gemv_reduce_split(const float* partial, half* y,
                                             int n, int splits) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  float sum = 0.0f;
  for (int s = 0; s < splits; ++s) sum += partial[s * n + row];
  y[row] = __float2half_rn(sum);
}

#define GEMV_LOCAL_SPLIT(Name, Warps, Outputs, Splits)                         \
  extern "C" __global__ void Name(const half* x, const uint32_t* weights,      \
                                  const half* scales, half* y, int k, int n) { \
    __shared__ float partial[Splits * Warps * Outputs];                        \
    gemv_body<Warps, Outputs, true, true, 32, false, false, 0, true, false,    \
              false, Splits, true>(x, weights, scales, y, k, n, partial);      \
    __syncthreads();                                                           \
    if (threadIdx.x < Warps * Outputs) {                                       \
      float sum = 0.0f;                                                        \
      for (int s = 0; s < Splits; ++s)                                         \
        sum += partial[s * Warps * Outputs + threadIdx.x];                     \
      y[blockIdx.x * Warps * Outputs + threadIdx.x] = __float2half_rn(sum);    \
    }                                                                          \
  }

GEMV_LOCAL_SPLIT(gemv_local2_w4_o2, 4, 2, 2)
GEMV_LOCAL_SPLIT(gemv_local4_w4_o2, 4, 2, 4)
GEMV_LOCAL_SPLIT(gemv_local8_w4_o2, 4, 2, 8)
GEMV_LOCAL_SPLIT(gemv_local2_w4_o1, 4, 1, 2)
GEMV_LOCAL_SPLIT(gemv_local4_w4_o1, 4, 1, 4)
GEMV_LOCAL_SPLIT(gemv_local2_w2_o2, 2, 2, 2)
