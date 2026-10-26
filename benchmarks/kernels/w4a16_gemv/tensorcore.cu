// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t dequant_pair(uint32_t q, half scale) {
  uint32_t bits = (q & 0x000f000fu) | 0x64006400u;
  half2 value =
      __hsub2(*reinterpret_cast<half2*>(&bits), __float2half2_rn(1032.0f));
  value = __hmul2(value, __half2half2(scale));
  return *reinterpret_cast<uint32_t*>(&value);
}

__device__ __forceinline__ uint32_t adjacent_pair(uint32_t q, int pair,
                                                  half scale) {
  q = (q >> (pair * 4)) & 0xffu;
  return dequant_pair((q & 15u) | ((q & 0xf0u) << 12), scale);
}

// The MMA is W[16,16] @ X.T[16,8], giving 16 outputs for eight rows.
template <int NWarps, int KWarps, bool Vector = false, int Tiles = 1,
          bool Stream = false, bool Raw = false>
__device__ __forceinline__ void tensor_body(
    const half* __restrict__ x, const uint32_t* __restrict__ weights,
    const half* __restrict__ scales, half* __restrict__ y, int m, int k, int n,
    float* shared) {
  const int lane = threadIdx.x % 32;
  const int nw = threadIdx.x / 32 % NWarps;
  const int kw = threadIdx.x / 32 / NWarps;
  const int first_tile = (blockIdx.x * NWarps + nw) * Tiles;
  const int groups = k / 128;
  const int row = lane / 4;
  const int pair = lane % 4 * 2;
  float c[Tiles][4] = {};
  for (int group = kw; group < groups; group += KWarps) {
    half s0[Tiles], s1[Tiles];
#pragma unroll
    for (int t = 0; t < Tiles; ++t) {
      if constexpr (Raw) {
        s0[t] = scales[((first_tile + t) * 16 + row) * groups + group];
        s1[t] = scales[((first_tile + t) * 16 + row + 8) * groups + group];
      } else {
        s0[t] = scales[((first_tile + t) * groups + group) * 16 + row];
        s1[t] = scales[((first_tile + t) * groups + group) * 16 + row + 8];
      }
    }
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
      uint4 packed[Tiles];
      if constexpr (Vector && !Raw) {
#pragma unroll
        for (int t = 0; t < Tiles; ++t) {
          const uint4* ptr =
              reinterpret_cast<const uint4*>(weights) +
              (((first_tile + t) * groups + group) * 2 + chunk) * 32 + lane;
          if constexpr (Stream) {
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(packed[t].x), "=r"(packed[t].y),
                           "=r"(packed[t].z), "=r"(packed[t].w)
                         : "l"(ptr));
          } else {
            packed[t] = *ptr;
          }
        }
      }
#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        const int step = chunk * 4 + sub;
        uint32_t b0 = 0, b1 = 0;
        if (row < m) {
          const half* a = x + row * k + group * 128 + step * 16 + pair;
          b0 = *reinterpret_cast<const uint32_t*>(a);
          b1 = *reinterpret_cast<const uint32_t*>(a + 8);
        }
#pragma unroll
        for (int t = 0; t < Tiles; ++t) {
          uint32_t a0, a1, a2, a3;
          if constexpr (Raw) {
            const uint2* p0 = reinterpret_cast<const uint2*>(
                weights + ((first_tile + t) * 16 + row) * (k / 8) + group * 16 +
                step * 2);
            const uint2* p1 = p0 + 8 * (k / 16);
            uint2 q0, q1;
            if constexpr (Stream) {
              asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];"
                           : "=r"(q0.x), "=r"(q0.y)
                           : "l"(p0));
              asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];"
                           : "=r"(q1.x), "=r"(q1.y)
                           : "l"(p1));
            } else {
              q0 = *p0;
              q1 = *p1;
            }
            a0 = adjacent_pair(q0.x, pair, s0[t]);
            a1 = adjacent_pair(q1.x, pair, s1[t]);
            a2 = adjacent_pair(q0.y, pair, s0[t]);
            a3 = adjacent_pair(q1.y, pair, s1[t]);
          } else {
            const uint32_t q =
                Vector
                    ? reinterpret_cast<uint32_t*>(&packed[t])[sub]
                    : weights[(((first_tile + t) * groups + group) * 8 + step) *
                                  32 +
                              lane];
            a0 = dequant_pair(q, s0[t]);
            a1 = dequant_pair(q >> 4, s1[t]);
            a2 = dequant_pair(q >> 8, s0[t]);
            a3 = dequant_pair(q >> 12, s1[t]);
          }
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(c[t][0]), "+f"(c[t][1]), "+f"(c[t][2]), "+f"(c[t][3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
    }
  }
  if constexpr (KWarps > 1) {
#pragma unroll
    for (int t = 0; t < Tiles; ++t) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        shared[((kw * NWarps + nw) * Tiles + t) * 128 + i * 32 + lane] =
            c[t][i];
    }
    __syncthreads();
    if (kw != 0) return;
#pragma unroll
    for (int t = 0; t < Tiles; ++t) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int part = 1; part < KWarps; ++part)
          c[t][i] +=
              shared[((part * NWarps + nw) * Tiles + t) * 128 + i * 32 + lane];
      }
    }
  }
#pragma unroll
  for (int t = 0; t < Tiles; ++t) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = pair + i % 2;
      const int col = (first_tile + t) * 16 + row + i / 2 * 8;
      if (r < m) y[r * n + col] = __float2half_rn(c[t][i]);
    }
  }
}

#define TENSOR_ENTRY(NW, KW)                                                  \
  extern "C" __global__ void tensor_n##NW##_k##KW(                            \
      const half* x, const uint32_t* w, const half* s, half* y, int m, int k, \
      int n) {                                                                \
    __shared__ float scratch[NW * KW * 128];                                  \
    tensor_body<NW, KW>(x, w, s, y, m, k, n, scratch);                        \
  }

TENSOR_ENTRY(1, 1)
TENSOR_ENTRY(2, 1)
TENSOR_ENTRY(4, 1)
TENSOR_ENTRY(1, 2)
TENSOR_ENTRY(2, 2)
TENSOR_ENTRY(4, 2)
TENSOR_ENTRY(1, 4)
TENSOR_ENTRY(2, 4)
TENSOR_ENTRY(1, 8)

#define VECTOR_ENTRY(NW, KW)                                                  \
  extern "C" __global__ void tensorv_n##NW##_k##KW(                           \
      const half* x, const uint32_t* w, const half* s, half* y, int m, int k, \
      int n) {                                                                \
    __shared__ float scratch[NW * KW * 128];                                  \
    tensor_body<NW, KW, true>(x, w, s, y, m, k, n, scratch);                  \
  }

VECTOR_ENTRY(1, 2)
VECTOR_ENTRY(1, 4)
VECTOR_ENTRY(2, 4)
VECTOR_ENTRY(1, 8)
VECTOR_ENTRY(1, 16)
VECTOR_ENTRY(1, 32)

#define WIDE_ENTRY(T, NW, KW)                                                 \
  extern "C" __global__ void tensorw##T##_n##NW##_k##KW(                      \
      const half* x, const uint32_t* w, const half* s, half* y, int m, int k, \
      int n) {                                                                \
    __shared__ float scratch[NW * KW * T * 128];                              \
    tensor_body<NW, KW, true, T>(x, w, s, y, m, k, n, scratch);               \
  }

WIDE_ENTRY(2, 1, 4)
WIDE_ENTRY(2, 1, 8)
WIDE_ENTRY(2, 2, 4)
WIDE_ENTRY(2, 1, 16)
WIDE_ENTRY(4, 1, 8)
WIDE_ENTRY(4, 1, 16)

#define CACHE_ENTRY(T, KW)                                                    \
  extern "C" __global__ void tensorc##T##_n1_k##KW(                           \
      const half* x, const uint32_t* w, const half* s, half* y, int m, int k, \
      int n) {                                                                \
    __shared__ float scratch[KW * T * 128];                                   \
    tensor_body<1, KW, true, T, true>(x, w, s, y, m, k, n, scratch);          \
  }

CACHE_ENTRY(1, 8)
CACHE_ENTRY(2, 8)
CACHE_ENTRY(2, 16)

#define RAW_ENTRY(T, NW, KW, C)                                               \
  extern "C" __global__ void raw_t##T##_n##NW##_k##KW##_c##C(                 \
      const half* x, const uint32_t* w, const half* s, half* y, int m, int k, \
      int n) {                                                                \
    __shared__ float scratch[NW * KW * T * 128];                              \
    tensor_body<NW, KW, false, T, C, true>(x, w, s, y, m, k, n, scratch);     \
  }

RAW_ENTRY(1, 1, 2, 0)
RAW_ENTRY(1, 1, 4, 0)
RAW_ENTRY(1, 1, 8, 0)
RAW_ENTRY(1, 1, 16, 0)
RAW_ENTRY(1, 1, 32, 0)
RAW_ENTRY(1, 2, 2, 0)
RAW_ENTRY(1, 2, 4, 0)
RAW_ENTRY(1, 4, 2, 0)
RAW_ENTRY(2, 1, 4, 0)
RAW_ENTRY(2, 1, 8, 0)
RAW_ENTRY(2, 1, 16, 0)
RAW_ENTRY(2, 2, 4, 0)
RAW_ENTRY(4, 1, 8, 0)
RAW_ENTRY(4, 1, 16, 0)
RAW_ENTRY(2, 1, 8, 1)
RAW_ENTRY(2, 1, 16, 1)

extern "C" __global__ void repack_vector(const uint32_t* weights,
                                         const half* scales, uint32_t* packed,
                                         half* packed_scales, int k, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int groups = k / 128;
  if (index < n * (k / 8)) {
    const int lane = index / 4 % 32;
    const int step = index / 128 % 2 * 4 + index % 4;
    const int group = index / 256 % groups;
    const int tile = index / (groups * 256);
    const int row = tile * 16 + lane / 4;
    const int word = group * 16 + step * 2;
    const uint2 q0 =
        *reinterpret_cast<const uint2*>(weights + row * (k / 8) + word);
    const uint2 q1 =
        *reinterpret_cast<const uint2*>(weights + (row + 8) * (k / 8) + word);
    const uint32_t fragments[4] = {q0.x, q1.x, q0.y, q1.y};
    uint32_t result = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int shift = (lane % 4 * 2 + i / 4) * 4;
      result |= ((fragments[i % 4] >> shift) & 15u) << (i * 4);
    }
    packed[index] = result;
  }
  if (index < n * groups) {
    const int row = index / (groups * 16) * 16 + index % 16;
    const int group = index / 16 % groups;
    packed_scales[index] = scales[row * groups + group];
  }
}
