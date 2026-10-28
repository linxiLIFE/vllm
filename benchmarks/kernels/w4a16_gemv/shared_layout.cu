// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t humming_pair(uint32_t q, half scale) {
  uint32_t bits = (q & 0x000f000fu) | 0x64006400u;
  half2 value =
      __hsub2(*reinterpret_cast<half2*>(&bits), __float2half2_rn(1032.0f));
  value = __hmul2(value, __half2half2(scale));
  return *reinterpret_cast<uint32_t*>(&value);
}

// Humming SM86 W4A16: [K/16, N/64, lane, four N tiles], INT4 interleave 3.
template <int Tiles, int NWarps, int KWarps, bool Stream,
          bool VectorScale = false, int Steps = 8, bool TileRows = false,
          int RowTiles = 1, int StageWeights = 0, bool Prefetch = false>
__device__ __forceinline__ void shared_body(
    const half* __restrict__ x, const uint32_t* __restrict__ weight,
    const half* __restrict__ scales, half* __restrict__ y, int m, int k, int n,
    float* scratch, uint32_t* staging = nullptr, const half* bias = nullptr) {
  const int lane = threadIdx.x % 32;
  const int nw = threadIdx.x / 32 % NWarps;
  const int kw = threadIdx.x / 32 / NWarps;
  const int first = (blockIdx.x * NWarps + nw) * Tiles;
  const int row = lane / 4;
  const int first_row = TileRows ? blockIdx.y * 8 * RowTiles : 0;
  const int pair = lane % 4 * 2;
  float c[RowTiles][Tiles][4] = {};
  for (int part = kw; part < k / (16 * Steps); part += KWarps) {
    const int first_step = part * Steps;
    const int group = first_step / 8;
    half s0[Tiles], s1[Tiles];
    uint32_t scale_bits[Tiles];
    if constexpr (VectorScale) {
      const int col = first * 16 + row;
      const half* ptr =
          scales + group * n + col / 64 * 64 + col % 8 * 8 + col % 64 / 8;
      if constexpr (Tiles == 4) {
        const uint4 v = *reinterpret_cast<const uint4*>(ptr);
#pragma unroll
        for (int t = 0; t < Tiles; ++t)
          scale_bits[t] = reinterpret_cast<const uint32_t*>(&v)[t];
      } else if constexpr (Tiles == 2) {
        const uint2 v = *reinterpret_cast<const uint2*>(ptr);
        scale_bits[0] = v.x;
        scale_bits[1] = v.y;
      } else {
        scale_bits[0] = *reinterpret_cast<const uint32_t*>(ptr);
      }
    }
#pragma unroll
    for (int t = 0; t < Tiles; ++t) {
      if constexpr (VectorScale) {
        const half2 pair_scale =
            *reinterpret_cast<const half2*>(&scale_bits[t]);
        s0[t] = __low2half(pair_scale);
        s1[t] = __high2half(pair_scale);
      } else {
        const int col = (first + t) * 16 + row;
        const int base = group * n + col / 64 * 64;
        s0[t] = scales[base + col % 8 * 8 + col % 64 / 8];
        const int col1 = col + 8;
        s1[t] = scales[base + col1 % 8 * 8 + col1 % 64 / 8];
      }
    }
    if constexpr (StageWeights == 2) {
      const uint32_t* ptr = weight + (first_step * (n / 64) + first / 4) * 128 +
                            lane * 4 + first % 4;
      const uint32_t address = static_cast<uint32_t>(
          __cvta_generic_to_shared(staging + threadIdx.x * Tiles));
      asm volatile(
          "cp.async.ca.shared.global [%0], [%1], %2;\ncp.async.commit_group;" ::
              "r"(address),
          "l"(ptr), "n"(Tiles * 4)
          : "memory");
    }
    uint32_t prefetched[Steps][Tiles];
    if constexpr (Prefetch) {
#pragma unroll
      for (int step = 0; step < Steps; ++step) {
        const uint32_t* ptr =
            weight + ((first_step + step) * (n / 64) + first / 4) * 128 +
            lane * 4 + first % 4;
        if constexpr (Tiles == 4) {
          const uint4 v = *reinterpret_cast<const uint4*>(ptr);
#pragma unroll
          for (int t = 0; t < Tiles; ++t)
            prefetched[step][t] = reinterpret_cast<const uint32_t*>(&v)[t];
        } else {
          const uint2 v = *reinterpret_cast<const uint2*>(ptr);
          prefetched[step][0] = v.x;
          prefetched[step][1] = v.y;
        }
      }
    }
#pragma unroll
    for (int step = 0; step < Steps; ++step) {
      const int k_tile = first_step + step;
      const uint32_t* ptr =
          weight + (k_tile * (n / 64) + first / 4) * 128 + lane * 4 + first % 4;
      uint32_t q[Tiles];
      if constexpr (Prefetch) {
#pragma unroll
        for (int t = 0; t < Tiles; ++t) q[t] = prefetched[step][t];
      } else if constexpr (StageWeights) {
        const uint32_t address = static_cast<uint32_t>(
            __cvta_generic_to_shared(staging + threadIdx.x * Tiles));
        if constexpr (StageWeights == 1) {
          asm volatile(
              "cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(address),
              "l"(ptr), "n"(Tiles * 4)
              : "memory");
          asm volatile("cp.async.commit_group;" ::: "memory");
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncwarp();
#pragma unroll
        for (int t = 0; t < Tiles; ++t) q[t] = staging[threadIdx.x * Tiles + t];
        if constexpr (StageWeights == 2) {
          if (step + 1 < Steps) {
            const uint32_t* next = ptr + (n / 64) * 128;
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], "
                "%2;\ncp.async.commit_group;" ::"r"(address),
                "l"(next), "n"(Tiles * 4)
                : "memory");
          }
        }
      } else if constexpr (Tiles == 4) {
        uint4 v;
        if constexpr (Stream) {
          asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                       : "l"(ptr));
        } else {
          v = *reinterpret_cast<const uint4*>(ptr);
        }
#pragma unroll
        for (int t = 0; t < Tiles; ++t)
          q[t] = reinterpret_cast<uint32_t*>(&v)[t];
      } else if constexpr (Tiles == 2) {
        uint2 v;
        if constexpr (Stream) {
          asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];"
                       : "=r"(v.x), "=r"(v.y)
                       : "l"(ptr));
        } else {
          v = *reinterpret_cast<const uint2*>(ptr);
        }
        q[0] = v.x;
        q[1] = v.y;
      } else {
        q[0] = *ptr;
      }
      uint32_t b0[RowTiles] = {}, b1[RowTiles] = {};
#pragma unroll
      for (int r = 0; r < RowTiles; ++r)
        if (first_row + r * 8 + row < m) {
          const half* a =
              x + (first_row + r * 8 + row) * k + k_tile * 16 + pair;
          b0[r] = *reinterpret_cast<const uint32_t*>(a);
          b1[r] = *reinterpret_cast<const uint32_t*>(a + 8);
        }
#pragma unroll
      for (int t = 0; t < Tiles; ++t) {
        const uint32_t a0 = humming_pair(q[t], s0[t]);
        const uint32_t a1 = humming_pair(q[t] >> 8, s1[t]);
        const uint32_t a2 = humming_pair(q[t] >> 4, s0[t]);
        const uint32_t a3 = humming_pair(q[t] >> 12, s1[t]);
#pragma unroll
        for (int r = 0; r < RowTiles; ++r)
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(c[r][t][0]), "+f"(c[r][t][1]), "+f"(c[r][t][2]),
                "+f"(c[r][t][3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[r]), "r"(b1[r]));
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RowTiles; ++r) {
#pragma unroll
    for (int t = 0; t < Tiles; ++t) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        scratch[(((r * KWarps + kw) * NWarps + nw) * Tiles + t) * 128 + i * 32 +
                lane] = c[r][t][i];
    }
  }
  __syncthreads();
  if (kw != 0) return;
#pragma unroll
  for (int rt = 0; rt < RowTiles; ++rt) {
#pragma unroll
    for (int t = 0; t < Tiles; ++t) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int part = 1; part < KWarps; ++part)
          c[rt][t][i] +=
              scratch[(((rt * KWarps + part) * NWarps + nw) * Tiles + t) * 128 +
                      i * 32 + lane];
        const int r = first_row + rt * 8 + pair + i % 2;
        const int col = (first + t) * 16 + row + i / 2 * 8;
        if (r < m) {
          half value = __float2half_rn(c[rt][t][i]);
          if (bias != nullptr) {
            const int bias_col =
                col / 32 * 32 + col % 2 + col % 8 / 2 * 8 + col % 32 / 8 * 2;
            value = __hadd(value, bias[bias_col]);
          }
          y[r * n + col] = value;
        }
      }
    }
  }
}

#define SHARED_ENTRY(T, N, K, C)                                              \
  extern "C" __global__ void shared_t##T##_n##N##_k##K##_c##C(                \
      const half* x, const uint32_t* w, const half* s, half* y, int m, int k, \
      int n) {                                                                \
    __shared__ float scratch[T * N * K * 128];                                \
    shared_body<T, N, K, C>(x, w, s, y, m, k, n, scratch);                    \
  }

SHARED_ENTRY(1, 1, 2, 0)
SHARED_ENTRY(1, 1, 4, 0)
SHARED_ENTRY(1, 1, 8, 0)
SHARED_ENTRY(1, 1, 16, 0)
SHARED_ENTRY(1, 2, 4, 0)
SHARED_ENTRY(1, 2, 8, 0)
SHARED_ENTRY(2, 1, 4, 0)
SHARED_ENTRY(2, 1, 8, 0)
SHARED_ENTRY(2, 1, 16, 0)
SHARED_ENTRY(2, 2, 4, 0)
SHARED_ENTRY(2, 2, 8, 0)
SHARED_ENTRY(4, 1, 4, 0)
SHARED_ENTRY(4, 1, 8, 0)
SHARED_ENTRY(4, 1, 16, 0)
SHARED_ENTRY(2, 1, 8, 1)
SHARED_ENTRY(4, 1, 8, 1)
SHARED_ENTRY(4, 1, 16, 1)

#define SCALE_ENTRY(T, N, K, C)                                               \
  extern "C" __global__ void sharedv_t##T##_n##N##_k##K##_c##C(               \
      const half* x, const uint32_t* w, const half* s, half* y, int m, int k, \
      int n) {                                                                \
    __shared__ float scratch[T * N * K * 128];                                \
    shared_body<T, N, K, C, true>(x, w, s, y, m, k, n, scratch);              \
  }

SCALE_ENTRY(1, 1, 8, 0)
SCALE_ENTRY(1, 2, 4, 0)
SCALE_ENTRY(2, 1, 4, 0)
SCALE_ENTRY(2, 1, 8, 0)
SCALE_ENTRY(2, 1, 16, 0)
SCALE_ENTRY(2, 2, 4, 0)
SCALE_ENTRY(2, 2, 8, 0)
SCALE_ENTRY(4, 1, 4, 0)
SCALE_ENTRY(4, 1, 8, 0)
SCALE_ENTRY(4, 1, 16, 0)
SCALE_ENTRY(2, 1, 8, 1)
SCALE_ENTRY(4, 1, 8, 1)
SCALE_ENTRY(4, 1, 16, 1)

#define STEP_ENTRY(S, T, N, K)                                                \
  extern "C" __global__ void shareds##S##_t##T##_n##N##_k##K##_c0(            \
      const half* x, const uint32_t* w, const half* s, half* y, int m, int k, \
      int n) {                                                                \
    __shared__ float scratch[T * N * K * 128];                                \
    shared_body<T, N, K, false, true, S>(x, w, s, y, m, k, n, scratch);       \
  }

#define STEP_CONFIGS(S)   \
  STEP_ENTRY(S, 2, 1, 8)  \
  STEP_ENTRY(S, 2, 1, 16) \
  STEP_ENTRY(S, 2, 2, 8)  \
  STEP_ENTRY(S, 4, 1, 8)  \
  STEP_ENTRY(S, 4, 1, 16)

STEP_CONFIGS(1)
STEP_CONFIGS(2)
STEP_CONFIGS(4)

#define ROW_ENTRY(Name, T, N, K, S)                                           \
  extern "C" __global__ void Name(const half* x, const uint32_t* w,           \
                                  const half* s, half* y, int m, int k,       \
                                  int n) {                                    \
    __shared__ float scratch[T * N * K * 128];                                \
    shared_body<T, N, K, false, true, S, true>(x, w, s, y, m, k, n, scratch); \
  }

ROW_ENTRY(sharedv_t2_n1_k16_c0_rows, 2, 1, 16, 8)
ROW_ENTRY(shareds4_t2_n1_k8_c0_rows, 2, 1, 8, 4)
ROW_ENTRY(shareds4_t2_n2_k8_c0_rows, 2, 2, 8, 4)
ROW_ENTRY(shareds4_t2_n1_k16_c0_rows, 2, 1, 16, 4)

#define EXP_ENTRY(Name, T, K, S, R, Stage)                                    \
  extern "C" __global__ void Name(const half* x, const uint32_t* w,           \
                                  const half* s, half* y, int m, int k,       \
                                  int n) {                                    \
    __shared__ float scratch[R * T * K * 128];                                \
    __shared__ uint32_t staging[Stage ? T * K * 32 : 1];                      \
    shared_body<T, 1, K, false, true, S, true, R, Stage>(x, w, s, y, m, k, n, \
                                                         scratch, staging);   \
  }

EXP_ENTRY(reuse2_t2_k8_s4, 2, 8, 4, 2, false)
EXP_ENTRY(reuse2_t2_k4_s4, 2, 4, 4, 2, false)
EXP_ENTRY(reuse2_t2_k4_s8, 2, 4, 8, 2, false)
EXP_ENTRY(reuse2_t2_k16_s8, 2, 16, 8, 2, false)
EXP_ENTRY(reuse2_t2_k8_s8, 2, 8, 8, 2, false)
EXP_ENTRY(reuse4_t2_k8_s4, 2, 8, 4, 4, false)
EXP_ENTRY(reuse4_t2_k8_s8, 2, 8, 8, 4, false)
EXP_ENTRY(reuse4_t2_k4_s8, 2, 4, 8, 4, false)
EXP_ENTRY(reuse4_t2_k4_s4, 2, 4, 4, 4, false)
EXP_ENTRY(staged_t2_k8_s4, 2, 8, 4, 1, true)
EXP_ENTRY(staged_t2_k16_s8, 2, 16, 8, 1, true)
EXP_ENTRY(pipelined_t2_k8_s4, 2, 8, 4, 1, 2)
EXP_ENTRY(pipelined_t2_k16_s8, 2, 16, 8, 1, 2)
