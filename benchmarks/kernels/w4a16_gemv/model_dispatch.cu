#include "shared_layout.cu"

#define MODEL_ENTRY(Name, N, K, S, Rows)                                    \
  extern "C" __global__ void model_##Name(                                  \
      const half* x, const uint32_t* weight, const half* scales, half* out, \
      const half* bias, int m, int k, int n) {                              \
    __shared__ float scratch[Rows * 2 * N * K * 128];                       \
    shared_body<2, N, K, false, true, S, true, Rows>(                       \
        x, weight, scales, out, m, k, n, scratch, nullptr, bias);           \
  }

MODEL_ENTRY(shareds4_t2_n1_k8_c0, 1, 8, 4, 1)
MODEL_ENTRY(shareds4_t2_n1_k8_c0_rows, 1, 8, 4, 1)
MODEL_ENTRY(shareds4_t2_n1_k16_c0, 1, 16, 4, 1)
MODEL_ENTRY(shareds4_t2_n1_k16_c0_rows, 1, 16, 4, 1)
MODEL_ENTRY(shareds4_t2_n2_k8_c0, 2, 8, 4, 1)
MODEL_ENTRY(shareds4_t2_n2_k8_c0_rows, 2, 8, 4, 1)
MODEL_ENTRY(sharedv_t2_n1_k16_c0, 1, 16, 8, 1)
MODEL_ENTRY(sharedv_t2_n1_k16_c0_rows, 1, 16, 8, 1)
MODEL_ENTRY(reuse2_t2_k4_s4, 1, 4, 4, 2)
