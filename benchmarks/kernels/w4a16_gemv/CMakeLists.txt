cmake_minimum_required(VERSION 3.26)
project(w4a16_gemv_experiment LANGUAGES NONE)

find_program(NVCC nvcc REQUIRED)
set(GEMV_ARCH "86" CACHE STRING "CUDA compute capability")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gemv.cubin"
  COMMAND "${NVCC}" --cubin -std=c++17 -O3 -lineinfo
          "-arch=sm_${GEMV_ARCH}" -Xptxas=-v
          "${CMAKE_CURRENT_SOURCE_DIR}/gemv.cu"
          -o "${CMAKE_CURRENT_BINARY_DIR}/gemv.cubin"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gemv.cu"
  VERBATIM)
add_custom_target(gemv ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/gemv.cubin")

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/batched.cubin"
  COMMAND "${NVCC}" --cubin -std=c++17 -O3 -lineinfo
          "-arch=sm_${GEMV_ARCH}" -Xptxas=-v
          "${CMAKE_CURRENT_SOURCE_DIR}/batched.cu"
          -o "${CMAKE_CURRENT_BINARY_DIR}/batched.cubin"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/batched.cu"
  VERBATIM)
add_custom_target(batched ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/batched.cubin")

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tensorcore.cubin"
  COMMAND "${NVCC}" --cubin -std=c++17 -O3 -lineinfo
          "-arch=sm_${GEMV_ARCH}" -Xptxas=-v
          "${CMAKE_CURRENT_SOURCE_DIR}/tensorcore.cu"
          -o "${CMAKE_CURRENT_BINARY_DIR}/tensorcore.cubin"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tensorcore.cu"
  VERBATIM)
add_custom_target(tensorcore ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/tensorcore.cubin")

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shared_layout.cubin"
  COMMAND "${NVCC}" --cubin -std=c++17 -O3 -lineinfo
          "-arch=sm_${GEMV_ARCH}" -Xptxas=-v
          "${CMAKE_CURRENT_SOURCE_DIR}/shared_layout.cu"
          -o "${CMAKE_CURRENT_BINARY_DIR}/shared_layout.cubin"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shared_layout.cu"
  VERBATIM)
add_custom_target(shared_layout ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/shared_layout.cubin")
