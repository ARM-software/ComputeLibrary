/*
 * Copyright (c) 2017-2019 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cstring>
#include "utils.hpp"
#include "winograd.hpp"

using namespace winograd;
using array2 = std::pair<unsigned int, unsigned int>;

#define MEMBERFN(RTYPE)                                                        \
  template <int output_tile_rows, int output_tile_cols, int kernel_rows,       \
            int kernel_cols, WinogradRoots roots>                              \
  template <typename TOut, typename TIn, typename TGEMMIn, typename TGEMMOut>  \
  RTYPE WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows,          \
                     kernel_cols,                                              \
                     roots>::Convolution<TOut, TIn, TGEMMIn, TGEMMOut>

/** Get the output shape of a convolution. */
MEMBERFN(array2)
::get_output_shape(const std::pair<unsigned int, unsigned int> input_shape,
                   const bool padding_same) {
  const unsigned int n_rows =
      padding_same ? input_shape.first : input_shape.first - (kernel_rows - 1);
  const unsigned int n_cols = padding_same
                                  ? input_shape.second
                                  : input_shape.second - (kernel_cols - 1);
  return {n_rows, n_cols};
}

/** Get the memory required to store the kernel transformed into the
 * Winograd domain.
 */
MEMBERFN(size_t)
::get_kernel_storage_size(const unsigned int n_input_channels,
                          const unsigned int n_output_channels) {
  return N_GEMMS * get_kernel_matrix_size(n_input_channels, n_output_channels);
}

MEMBERFN(size_t)
::get_input_storage_size(const unsigned int n_batches,
                         const unsigned int n_rows, const unsigned int n_cols,
                         const unsigned int n_channels,
                         const bool same_padding) {
  return N_GEMMS * get_input_matrix_size(n_batches, n_rows, n_cols, n_channels,
                                         same_padding);
}

MEMBERFN(size_t)
::get_output_storage_size(const unsigned int n_batches,
                          const unsigned int n_rows, const unsigned int n_cols,
                          const unsigned int n_channels) {
  return N_GEMMS *
         get_output_matrix_size(n_batches, n_rows, n_cols, n_channels);
}

/** Get the memory required to apply a Winograd operator to some input.
 */
MEMBERFN(size_t)
::get_working_space_size(const unsigned int n_batches,
                         const unsigned int n_rows, const unsigned int n_cols,
                         const unsigned int n_input_channels,
                         const unsigned int n_output_channels,
                         const bool padding_same) {
  const auto output_shape = get_output_shape({n_rows, n_cols}, padding_same);

  // Get the memory required to store the matrices
  const size_t matrix_sizes =
      N_GEMMS *
      (get_input_matrix_size(n_batches, n_rows, n_cols, n_input_channels,
                             padding_same) +
       get_output_matrix_size(n_batches, output_shape.first,
                              output_shape.second, n_output_channels));
  return matrix_sizes;
}

/* Get the memory required by a single "input" matrix.
 */
MEMBERFN(size_t)
::get_input_matrix_size(const unsigned int n_batches, const unsigned int n_rows,
                        const unsigned int n_cols,
                        const unsigned int n_channels,
                        const bool same_padding) {
  return get_input_matrix_stride(n_batches, n_rows, n_cols, n_channels,
                                 same_padding) *
         sizeof(TGEMMIn);
}

MEMBERFN(int)
::get_input_matrix_stride(const unsigned int n_batches, const unsigned int n_rows,
                        const unsigned int n_cols,
                        const unsigned int n_channels,
                        const bool same_padding) {
  const auto output_shape = get_output_shape({n_rows, n_cols}, same_padding);
  const unsigned int tile_rows = iceildiv(output_shape.first, output_tile_rows);
  const unsigned int tile_cols =
      iceildiv(output_shape.second, output_tile_cols);
  const unsigned int M =
      roundup<unsigned int>(n_batches * tile_rows * tile_cols, M_BLOCK);
  const unsigned int K = n_channels;

  return M * K;
}

/* Get the memory required by a single "output" matrix.
 */
MEMBERFN(size_t)
::get_output_matrix_size(const unsigned int n_batches,
                         const unsigned int n_rows, const unsigned int n_cols,
                         const unsigned int n_channels) {
  return get_output_matrix_stride(n_batches, n_rows, n_cols, n_channels) *
         sizeof(TGEMMOut);
}

MEMBERFN(int)
::get_output_matrix_stride(const unsigned int n_batches,
                           const unsigned int n_rows, const unsigned int n_cols,
                           const unsigned int n_channels) {
  // Compute shape for the GEMM
  const int tile_rows = iceildiv(n_rows, output_tile_rows);
  const int tile_cols = iceildiv(n_cols, output_tile_cols);
  const int M = roundup<int>(tile_rows * tile_cols, M_BLOCK);
  const int N = roundup<int>(n_channels, N_BLOCK);

  return n_batches * M * N;
}


/* Get the memory required by a single "kernel" matrix.
 */
MEMBERFN(size_t)
::get_kernel_matrix_size(const unsigned int n_input_channels,
                         const unsigned int n_output_channels) {
  return sizeof(TGEMMIn) *
         get_kernel_matrix_stride(n_input_channels, n_output_channels);
}

MEMBERFN(int)
::get_kernel_matrix_stride(const unsigned int n_input_channels,
                           const unsigned int n_output_channels) {
  return n_input_channels * roundup<int>(n_output_channels, N_BLOCK);
}

// Instantiate required implementations
template class WinogradGEMM<2, 2, 3, 3, WinogradRoots::Integers>::Convolution<float, float, float, float>;
template class WinogradGEMM<4, 4, 3, 3, WinogradRoots::Integers>::Convolution<float, float, float, float>;

template class WinogradGEMM<1, 6, 1, 3, WinogradRoots::Integers>::Convolution<float, float, float, float>;
template class WinogradGEMM<6, 1, 3, 1, WinogradRoots::Integers>::Convolution<float, float, float, float>;

template class WinogradGEMM<2, 2, 5, 5, WinogradRoots::Integers>::Convolution<float, float, float, float>;

template class WinogradGEMM<1, 4, 1, 5, WinogradRoots::Integers>::Convolution<float, float, float, float>;
template class WinogradGEMM<4, 1, 5, 1, WinogradRoots::Integers>::Convolution<float, float, float, float>;

template class WinogradGEMM<1, 2, 1, 7, WinogradRoots::Integers>::Convolution<float, float, float, float>;
template class WinogradGEMM<2, 1, 7, 1, WinogradRoots::Integers>::Convolution<float, float, float, float>;

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template class WinogradGEMM<4, 4, 3, 3, WinogradRoots::Integers>::Convolution<__fp16, __fp16, __fp16, __fp16>;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
