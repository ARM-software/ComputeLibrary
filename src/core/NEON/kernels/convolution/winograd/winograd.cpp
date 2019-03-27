/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "winograd.hpp"
using namespace winograd;

/** Get the output shape of a convolution. */
template <int kr, int kc, int itr, int itc, WinogradRoots R>
template <typename TOut, typename TIn, typename TInGEMM, typename TOutGEMM>
Tensor4DShape WinogradGEMM<kr, kc, itr, itc, R>::Convolution<TOut, TIn, TInGEMM, TOutGEMM>::get_output_shape(
  const KernelShape &kernel_shape,
  const Tensor4DShape &in_shape,
  const PaddingType padding
)
{
  return Tensor4DShape {
    in_shape.n_batches,
    (padding == PADDING_SAME) ? in_shape.n_rows : in_shape.n_rows - (kernel_rows - 1),
    (padding == PADDING_SAME) ? in_shape.n_cols : in_shape.n_cols - (kernel_cols - 1),
    kernel_shape.n_output_channels,
    in_shape.ordering
  };
}

/* Get the memory required to transform the kernel.
 */
template <int kernel_rows, int kernel_cols,
          int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_kernel_transform_working_size(const KernelShape &shape)
{
  if (shape.ordering == HWIO)
  {
    // Kernel is already in the correct order, so no additional memory is
    // required.
    return 0;
  }
  else
  {
    // Need to re-order the kernel into HWIO form, require enough space to
    // represent the tensor.
    return sizeof(TIn) * shape.size();
  }
}

/** Get the memory required to store the kernel transformed into the
 * Winograd domain.
 */
template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_kernel_storage_size(const KernelShape &shape)
{
  return N_GEMMS * get_kernel_matrix_size(shape);
}


template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_input_storage_size(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding
)
{
  return N_GEMMS * get_input_matrix_size(kernel_shape, input_shape, padding);
}


template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_output_storage_size(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding
)
{
  return N_GEMMS * get_output_matrix_size(kernel_shape, input_shape, padding);
}


/** Get the memory required to apply a Winograd operator to some input.
 */
template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_working_space_size(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding_type
)
{
  const auto output_shape = get_output_shape(kernel_shape, input_shape, padding_type);

  // Get the memory required to store the matrices
  const size_t matrix_sizes = N_GEMMS * (
    get_input_matrix_size(kernel_shape, input_shape, padding_type) +
    get_output_matrix_size(kernel_shape, input_shape, padding_type)
  );

  // Add additional space to re-order the input and output if the input tensor
  // is not in NHWC format.
  if (input_shape.ordering == NHWC)
  {
    return matrix_sizes;  // No extra spacing required
  }
  else  // NCHW, must reorder the input and output tensors
  {
    // We only need to re-order the input or output at any one time, so request
    // enough memory to do the largest of these.
    const size_t extra_memory = std::max(
      sizeof(TIn) * input_shape.size(),
      sizeof(TOut) * output_shape.size()
    );
    return matrix_sizes + extra_memory;
  }
}


/* Get the memory required by a single "input" matrix.
 */
template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_input_matrix_size(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding_type
)
{
  return get_input_matrix_stride(kernel_shape, input_shape, padding_type) * sizeof(TGIn);
}

template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
int WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_input_matrix_stride(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding_type
)
{
  // Compute shape for the GEMM
  const auto output_shape = get_output_shape(kernel_shape, input_shape, padding_type);
  const int tile_rows = iceildiv(output_shape.n_rows, output_tile_rows);
  const int tile_cols = iceildiv(output_shape.n_cols, output_tile_cols);
  const int M = roundup(input_shape.n_batches * tile_rows * tile_cols, M_BLOCK);
  const int K = kernel_shape.n_input_channels;

  return M * K;
}


/* Get the memory required by a single "output" matrix.
 */
template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_output_matrix_size(
    const KernelShape &kernel_shape,
    const Tensor4DShape &input_shape,
    const PaddingType padding_type
)
{
  return get_output_matrix_stride(kernel_shape, input_shape, padding_type) * sizeof(TGOut);
}


template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
int WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_output_matrix_stride(
    const KernelShape &kernel_shape,
    const Tensor4DShape &input_shape,
    const PaddingType padding_type
)
{
  // Compute shape for the GEMM
  const auto output_shape = get_output_shape(kernel_shape, input_shape, padding_type);
  const int tile_rows = iceildiv(output_shape.n_rows, output_tile_rows);
  const int tile_cols = iceildiv(output_shape.n_cols, output_tile_cols);
  const int M = roundup(tile_rows * tile_cols, M_BLOCK);
  const int N = roundup(kernel_shape.n_output_channels, N_BLOCK);

  return input_shape.n_batches * M * N;
}


/* Get the memory required by a single "kernel" matrix.
 */
template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_kernel_matrix_size(const KernelShape &shape)
{
  return sizeof(TGIn) * get_kernel_matrix_stride(shape);
}

template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols, WinogradRoots roots>
template <typename TOut, typename TIn, typename TGIn, typename TGOut>
int WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols, roots>::Convolution<TOut, TIn, TGIn, TGOut>::get_kernel_matrix_stride(const KernelShape &shape)
{
  const int K = shape.n_input_channels;
  const int N = roundup(shape.n_output_channels, N_BLOCK);
  return K * N;
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
