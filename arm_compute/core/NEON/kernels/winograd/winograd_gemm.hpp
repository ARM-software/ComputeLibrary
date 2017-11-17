/*
 * Copyright (c) 2017 ARM Limited.
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
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cassert>

#include "alloc.hpp"
#include "gemm.hpp"
#include "profiler.hpp"
#include "utils.hpp"
#include "shims.hpp"

#include "transforms.hpp"

namespace winograd {
  /***************************************************************************/
  /* Implementation of the Winograd F(2x2, 3x3, 4x4) algorithm using GEMM
   * internally.
   */
  template <typename TOut, typename TIn>
  class Winograd2x2_3x3GEMM {
    public:
      /* Instantiate a new Winograd operator.
       */
      Winograd2x2_3x3GEMM(const KernelShape &kernel_shape, const Tensor4DShape input_shape, const PaddingType padding_type, void *kernel_storage);
      virtual ~Winograd2x2_3x3GEMM();

      /** Transform the weights into the Winograd domain.
       */
      template <typename KernelTransform=winograd2x2_3x3_gemm_kernel_transform_impl<TIn>>
      void transform_weights(const TIn* const kernel, void *transform_working_space);

      /* Initializes matrices pointers, to be called once before execute()
       */
      template <typename InputTransform=Winograd2x2_3x3GemmInputChannelwise<TIn>>
      void reshape_input(const Tensor4DShape &input_shape, const PaddingType padding_type, const TIn* const input, void* working_space);

      /* Apply the Winograd operator to some input.
       */
      template <typename OutputTransform=Winograd2x2_3x3GemmOutput<TOut>>
      void reshape_output(const Tensor4DShape& input_shape, const PaddingType padding_type, TOut* const output);


      /* Apply the Winograd operator to some input.
       */
      void execute(size_t first, size_t last);

      /* Get the memory required to transform the kernel.
       */
      static inline size_t get_kernel_transform_working_size(const KernelShape &shape);

      /* Get the output shape of a convolution.
       */
      static Tensor4DShape get_output_shape(const Tensor4DShape &input_shape, const KernelShape &k_shape,
                                     const PaddingType padding_type);

      /* Get the memory required to instantiate a new Winograd operator.
       */
      static size_t get_kernel_storage_size(const KernelShape &shape);

      /* Get the memory required to apply a Winograd operator to some input.
       */
      static size_t get_working_space_size(const Tensor4DShape &input_shape,const KernelShape &k_shape,
                                    const PaddingType padding);


      Winograd2x2_3x3GEMM(const Winograd2x2_3x3GEMM &) = delete;
      /** Prevent instances of this class from being copied (As this class contains pointers) */
      Winograd2x2_3x3GEMM &operator=(const Winograd2x2_3x3GEMM &) = delete;
      /** Allow instances of this class to be moved */
      Winograd2x2_3x3GEMM(Winograd2x2_3x3GEMM &&) = default;
      /** Allow instances of this class to be moved */
      Winograd2x2_3x3GEMM &operator=(Winograd2x2_3x3GEMM &&) = default;

    protected:
      /* Get the memory required by a single "input" matrix.
       */
      static size_t get_input_matrix_size(const Tensor4DShape &input_shape,const KernelShape &k_shape,
                                   const PaddingType padding);

      /* Get the memory required by a single "output" matrix.
       */
      static size_t get_output_matrix_size(const Tensor4DShape &input_shape, const KernelShape &k_shape,
                                    const PaddingType padding);

      /* Get the memory required by a single "kernel" matrix.
       */
      static size_t get_kernel_matrix_size(const KernelShape &shape);

      const KernelShape kernel_shape;  // Shape of applied kernel
      const Tensor4DShape in_shape;
      const PaddingType padding;

      const int kernel_matrix_row_stride;  // Stride within kernel matrix

      const bool manage_kernel_storage;  // Free kernel storage when done
      void* const _kernel_storage;  // Base pointer for kernel matrices

      profiler prof;  // Profiler

      TIn *kernel_matrices[16];  // Prepared form of kernel
      TIn *input_matrices[16];
      TOut *output_matrices[16];


      static const int M_BLOCK = 4;
      static const int N_BLOCK = 16;
  };
} // namespace winograd

template <typename TOut, typename TIn>
size_t winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_kernel_transform_working_size(
    const KernelShape &shape
)
{
    // Need to re-order the kernel into HWIO form, require enough space to
    // represent the tensor.
    return sizeof(TIn) * shape.size();
}


template <typename TOut, typename TIn>
template <typename KernelTransform>
void winograd::Winograd2x2_3x3GEMM<TOut, TIn>::transform_weights(
  const TIn* const kernel,
  void *transform_working_space
)
{
    const int kernel_matrix_size_bytes = get_kernel_matrix_size(kernel_shape);
    int8_t* const ks_bytes = reinterpret_cast<int8_t *>(_kernel_storage);
    for (int i = 0; i < 16; i++) {
        kernel_matrices[i] = reinterpret_cast<TIn *>(
        ks_bytes + i*kernel_matrix_size_bytes);
    }

    const TIn *kernel_hwio = kernel;
    if( transform_working_space)
    {
            kernel_hwio = reinterpret_cast<TIn *>(transform_working_space);
            ofm_ifm_h_w_to_h_w_ifm_ofm(
                  kernel, const_cast<TIn *>(kernel_hwio),
                  kernel_shape.n_output_channels,
                  kernel_shape.n_input_channels,
                  kernel_shape.n_rows,
                  kernel_shape.n_cols
                );
    }
    KernelTransform::execute(
      kernel_shape, kernel_hwio, kernel_matrices[0],
      kernel_matrix_size_bytes / sizeof(TIn),
      kernel_matrix_row_stride
    );
}

template <typename TOut, typename TIn>
winograd::Winograd2x2_3x3GEMM<TOut, TIn>::Winograd2x2_3x3GEMM( const KernelShape &kernel_shape, const Tensor4DShape input_shape,
        const PaddingType padding_type, void *kernel_storage)
    : kernel_shape(kernel_shape), in_shape(input_shape), padding(padding_type),kernel_matrix_row_stride(roundup(kernel_shape.n_output_channels, N_BLOCK)), manage_kernel_storage(false),
        _kernel_storage(kernel_storage), prof() {
     memset(kernel_matrices, 0x00, sizeof(TIn)*16);
     memset(input_matrices, 0x00, sizeof(TIn)*16);
     memset(output_matrices, 0x00, sizeof(TOut)*16);
}

/*****************************************************************************/
template <typename TOut, typename TIn>
winograd::Winograd2x2_3x3GEMM<TOut, TIn>::~Winograd2x2_3x3GEMM() {}

/*****************************************************************************/
template <typename TOut, typename TIn>
template <typename InputTransform>
void winograd::Winograd2x2_3x3GEMM<TOut, TIn>::reshape_input(
    const Tensor4DShape& input_shape,
    const PaddingType padding_type,
    const TIn* const input,
    void *working_space
) {
  assert(working_space);
  int8_t* const ws_bytes = reinterpret_cast<int8_t *>(working_space);
  // Split the working space into that required for 16 input matrices and
  // output matrices.
  const int in_matrix_stride_bytes = get_input_matrix_size(input_shape, kernel_shape, padding_type);
  const int out_matrix_stride_bytes = get_output_matrix_size(input_shape, kernel_shape, padding_type);

  for (int i = 0; i < 16; i++) {
    input_matrices[i] = reinterpret_cast<TIn *>(
        ws_bytes + i*in_matrix_stride_bytes);
    output_matrices[i] = reinterpret_cast<TIn *>(
        ws_bytes + 16*in_matrix_stride_bytes + i*out_matrix_stride_bytes);
  }

  // Compute shape for the GEMM
  const auto output_shape = get_output_shape(input_shape,kernel_shape, padding_type);
  const int tile_rows = iceildiv(output_shape.n_rows, 2);
  const int tile_cols = iceildiv(output_shape.n_cols, 2);
  const int K = kernel_shape.n_input_channels;

  const int in_matrix_row_stride = K;
  const int in_matrix_batch_stride = tile_rows*tile_cols*in_matrix_row_stride;

  // Transform the input tensor into an appropriate form
  auto input_prep = [&] () {
    InputTransform::execute(
      input, input_shape, padding_type, tile_rows, tile_cols,
      input_matrices[0], in_matrix_stride_bytes / sizeof(TIn),
      in_matrix_batch_stride, in_matrix_row_stride
    );
  };
  prof(
    "Input Prep", input_prep,
    InputTransform::bytes_read(input_shape, output_shape),
    InputTransform::flops_performed(input_shape, output_shape),
    InputTransform::bytes_written(input_shape, output_shape)
  );

}

/*****************************************************************************/
template <typename TOut, typename TIn>
template <typename OutputTransform>
void winograd::Winograd2x2_3x3GEMM<TOut, TIn>::reshape_output(const Tensor4DShape& input_shape, const PaddingType padding_type, TOut* const output) {
  assert(output_matrices[0]);
  const int out_matrix_stride_bytes = get_output_matrix_size(input_shape, kernel_shape, padding_type);
  const auto output_shape = get_output_shape(input_shape,kernel_shape, padding_type);
  const int out_matrix_row_stride = kernel_matrix_row_stride;

  // Transform the output tensor into an appropriate form
    OutputTransform::execute(
      output_shape,
      output_matrices[0],
      out_matrix_stride_bytes / sizeof(TOut),
      out_matrix_row_stride,
      output
    );
}


/*****************************************************************************/
template <typename TOut, typename TIn>
void winograd::Winograd2x2_3x3GEMM<TOut, TIn>::execute( size_t first, size_t last ) {
  assert(input_matrices[0] && kernel_matrices[0] && output_matrices[0]);
  assert(first < 16 && last < 16 && first < last);
  // Compute shape for the GEMM
  const auto output_shape = get_output_shape(in_shape,kernel_shape, padding);
  const int tile_rows = iceildiv(output_shape.n_rows, 2);
  const int tile_cols = iceildiv(output_shape.n_cols, 2);
  const int M = in_shape.n_batches * tile_rows * tile_cols;
  const int K = kernel_shape.n_input_channels;
  const int N = kernel_shape.n_output_channels;

  const int in_matrix_row_stride = K;
  const int out_matrix_row_stride = kernel_matrix_row_stride;
  // Perform the GEMMs
  for (size_t i = first; i <= last; i++) {
      BlockedGemm<M_BLOCK, N_BLOCK>(
        input_matrices[i], kernel_matrices[i], output_matrices[i], M, K, N,
        in_matrix_row_stride, kernel_matrix_row_stride, out_matrix_row_stride
      );
//    prof("GEMM", perform_gemm, 0, 2*M*K*N, 0);  // TODO Memory
  }

}

/*****************************************************************************/
template <typename TOut, typename TIn>
Tensor4DShape winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_output_shape(
    const Tensor4DShape &in_shape, const KernelShape &k_shape, const PaddingType padding)  {
  return Tensor4DShape {
    in_shape.n_batches,
    (padding == PADDING_SAME) ? in_shape.n_rows : in_shape.n_rows - 2,
    (padding == PADDING_SAME) ? in_shape.n_cols : in_shape.n_cols - 2,
    k_shape.n_output_channels
  };
}

template <typename TOut, typename TIn>
size_t winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_kernel_storage_size(
    const KernelShape &shape) {
  return 16 * get_kernel_matrix_size(shape);
}

template <typename TOut, typename TIn>
size_t winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_kernel_matrix_size(
    const KernelShape &shape) {
  const int K = shape.n_input_channels;
  const int N = roundup(shape.n_output_channels, N_BLOCK);
  return sizeof(TIn) * K * N;
}

template <typename TOut, typename TIn>
size_t winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_working_space_size(
    const Tensor4DShape& input_shape, const KernelShape &k_shape, const PaddingType padding_type
)  {
  return 16 * get_input_matrix_size(input_shape, k_shape, padding_type) +
         16 * get_output_matrix_size(input_shape, k_shape, padding_type);
}

template <typename TOut, typename TIn>
size_t winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_input_matrix_size(
    const Tensor4DShape& input_shape, const KernelShape &k_shape, const PaddingType padding_type
)  {
  // Compute shape for the GEMM
  const auto output_shape = get_output_shape(input_shape, k_shape, padding_type);
  const int tile_rows = iceildiv(output_shape.n_rows, 2);
  const int tile_cols = iceildiv(output_shape.n_cols, 2);
  const int M = roundup(tile_rows * tile_cols, M_BLOCK);
  const int K = k_shape.n_input_channels;

  return input_shape.n_batches * M * K * sizeof(TIn);
}

template <typename TOut, typename TIn>
size_t winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_output_matrix_size(
    const Tensor4DShape& input_shape, const KernelShape &k_shape,const PaddingType padding_type
)  {
  // Compute shape for the GEMM
  const auto output_shape = get_output_shape(input_shape, k_shape, padding_type);
  const int tile_rows = iceildiv(output_shape.n_rows, 2);
  const int tile_cols = iceildiv(output_shape.n_cols, 2);
  const int M = roundup(tile_rows * tile_cols, M_BLOCK);
  const int N = roundup(k_shape.n_output_channels, N_BLOCK);

  return input_shape.n_batches * M * N * sizeof(TOut);
}
