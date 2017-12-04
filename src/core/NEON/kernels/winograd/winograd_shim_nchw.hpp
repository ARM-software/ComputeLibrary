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

#include "gemm.hpp"
#include "profiler.hpp"
#include "utils.hpp"
#include "shims.hpp"
#include "winograd_gemm.hpp"

#include "transforms.hpp"
 
#ifndef ALLOC_ALIGN
#define ALLOC_ALIGN 64
#endif  // ALLOC_ALIGN


namespace winograd_shim_nchw {
  /***************************************************************************/
  /* Implementation of the Winograd F(2x2, 3x3, 4x4) algorithm using GEMM
   * internally.
   */
  template <typename TOut, typename TIn>
  class Winograd2x2_3x3GEMM : public winograd::Winograd2x2_3x3GEMM<TOut, TIn> {
    public:
      /* Instantiate a new Winograd operator.
       */
      Winograd2x2_3x3GEMM(const KernelShape &kernel_shape, const Tensor4DShape input_shape, const PaddingType padding_type, void *kernel_storage);

      void nchw2nhwc( const Tensor4DShape& input_shape, const PaddingType padding_type, void *working_space, const TIn* const input);
      void nhwc2nchw( const Tensor4DShape& input_shape, const PaddingType padding_type, void *working_space, TOut* const output);


      std::pair<TOut*,TIn*> get_nhwc_ptrs(const Tensor4DShape& input_shape,const PaddingType padding_type,void *working_space);

      static size_t get_working_space_size(const Tensor4DShape &input_shape,const KernelShape &k_shape, const PaddingType padding);
    protected:
      /* Get the memory required to store an NHWC copy of the input tensor. */
      static size_t get_working_nhwc_input_size(const Tensor4DShape &input_shape);

      /* Get the memory required to store an NHWC copy of the input tensor. */
      static size_t get_working_nhwc_output_size(const Tensor4DShape &output_shape, const KernelShape &k_shape, const PaddingType padding) ;
  };
} // namespace winograd

/*****************************************************************************/
template <typename TOut, typename TIn>
winograd_shim_nchw::Winograd2x2_3x3GEMM<TOut, TIn>::Winograd2x2_3x3GEMM(
    const KernelShape &kernel_shape, const Tensor4DShape input_shape,
        const PaddingType padding_type, void *kernel_storage
) : winograd::Winograd2x2_3x3GEMM<TOut, TIn>(kernel_shape,input_shape,padding_type,kernel_storage) {
}

/*****************************************************************************/
template <typename TOut, typename TIn>
void winograd_shim_nchw::Winograd2x2_3x3GEMM<TOut, TIn>::nchw2nhwc(const Tensor4DShape& input_shape, const PaddingType padding_type, void *working_space, const TIn* const input) {
  assert(working_space);
  int8_t* const ws_bytes = reinterpret_cast<int8_t *>(working_space);

  // Extract the top chunk of the working space to store the input and output
  // tensors in NHWC format.
  const int in_matrix_stride_bytes = winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_input_matrix_size(input_shape, this->kernel_shape, padding_type);
  const int out_matrix_stride_bytes = winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_output_matrix_size(input_shape, this->kernel_shape, padding_type);

  // Allocate working space for the input and output in NHWC format
  TIn* const input_nhwc = reinterpret_cast<TIn *>(
      ws_bytes + 16*(in_matrix_stride_bytes + out_matrix_stride_bytes)
  );

  // Re-order the input tensor
  this->prof(
    "NCHW -> NHWC",
    [input, input_shape, input_nhwc] () {
      nchw_to_nhwc(
        input, input_nhwc,
        input_shape.n_batches,
        input_shape.n_channels,
        input_shape.n_rows,
        input_shape.n_cols
      );
    },
    input_shape.size(), 0, input_shape.size()
  );
}

/*****************************************************************************/
template <typename TOut, typename TIn>
void winograd_shim_nchw::Winograd2x2_3x3GEMM<TOut, TIn>::nhwc2nchw(const Tensor4DShape& input_shape, const PaddingType padding_type, 
            void *working_space, TOut* const output) {

  assert(working_space);
  int8_t* const ws_bytes = reinterpret_cast<int8_t *>(working_space);

  // Extract the top chunk of the working space to store the input and output
  // tensors in NHWC format.
  const int in_matrix_stride_bytes = winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_input_matrix_size(input_shape, this->kernel_shape, padding_type);
  const int out_matrix_stride_bytes = winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_output_matrix_size(input_shape, this->kernel_shape, padding_type);

  TOut* const output_nhwc = reinterpret_cast<TOut *>(ws_bytes + 16*(in_matrix_stride_bytes + out_matrix_stride_bytes) + get_working_nhwc_input_size(input_shape));

  // Re-order the output tensor into NCHW
  const auto output_shape = winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_output_shape(input_shape, this->kernel_shape, padding_type);
  this->prof(
    "NHWC -> NCHW",
    [output_nhwc, output_shape, output] () {
      nhwc_to_nchw(
        output_nhwc, output,
        output_shape.n_batches,
        output_shape.n_rows,
        output_shape.n_cols,
        output_shape.n_channels
      );
    },
    output_shape.size(), 0, output_shape.size()
  );
}


/*****************************************************************************/
template <typename TOut, typename TIn>
std::pair<TOut*,TIn*> winograd_shim_nchw::Winograd2x2_3x3GEMM<TOut, TIn>::get_nhwc_ptrs(
    const Tensor4DShape& input_shape,
    const PaddingType padding_type,
    void *working_space
) {
  assert(working_space);
  int8_t* const ws_bytes = reinterpret_cast<int8_t *>(working_space);

  // Extract the top chunk of the working space to store the input and output
  // tensors in NHWC format.
  const int in_matrix_stride_bytes = winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_input_matrix_size(input_shape, this->kernel_shape, padding_type);
  const int out_matrix_stride_bytes = winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_output_matrix_size(input_shape, this->kernel_shape, padding_type);

  // Allocate working space for the input and output in NHWC format
  TIn* input_nhwc = reinterpret_cast<TIn *>(ws_bytes + 16*(in_matrix_stride_bytes + out_matrix_stride_bytes));
  TOut* output_nhwc = reinterpret_cast<TOut *>(ws_bytes + 16*(in_matrix_stride_bytes + out_matrix_stride_bytes) + get_working_nhwc_input_size(input_shape));
  return std::make_pair(output_nhwc,input_nhwc);
}




/*****************************************************************************/
template <typename TOut, typename TIn>
size_t winograd_shim_nchw::Winograd2x2_3x3GEMM<TOut, TIn>::get_working_space_size(
    const Tensor4DShape& input_shape, const KernelShape &k_shape, const PaddingType padding_type
)  {
  // TODO Add memory required for NHWC copies of input tensors
  return winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_working_space_size(
      input_shape, k_shape, padding_type)
      + get_working_nhwc_input_size(input_shape)
      + get_working_nhwc_output_size(input_shape, k_shape, padding_type);
}

template <typename TOut, typename TIn>
size_t winograd_shim_nchw::Winograd2x2_3x3GEMM<TOut, TIn>::get_working_nhwc_input_size(
    const Tensor4DShape& input_shape
)  {
  return roundup(input_shape.size() * sizeof(TIn), static_cast<size_t>(ALLOC_ALIGN));
}

template <typename TOut, typename TIn>
size_t winograd_shim_nchw::Winograd2x2_3x3GEMM<TOut, TIn>::get_working_nhwc_output_size(
    const Tensor4DShape& input_shape, const KernelShape &k_shape, const PaddingType padding_type
)  {
  const auto output_shape = winograd::Winograd2x2_3x3GEMM<TOut, TIn>::get_output_shape(input_shape,k_shape, padding_type);
  return roundup(output_shape.size() * sizeof(TIn), static_cast<size_t>(ALLOC_ALIGN));
}
