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

#include "arm_compute/core/NEON/kernels/convolution/common/alloc.hpp"
#include "arm_compute/core/NEON/kernels/convolution/common/convolution.hpp"
#include "gemm.hpp"
#include "arm_compute/core/NEON/kernels/convolution/common/shims.hpp"
#include "arm_compute/core/NEON/kernels/convolution/common/tensor.hpp"
#include "arm_compute/core/NEON/kernels/convolution/common/utils.hpp"
#include "winograd_input_transform.hpp"
#include "winograd_output_transform.hpp"

#include <thread>
#include <utility>
#include <vector>

// Generic Winograd implementation using GEMM
namespace winograd
{

template <int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class WinogradGEMM
{
  public:
    // Information about the specific Winograd instance
    static constexpr int output_tile_rows = OutputTileRows;
    static constexpr int output_tile_cols = OutputTileCols;
    static constexpr int kernel_rows = KernelRows;
    static constexpr int kernel_cols = KernelCols;
    static constexpr int inner_tile_rows = output_tile_rows + kernel_rows - 1;
    static constexpr int inner_tile_cols = output_tile_cols + kernel_cols - 1;
    static constexpr int N_GEMMS = inner_tile_rows * inner_tile_cols;

    /** Transform weights from the spatial to the Winograd domain. */
    template <typename T>
    struct WeightsTransform
    {
      /** Get the bytes read during the transform. */
      static inline size_t bytes_read(const KernelShape &shape)
      {
        return shape.size() * sizeof(T);
      }

      /** Get the bytes written during the transform. */
      static inline size_t bytes_written(const KernelShape &shape)
      {
        const int inner_tile_size = inner_tile_rows * inner_tile_cols;
        return (inner_tile_size * shape.n_input_channels *
                shape.n_output_channels * sizeof(T));
      }

      /** Get the count of operations performed by the transform. */
      static int ops_performed(const KernelShape &shape);

      /** Apply the transform to a tensor. */
      static void execute(
        const int n_output_channels,
        const int n_input_channels,
        const T* const input,
        T* const output,
        const int matrix_stride,
        const int matrix_row_stride
      );

      /** Create a WeightsTransform operator fixed on a given problem and set
       * of pointers.
       */
      WeightsTransform(
        const T* const input,
        T* const output,
        const int matrix_stride,       /** Stride across matrices in the output. */
        const int matrix_row_stride,   /** Stride across rows of the matrix. */
        const int n_output_channels,   /** Number of filters. */
        const int n_input_channels     /** Number of channels in each filter. */
      );

      /** Get the window of work a given operator can perform. */
      unsigned int get_window() const;

      /** Perform work upon a window of the input. */
      void run(const unsigned int start, const unsigned int stop);

      private:
        const T* const inptr;         /** Fixed pointer to input data. */
        T* const outptr;              /** Fixed pointer to output memory. */
        const int matrix_stride;      /** Stride between output matrices. */
        const int matrix_row_stride;  /** Stride within output matrices. */
        const int n_output_channels;  /** Number of filters. */
        const int n_input_channels;   /** Number of channels in each filter. */
    };

    /** Transform input feature maps from the spatial to the Winograd domain.
     */
    template <typename T>
    using InputTransform = InputTransform<
      KernelRows, KernelCols,
      (OutputTileRows + KernelRows - 1),
      (OutputTileCols + KernelCols - 1),
      T
    >;

    /** Transform output feature maps from the Winograd to the spatial domain.
     */
    template <typename T>
     using OutputTransform = OutputTransform<
      KernelRows, KernelCols,
      (OutputTileRows + KernelRows - 1),
      (OutputTileCols + KernelCols - 1),
      T
    >;


    /** Perform a convolution.
     */
    template <typename TOut, typename TIn>
    class Convolution
    {
      public:
        // Information about the typed Winograd instance
        typedef TOut OutputType;
        typedef TIn InputType;

        /** Get the output shape of a convolution. */
        static Tensor4DShape get_output_shape(
          const KernelShape &kernel_shape,
          const Tensor4DShape &in_shape,
          const PaddingType padding
        );

        /* Get the memory required to transform the kernel.
         */
        static size_t get_kernel_transform_working_size(const KernelShape &shape);

        /** Get the memory required to store the kernel transformed into the
         * Winograd domain.
         */
        static size_t get_kernel_storage_size(const KernelShape &shape);

        /** Get the memory required to store the input tensor transformed into
         * the Winograd domain.
         */
        static size_t get_input_storage_size(
          const KernelShape &kernel_shape,
          const Tensor4DShape &input_shape,
          const PaddingType padding_type
        );

        /** Get the memory required to store the output tensor in the Winograd
         * domain.
         */
        static size_t get_output_storage_size(
          const KernelShape &kernel_shape,
          const Tensor4DShape &input_shape,
          const PaddingType padding_type
        );

        /** Get the memory required to apply a Winograd operator to some input.
         */
        static size_t get_working_space_size(
          const KernelShape &kernel_shape,
          const Tensor4DShape &input_shape,
          const PaddingType padding_type
        );

        /* Get the memory required by a single "input" matrix.
         */
        static size_t get_input_matrix_size(
          const KernelShape &kernel_shape,
          const Tensor4DShape &input_shape,
          const PaddingType padding_type
        );

        static int get_input_matrix_stride(
          const KernelShape &kernel_shape,
          const Tensor4DShape &input_shape,
          const PaddingType padding_type
        );

        /* Get the memory required by a single "output" matrix.
         */
        static size_t get_output_matrix_size(
          const KernelShape &kernel_shape,
          const Tensor4DShape &input_shape,
          const PaddingType padding_type
        );

        static int get_output_matrix_stride(
          const KernelShape &kernel_shape,
          const Tensor4DShape &input_shape,
          const PaddingType padding_type
        );

        /* Get the memory required by a single "kernel" matrix.
         */
        static size_t get_kernel_matrix_size(const KernelShape &shape);
        static int get_kernel_matrix_stride(const KernelShape &shape);

        static constexpr int M_BLOCK = 4;   /** Size of block used by GEMM. */
        static constexpr int N_BLOCK = 16;  /** Size of block used by GEMM. */
    };
};

}  // namespace winograd
