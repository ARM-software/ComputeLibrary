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
    static constexpr int inner_tile_rows = output_tile_rows + kernel_rows - 1;  // TODO Check
    static constexpr int inner_tile_cols = output_tile_cols + kernel_cols - 1;  // TODO Check
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
    struct InputTransform
    {
      /** Get the bytes read during the transform. */
      static size_t bytes_read(const Tensor4DShape &shape)
      {
        return shape.size() * sizeof(T);
      }

      /** Get the bytes written during the transform. */
      static size_t bytes_written(const Tensor4DShape &shape)
      {
        const int M = iceildiv(shape.n_rows, inner_tile_rows) *
                      iceildiv(shape.n_cols, inner_tile_cols);
        const int K = shape.n_channels;
        return inner_tile_rows * inner_tile_cols * M * K * sizeof(T);
      }

      /** Get the count of operations performed by the transform. */
      static int ops_performed(const Tensor4DShape &shape);

      /** Apply the transform to a tensor. */
      static void execute(
          const T* const input,        /** Input tensor data */
          const int n_batches,         /** Number of batches in input tensor. */
          const int in_batch_stride,   /** Stride between batches of the input. */
          const int n_rows,            /** Number of rows in input tensor. */
          const int in_row_stride,     /** Stride between rows of the input. */
          const int n_cols,            /** Number of columns in input tensor. */
          const int in_col_stride,     /** Stride between columns of the input. */
          const int n_channels,        /** Number of channels in input tensor. */
          const PaddingType padding,   /** Padding type. */
          const int tile_M,
          const int tile_N,
          T* const output,             /** Base of output matrices. */
          const int matrix_stride,     /** Stride between output matrices. */
          const int matrix_batch_stride,  /** Stride between batches within the matrix. */
          const int matrix_row_stride  /** Stride within matrices. */
      );

      /***********************************************************************/
      /** Create an InputTransform operator fixed on a given problem and set of
       * pointers.
       */
      InputTransform(
          const T* const input,        /** Input tensor data */
          const int n_batches,         /** Number of batches in input tensor. */
          const int n_rows,            /** Number of rows in input tensor. */
          const int n_cols,            /** Number of columns in input tensor. */
          const int n_channels,        /** Number of channels in input tensor. */
          const PaddingType padding,   /** Padding type. */
          T* const output,             /** Base of output matrices. */
          const int matrix_stride,     /** Stride between output matrices. */
          const int matrix_row_stride, /** Stride within matrices. */
          const int in_batch_stride=0, /** Stride between input batches. */
          const int in_row_stride=0,   /** Stride between input rows. */
          const int in_col_stride=0    /** Stride between input columns. */
      );

      /** Get the window of work a given operator can perform. */
      unsigned int get_window() const;
      static constexpr unsigned int WINDOW_BLOCK = 16;  // Base size of window

      /** Perform work upon a window of the input. */
      void run(const unsigned int start, const unsigned int stop);
      /***********************************************************************/

      private:
        static void process_tile_row(
          const int tile_N,
          int n_channels,
          const T* const input_base,
          const int input_row_stride,
          const int input_col_stride,
          T* const matrix_base,
          const int matrix_stride,
          const int matrix_row_stride,
          const int row_pad_top,
          const int row_pad_left,
          const int row_pad_bottom,
          const int n_cols
        );

        static constexpr int max_pad_bottom = inner_tile_rows - 1;
        static constexpr int max_pad_right = inner_tile_cols - 1;

        /** Process a single tile of the input tensor. */
        template <int pad_top, int pad_left, int pad_bottom, int pad_right>
        static void process_tile(int, const T*, int, int, T*, int);

        // Array of methods to transform tiles of the input tensor.
        typedef void (*TileFn)(int, const T*, int, int, T*, int);
        static const TileFn tile_fns[2][2][max_pad_bottom][max_pad_right];

        /* Member values for instance-based API. */
        const T* const _inptr;
        T* const _outptr;
        const int _n_batches, _n_rows, _n_cols, _n_channels, _matrix_stride,
                  _matrix_row_stride, _tiles_M, _tiles_N;
        const int _in_col_stride, _in_row_stride, _in_batch_stride;
        const PaddingType _padding_type;
    };

    /** Transform output feature maps from the Winograd to the spatial domain.
     */
    template <typename T>
    struct OutputTransform
    {
      /** Get the bytes read during the transform. */
      static size_t bytes_read(const Tensor4DShape &shape);

      /** Get the bytes written during the transform. */
      static size_t bytes_written(const Tensor4DShape &shape);

      /** Get the count of operations performed by the transform. */
      static int ops_performed(const Tensor4DShape &shape);

      /** Apply the transform to create a tensor. */
      static void execute(
        const int n_batches,
        const int out_batch_stride,
        const int n_rows,
        const int out_row_stride,
        const int n_cols,
        const int out_col_stride,
        const int n_channels,
        const T* const matrix_base,
        const int matrix_stride,
        const int matrix_row_stride,
        const T* const biases,
        T* const output
      );

      /***********************************************************************/
      /** Create an OutputTransform operator fixed on a given problem and set
       * of pointers.
       */
      OutputTransform(
        const T* const matrix_base,   /** Pointer to base of matrices. */
        const int matrix_stride,      /** Stride between matrices. */
        const int matrix_row_stride,  /** Stride within a matrix. */
        const T* const biases,        /** Pointer to biases vector. */
        T* const output,              /** Pointer to output tensor. */
        const int n_batches,          /** Number of batches in output tensor. */
        const int n_rows,             /** Number of rows in output tensor. */
        const int n_cols,             /** Number of columns in output tensor. */
        const int n_channels,         /** Number of channels in output tensor. */
        const int out_batch_stride=0, /** Output batch stride. */
        const int out_row_stride=0,   /** Output row stride. */
        const int out_col_stride=0    /** Output column stride. */
      );

      /** Get the window of work a given operator can perform. */
      unsigned int get_window() const;
      static constexpr unsigned int WINDOW_BLOCK = 16;  // Base size of window

      /** Perform work upon a window of the input. */
      void run(const unsigned int start, const unsigned int stop);
      /***********************************************************************/

      private:
        static void process_tile_row(
          const int tile_N,
          const int n_channels,
          const T* const matrix_base,
          const int matrix_stride,
          const int matrix_row_stride,
          const T* const biases,
          T* const output,
          const int output_row_stride,
          const int output_col_stride,
          const int row_pad_bottom,
          const int row_pad_right
        );

        // Limits on the amount of anti-padding to be applied
        static constexpr int max_pad_bottom = output_tile_rows;
        static constexpr int max_pad_right = output_tile_cols;

        /** Prepare a single tile of the output tensor. */
        template <int pad_bottom, int pad_right>
        static void process_tile(int, const T*, int, const T*, T*, int, int);

        // Array of methods to produce tiles of output tensor.
        typedef void (*TileFn)(int, const T*, int, const T*, T*, int, int);
        static const TileFn tile_fns[max_pad_bottom][max_pad_right];

        /** Member constants for instances of the transform. */
        const T* const _matrix_base;
        const T* const _biases;
        const int _matrix_stride, _matrix_row_stride;
        T* const _outptr;
        const int _n_batches, _n_rows, _n_cols, _n_channels, _tile_M, _tile_N;
        const int _out_col_stride, _out_row_stride, _out_batch_stride;
    };

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
