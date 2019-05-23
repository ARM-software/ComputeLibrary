/*
 * Copyright (c) 2019 ARM Limited.
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

#include "convolution.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace winograd
{

class ITransform
{
  public:
    virtual ~ITransform() = default;

    /**
     * Get the working space required to perform the transformation.
     *
     * Note, the working space is only required when performing the
     * transformation - hence it can be reused whenever the transformation is
     * not running.
     *
     * @param nthreads The greatest number of threads that will be used to execute the transform.
     * @return Size of working space required in bytes.
     */
    virtual size_t get_working_space_size(unsigned int nthreads=1) const = 0;

    /**
     * Set the working space to be used by the transformation.
     *
     * Note, the working space is only required when performing the
     * transformation - hence it can be reused whenever the transformation is
     * not running.
     *
     * @param Pointer to the working space.
     */
    virtual void set_working_space(void *buffer) = 0;

    /**
     * Get the window of work a given operator can perform.
     */
    virtual unsigned int get_window() const = 0;

    /**
     * Perform work upon a window of the transform.
     */
    virtual void run(unsigned int start, unsigned int stop, unsigned int threadid=0) = 0;
};

class IInputTransform : public ITransform
{
  public:
    virtual ~IInputTransform() = default;

    /**
     * Set the pointer to the (NHWC-ordered) tensor to be transformed.
     */
    virtual void set_input_tensor(const void *input) = 0;

    /**
     * Set the pointer to the (NHWC-ordered) tensor to be transformed.
     * @param col_stride Stride between columns of the tensor, measured in elements (not bytes).
     */
    virtual void set_input_tensor(const void *input, int col_stride) = 0;

    /**
     * Set the pointer to the (NHWC-ordered) tensor to be transformed.
     * @param row_stride Stride between rows of the tensor, measured in elements (not bytes).
     * @param col_stride Stride between columns of the tensor, measured in elements (not bytes).
     */
    virtual void set_input_tensor(const void *input, int row_stride, int col_stride) = 0;

    /**
     * Set the pointer to the (NHWC-ordered) tensor to be transformed.
     * @param batch_stride Stride between batches of the tensor, measured in elements (not bytes).
     * @param row_stride Stride between rows of the tensor, measured in elements (not bytes).
     * @param col_stride Stride between columns of the tensor, measured in elements (not bytes).
     */
    virtual void set_input_tensor(const void *input, int batch_stride, int row_stride, int col_stride) = 0;

    /**
     * Set pointers to the matrices written by the transform.
     * @param matrices Pointer to the start of the first matrix representing the transformed input.
     * @param inter_matrix_stride Stride (in elements) between matrices.
     * @param matrix_row_stride Stride (in elements) between the rows within a single matrix.
     */
    virtual void set_output_matrices(void *matrices, int inter_matrix_stride, int matrix_row_stride) = 0;
};

class IOutputTransform : public ITransform
{
  public:
    virtual ~IOutputTransform() = default;

    /**
     * Set pointers to the matrices written by the transform.
     * @param matrices Pointer to the start of the first matrix representing the input to the transform.
     * @param inter_matrix_stride Stride (in elements) between matrices.
     * @param matrix_row_stride Stride (in elements) between the rows within a single matrix.
     */
    virtual void set_input_matrices(const void *matrices, int inter_matrix_stride, int matrix_row_stride) = 0;

    /**
     * Set pointer to the bias tensor (can be ignored or called with nullptr for no bias.
     */
    virtual void set_bias(const void *bias=nullptr) = 0;

    /**
     * Set pointer to the output tensor produced by the transform.
     */
    virtual void set_output_tensor(void *output) = 0;

    /**
     * Set pointer to the output tensor produced by the transform.
     * @param col_stride Stride between columns of the tensor, measured in elements (not bytes).
     */
    virtual void set_output_tensor(void *output, int col_stride) = 0;

    /**
     * Set pointer to the output tensor produced by the transform.
     * @param row_stride Stride between rows of the tensor, measured in elements (not bytes).
     * @param col_stride Stride between columns of the tensor, measured in elements (not bytes).
     */
    virtual void set_output_tensor(void *output, int row_stride, int col_stride) = 0;

    /**
     * Set pointer to the output tensor produced by the transform.
     * @param batch_stride Stride between batches of the tensor, measured in elements (not bytes).
     * @param row_stride Stride between rows of the tensor, measured in elements (not bytes).
     * @param col_stride Stride between columns of the tensor, measured in elements (not bytes).
     */
    virtual void set_output_tensor(void *output, int batch_stride, int row_stride, int col_stride) = 0;
};

class IWeightTransform : public ITransform
{
  public:
    virtual ~IWeightTransform() = default;

    /** Set pointer to the weight tensor read by the transform. */
    virtual void set_weight_tensor(const void *weights) = 0;

    /**
     * Set pointers to the matrices written by the transform.
     * @param matrices Pointer to the start of the first matrix representing the transformed input.
     * @param inter_matrix_stride Stride (in elements) between matrices.
     * @param matrix_row_stride Stride (in elements) between the rows within a single matrix.
     */
    virtual void set_output_matrices(void *matrices, int inter_matrix_stride, int matrix_row_stride) = 0;
};

enum class WinogradRoots
{
  Integers,
};

template <int InnerTileRows, int InnerTileCols, typename TIn, typename TOut, WinogradRoots Roots>
class InputTransform : public IInputTransform
{
  public:
    /** Create an InputTransform operator fixed on a given problem and set of
     * pointers.
     */
    InputTransform(
        int kernel_rows,     /**< Number of rows in the kernel */
        int kernel_cols,     /**< Number of columns in the kernel */
        int n_batches,       /**< Number of batches in input tensor. */
        int n_rows,          /**< Number of rows in input tensor. */
        int n_cols,          /**< Number of columns in input tensor. */
        int n_channels,      /**< Number of channels in input tensor. */
        int padding_top,     /**< Padding to apply to the top of the image. */
        int padding_left,    /**< Padding to apply to the left of the image. */
        int padding_bottom,  /**< Padding to apply to the bottom of the image. */
        int padding_right    /**< Padding to apply to the right of the image. */
    );

    InputTransform(InputTransform&) = delete;
    InputTransform operator=(InputTransform&) = delete;

    /** Set pointers to the input tensor read by the transform. */
    void set_input_tensor(const void *input) override;
    void set_input_tensor(const void *input, int col_stride) override;
    void set_input_tensor(const void *input, int row_stride, int col_stride) override;
    void set_input_tensor(const void *input, int batch_stride, int row_stride, int col_stride) override;

    /** Set pointers to the matrices written by the transform. */
    void set_output_matrices(void *matrices, int iter_matrix_stride, int matrix_row_stride) override;

    /** Get the working space required to perform the transformation. */
    size_t get_working_space_size(unsigned int nthreads=1) const override;
    void set_working_space(void *buffer) override;

    /** Get the window of work a given operator can perform. */
    unsigned int get_window() const override;
    static constexpr unsigned int WINDOW_BLOCK = 16;  // Base size of window

    /** Perform work upon a window of the input. */
    void run(unsigned int start, unsigned int stop, unsigned int threadid=0) override;

  protected:
    const int _n_batches, _n_rows, _n_cols, _n_channels;

  private:
    void transform_unpadded_tile(
      unsigned int threadid,
      int n_channels,
      TOut *outptr,
      const TIn *inptr
    );

    void transform_padded_tile(
      unsigned int threadid,
      int n_channels,
      TOut *outptr,
      const TIn *inptr,
      int padding_top,
      int padding_left,
      int padding_bottom,
      int padding_right
    );
    
    /* Tile implementation */
    static void transform_tile(
      int n_channels,         /** @param[in] Number of channels in the tensor. */
      const TIn* inptr_base,  /** @param[in] Pointer to the base of the input tile. */
      int input_row_stride,   /** @param[in] Stride between rows of the input tensor. */
      int input_col_stride,   /** @param[in] Stride between columns of the input tensor. */
      TOut* mptr_base,        /** @param[out] Base pointer to transformed input matrices. */
      int matrix_stride       /** @param[in] Stride between matrices in the input space. */
    );

    /** Get the working space for a thread. */
    void * get_working_space(unsigned int threadid) const;

    const TIn* _inptr;
    TOut* _outptr;

    const int _overlap_rows, _overlap_cols;
    const int _padding_top, _padding_left, _padding_bottom, _padding_right;
    const int _tiles_M, _tiles_N;
    int _matrix_stride, _matrix_row_stride, _matrix_batch_stride;
    int _in_col_stride, _in_row_stride, _in_batch_stride;

    const int _working_space_col_stride, _working_space_row_stride;
    TIn *_working_space;
};

template <int InnerTileRows, typename TIn, typename TOut, WinogradRoots Roots>
class InputTransform<InnerTileRows, 1, TIn, TOut, Roots> :
  public InputTransform<1, InnerTileRows, TIn, TOut, Roots>
{
  using Base = InputTransform<1, InnerTileRows, TIn, TOut, Roots>;

  public:
    InputTransform(
      int kernel_rows,     /**< Number of rows in the kernel. */
      int kernel_cols,     /**< Number of columns in the kernel. */
      int n_batches,       /**< Number of batches in input tensor. */
      int n_rows,          /**< Number of rows in input tensor. */
      int n_cols,          /**< Number of columns in input tensor. */
      int n_channels,      /**< Number of channels in input tensor. */
      int padding_top,     /**< Padding to apply to the top of the image. */
      int padding_left,    /**< Padding to apply to the left of the image. */
      int padding_bottom,  /**< Padding to apply to the bottom of the image. */
      int padding_right    /**< Padding to apply to the right of the image. */
    );

    /** Set pointers to the input tensor read by the transform. */
    void set_input_tensor(const void *input) override;
    void set_input_tensor(const void *input, int col_stride) override;
    void set_input_tensor(const void *input, int row_stride, int col_stride) override;
    void set_input_tensor(const void *input, int batch_stride, int row_stride, int col_stride) override;
};

template <
  int KernelRows, int KernelCols,
  int InnerTileRows, int InnerTileCols,
  typename TIn, typename TOut,
  WinogradRoots Roots
>
class OutputTransform : public IOutputTransform
{
  public:
    OutputTransform(
      int n_batches,  /**< Number of batches in output tensor. */
      int n_rows,     /**< Number of rows in output tensor. */
      int n_cols,     /**< Number of columns in output tensor. */
      int n_channels  /**< Number of channels in output tensor. */
    );

    OutputTransform(OutputTransform&) = delete;
    OutputTransform operator=(OutputTransform&) = delete;

    /** Set pointers to the matrices read by the transform. */
    void set_input_matrices(const void *matrices, int iter_matrix_stride, int matrix_row_stride) override;

    /** Set pointer to the bias tensor (can be ignored or called with nullptr for no bias */
    void set_bias(const void *bias=nullptr) override;

    /** Set pointers to the output tensor written by the transform. */
    void set_output_tensor(void *output) override;
    void set_output_tensor(void *output, int col_stride) override;
    void set_output_tensor(void *output, int row_stride, int col_stride) override;
    void set_output_tensor(void *output, int batch_stride, int row_stride, int col_stride) override;

    /** Get the working space required to perform the transformation. */
    size_t get_working_space_size(unsigned int nthreads=1) const override;
    void set_working_space(void *buffer) override;

    /** Get the window of work a given operator can perform. */
    unsigned int get_window() const override;
    static constexpr unsigned int WINDOW_BLOCK = 16;  // Base size of window

    /** Perform work upon a window of the input. */
    void run(unsigned int start, unsigned int stop, unsigned int threadid=0) override;

  protected:
    static constexpr int inner_tile_rows = InnerTileRows;
    static constexpr int inner_tile_cols = InnerTileCols;
    static constexpr int output_tile_rows = InnerTileRows - KernelRows + 1;
    static constexpr int output_tile_cols = InnerTileCols - KernelCols + 1;

    const int _n_batches, _n_rows, _n_cols, _n_channels;

  private:
    void transform_uncropped_tile(
      unsigned int threadid,
      int n_channels,
      TOut *outptr,
      const TIn *inptr,
      const TOut *biases
    );

    void transform_cropped_tile(
      unsigned int threadid,
      int n_channels,
      TOut *outptr,
      const TIn *inptr,
      const TOut *biases,
      int pad_bottom,
      int pad_right
    );

    /** Implementation of the tile transformation method. */
    static void transform_tile(
      int n_channels,
      const TIn* matrix_base,
      int matrix_stride,
      const TOut* biases,
      TOut* output,
      int output_row_stride,
      int output_col_stride
    );

    /** Get the working space for a thread. */
    void * get_working_space(unsigned int threadid) const;

    const TIn* _matrix_base;
    const TOut* _biases;
    int _matrix_stride, _matrix_row_stride, _matrix_batch_stride;
    TOut* _outptr;
    const int _tiles_M, _tiles_N;
    int _out_col_stride, _out_row_stride, _out_batch_stride;

    const int _working_space_col_stride, _working_space_row_stride;
    TOut *_working_space;
};

template <
  int KernelRows,
  int InnerTileRows,
  typename TIn, typename TOut,
  WinogradRoots Roots
>
class OutputTransform<KernelRows, 1, InnerTileRows, 1, TIn, TOut, Roots> :
  public OutputTransform<1, KernelRows, 1, InnerTileRows, TIn, TOut, Roots>
{
  using Base = OutputTransform<1, KernelRows, 1, InnerTileRows, TIn, TOut, Roots>;

  public:
    OutputTransform(
      int n_batches,  /**< Number of batches in output tensor. */
      int n_rows,     /**< Number of rows in output tensor. */
      int n_cols,     /**< Number of columns in output tensor. */
      int n_channels  /**< Number of channels in output tensor. */
    );

    /** Set pointers to the output tensor written by the transform. */
    void set_output_tensor(void *output) override;
    void set_output_tensor(void *output, int col_stride) override;
    void set_output_tensor(void *output, int row_stride, int col_stride) override;
    void set_output_tensor(void *output, int batch_stride, int row_stride, int col_stride) override;
};

template <
  int KernelRows, int KernelCols,
  int InnerTileRows, int InnerTileCols,
  typename TIn, typename TOut,
  WinogradRoots Roots
>
class WeightTransform : public IWeightTransform
{
  public:
    WeightTransform(
      int n_output_channels,  /**< Number of output channels in the kernel. */
      int n_input_channels    /**< Number of input channels in the kernel. */
    );

    WeightTransform(WeightTransform&) = delete;
    WeightTransform operator=(WeightTransform&) = delete;

    /** Set pointer to the weight tensor read by the transform. */
    void set_weight_tensor(const void *weights) override;

    /** Set pointer to the matrices written by the transform. */
    void set_output_matrices(void *matrices, int inter_matrix_stride, int matrix_row_stride) override;

    /** Get the working space required to perform the transformation. */
    size_t get_working_space_size(unsigned int nthreads=1) const override;
    void set_working_space(void *buffer) override;

    /** Get the window of work a given operator can perform. */
    unsigned int get_window() const override;
    static constexpr unsigned int WINDOW_BLOCK = 16;  // Base size of window

    /** Perform work upon a window of the input. */
    void run(unsigned int start, unsigned int stop, unsigned int threadid=0) override;

  protected:
    static const int kernel_rows = KernelRows;
    static const int kernel_cols = KernelCols;
    static const int inner_tile_rows = InnerTileRows;
    static const int inner_tile_cols = InnerTileCols;

  private:
    /** Apply the transform to a tensor. */
    static void execute(
      int n_output_channels,
      int n_input_channels,
      const TIn* input,
      TOut* output,
      int matrix_stride,
      int matrix_row_stride
    );

    const int _n_output_channels, _n_input_channels;
    TOut *_matrices;
    int _matrix_stride, _matrix_row_stride;
    const TIn *_weights;
};

template <int KernelRows, int InnerTileRows, typename TIn, typename TOut, WinogradRoots Roots>
class WeightTransform<KernelRows, 1, InnerTileRows, 1, TIn, TOut, Roots> :
  public WeightTransform<1, KernelRows, 1, InnerTileRows, TIn, TOut, Roots>
{
  public:
    using WeightTransform<1, KernelRows, 1, InnerTileRows, TIn, TOut, Roots>::WeightTransform;
};

template <int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols, WinogradRoots Roots>
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
    template <typename TIn, typename TOut>
    using WeightsTransform = WeightTransform<
      KernelRows, KernelCols, inner_tile_rows, inner_tile_cols,
      TIn, TOut, Roots
    >;

    /** Transform input feature maps from the spatial to the Winograd domain.
     */
    template <typename TIn, typename TOut>
    using InputTransform = InputTransform<
      inner_tile_rows, inner_tile_cols, TIn, TOut, Roots
    >;

    /** Transform output feature maps from the Winograd to the spatial domain.
     */
    template <typename TIn, typename TOut>
    using OutputTransform = OutputTransform<
      KernelRows, KernelCols, inner_tile_rows, inner_tile_cols,
      TIn, TOut, Roots
    >;

    /** Perform a convolution.
     */
    template <typename TOut, typename TIn, typename TInGEMM=TIn, typename TOutGEMM=TOut>
    class Convolution
    {
      public:
        // Information about the typed Winograd instance
        typedef TOut OutputType;
        typedef TOutGEMM GemmOutputType;
        typedef TInGEMM GemmInputType;
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
