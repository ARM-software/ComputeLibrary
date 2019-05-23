/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#include <arm_neon.h>
#include "arm_compute/core/NEON/kernels/convolution/common/activation.hpp"
#include "arm_compute/core/NEON/kernels/convolution/common/padding.hpp"

namespace depthwise
{

namespace nck = neon_convolution_kernels;

class IDepthwiseConvolution
{
  public:
    virtual ~IDepthwiseConvolution() = default;

    virtual int output_size(
      int dim_size,
      unsigned int padding_before,
      unsigned int padding_after
    ) const = 0;

    /* Set input tensor and stride. */
    virtual void set_input(const void *inptr) = 0;
    virtual void set_input(const void *inptr, int column_stride) = 0;
    virtual void set_input(const void *inptr, int row_stride, int column_stride) = 0;
    virtual void set_input(const void *inptr, int batch_stride, int row_stride, int column_stride) = 0;

    /* Set output tensor and stride. */
    virtual void set_output(void *outptr) = 0;
    virtual void set_output(void *outptr, int column_stride) = 0;
    virtual void set_output(void *outptr, int row_stride, int column_stride) = 0;
    virtual void set_output(void *outptr, int batch_stride, int row_stride, int column_stride) = 0;

    /* Weights and biases are re-ordered to improve memory access patterns. Use
     * these methods to determine the size of the re-pack buffer and to set the
     * address (and implicitly reorder the weights and biases into) the buffer.
     */
    virtual size_t get_packed_params_size(void) const = 0;
    virtual void set_packed_params_buffer(void *) = 0;

    virtual void pack_params(const void *weights, const void *biases=nullptr) const = 0;
    virtual void pack_params(void *buffer, const void *weights, const void *biases=nullptr) const = 0;
    virtual void pack_params(
      void *buffer,
      const void* weights,
      unsigned int weight_row_stride,
      unsigned int weight_col_stride,
      const void *biases=nullptr
    ) const = 0;

    /* Working space is used to pad tensors on the fly. Before running any
     * inference check the amount of space required, allocate and provide a
     * pointer to the convolution engine.
     */
    virtual size_t get_working_space_size(unsigned int nthreads=1) const = 0;
    virtual void set_working_space(void *) = 0;

    virtual unsigned int get_window(void) const = 0;
    virtual void run(
      unsigned int start,
      unsigned int stop,
      unsigned int threadid=0
    ) = 0;
};

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols,
  typename TIn, typename TBias, typename TOut,
  typename Derived
>
class DepthwiseConvolutionBase : public IDepthwiseConvolution
{
  public:
    // Information about the specific convolution instance
    using InputType = TIn;
    using BiasType = TBias;
    using OutputType = TOut;
    static constexpr int output_tile_rows = OutputTileRows;
    static constexpr int output_tile_cols = OutputTileCols;
    static constexpr int kernel_rows = KernelRows;
    static constexpr int kernel_cols = KernelCols;
    static constexpr int stride_rows = StrideRows;
    static constexpr int stride_cols = StrideCols;
    static constexpr int inner_tile_rows = stride_rows * (output_tile_rows - 1) + kernel_rows;
    static constexpr int inner_tile_cols = stride_cols * (output_tile_cols - 1) + kernel_cols;

    /** Create a new depthwise convolution engine.
     *
     * @param[in] n_batches Number of batches tensors.
     * @param[in] n_input_rows Number of rows in input tensor.
     * @param[in] n_input_cols Number of columns in input tensor.
     * @param[in] n_channels Number of channels in input and output tensors.
     */
    DepthwiseConvolutionBase(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      nck::ActivationFunction activation,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right
    );

    // Cannot copy or move a DepthwiseConvolution.
    DepthwiseConvolutionBase(DepthwiseConvolutionBase&) = delete;
    DepthwiseConvolutionBase operator=(DepthwiseConvolutionBase&) = delete;

    /* Set input tensor and stride. */
    void set_input(const void *inptr) override;
    void set_input(const void *inptr, int column_stride) override;
    void set_input(const void *inptr, int row_stride, int column_stride) override;
    void set_input(const void *inptr, int batch_stride, int row_stride, int column_stride) override;

    /* Set output tensor and stride. */
    void set_output(void *outptr) override;
    void set_output(void *outptr, int column_stride) override;
    void set_output(void *outptr, int row_stride, int column_stride) override;
    void set_output(void *outptr, int batch_stride, int row_stride, int column_stride) override;

    /** Get the number of output rows/columns.
     *
     * @param[in] dim_size Number of elements in the dimension (rows/columns)
     * @param[in] same_padding True if the padding is SAME, otherwise false.
     */
    static int get_output_size(
      int dim_size, unsigned int padding_before, unsigned int padding_after
    );

    int output_size(
      int dim_size, unsigned int padding_before, unsigned int padding_after
    ) const override;

    /* Determine how much memory is required to store the packed weights and
     * biases.
     */
    size_t get_packed_params_size(void) const override;

    /* Set the buffer for the packed weights and biases, and perform the
     * packing.
     */
    void set_packed_params_buffer(void *buffer) override;

    void pack_params(const void *weights, const void *biases=nullptr) const override;

    void pack_params(
      void *buffer,
      const void *weights,
      const void *biases=nullptr
    ) const override;

    void pack_params(
      void *buffer,
      const void *weights,
      unsigned int weight_row_stride,
      unsigned int weight_col_stride,
      const void *biases=nullptr
    ) const override;

    /** Query the amount of working space required.
     * @param[in] The largest number of threads which will be used to execute
     *            the kernel.
     */
    size_t get_working_space_size(unsigned int n_threads=1) const override;

    /** Set the working space buffer.
     */
    void set_working_space(void *buffer) override;

    /** Get the window of work to be performed by an instance of the operator.
     */
    unsigned int get_window(void) const override;

    /** Perform a portion of the work associated with the operator.
     *
     * Will perform the window of work described by $[start, stop)$.
     *
     * @param[in] start Start of the window of work to perform.
     * @param[in] stop End of the work to perform.
     * @param[in] ID of the thread performing the work.
     */
    void run(
      unsigned int start,
      unsigned int stop,
      unsigned int threadid=0
    ) override;

  protected:
    /** Get the value to use to pad the tensor.
     */
    TIn _input_padding_value(void) const;

    /** Implementation of the parameter packing.
     */
    void _pack_params(
      void *buffer,
      const void *weights,
      unsigned int weight_row_stride,
      unsigned int weight_col_stride,
      const void *biases=nullptr
    ) const;

    /** Process a tile-row of the tensors.
     */
    void process_tile_row(
      unsigned int threadid,
      int n_channels,
      const void* packed_params,
      const InputType* inptr,
      OutputType* outptr,
      int row_pad_in_top,
      int row_pad_in_left,
      int row_pad_in_bottom,
      int row_pad_out_bottom,
      int n_tiles,
      int n_input_cols,
      int n_output_cols
    );

    /** Process a single tile of the tensor.
     *
     * This method will apply input/output padding (if required) and call the
     * depthwise tile implementation.
     */
    void process_tile(
      unsigned int threadid,
      int n_channels,
      const void* packed_params,
      const InputType* inptr,
      OutputType* outptr,
      int pad_in_top,
      int pad_in_left,
      int pad_in_bottom,
      int pad_in_right,
      int pad_out_bottom,
      int pad_out_right
    );

    /** Perform depthwise convolution on a single tile.
     */
    template <nck::ActivationFunction Activation>
    void execute_tile(
      int n_channels,
      const void* packed_params,
      const InputType* inptr,
      unsigned int in_row_stride,
      unsigned int in_col_stride,
      OutputType* outptr,
      unsigned int out_row_stride,
      unsigned int out_col_stride
    );

    template <nck::ActivationFunction Activation>
    void execute_tile(
      int n_channels,
      const void* packed_params,
      const InputType* inptrs[inner_tile_rows][inner_tile_cols],
      OutputType* outptrs[output_tile_rows][output_tile_cols]
    );

    int n_channels(void) const;

  private:
    // Member variables of instances of a convolution engine.
    const InputType* _input;
    OutputType* _output;
    void* _packed_parameters;
    void* _working_space;  // Per-thread working space
    const int _n_batches, _n_input_rows, _n_input_cols, _n_channels,
              _n_output_rows, _n_output_cols, _n_tile_rows, _n_tile_cols;
    const unsigned int _padding_top, _padding_left, _padding_bottom, _padding_right;
    const nck::ActivationFunction _activation;

    // Stride information for a convolution instance
    int _input_col_stride, _input_row_stride, _input_batch_stride;
    int _output_col_stride, _output_row_stride, _output_batch_stride;

    // Methods for getting access to working space
    size_t _get_input_working_space_size(void) const;
    size_t _get_output_working_space_size(void) const;

    void *_get_input_working_space(unsigned int threadid) const;
    void *_get_output_working_space(unsigned int threadid) const;
};


template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols,
  typename TIn, typename TBias, typename TOut
>
class DepthwiseConvolution : public DepthwiseConvolutionBase<
  OutputTileRows, OutputTileCols,
  KernelRows, KernelCols,
  StrideRows, StrideCols,
  TIn, TBias, TOut,
  DepthwiseConvolution<
    OutputTileRows, OutputTileCols,
    KernelRows, KernelCols,
    StrideRows, StrideCols,
    TIn, TBias, TOut
  >
>
{
  using Base = DepthwiseConvolutionBase<
    OutputTileRows, OutputTileCols,
    KernelRows, KernelCols,
    StrideRows, StrideCols,
    TIn, TBias, TOut,
    DepthwiseConvolution<
      OutputTileRows, OutputTileCols,
      KernelRows, KernelCols,
      StrideRows, StrideCols,
      TIn, TBias, TOut
  > >;
  friend Base;
  using InputType = typename Base::InputType;
  using OutputType = typename Base::OutputType;

  public:
    using Base::DepthwiseConvolutionBase;

  protected:
    template <nck::ActivationFunction Activation>
    void execute_tile(
      int n_channels,
      const void* packed_params,
      const TIn* inptr,
      unsigned int in_row_stride,
      unsigned int in_col_stride,
      TOut* outptr,
      unsigned int out_row_stride,
      unsigned int out_col_stride
    );

    template <nck::ActivationFunction Activation>
    void execute_tile(
      int n_channels,
      const void* packed_params,
      const InputType* inptrs[Base::inner_tile_rows][Base::inner_tile_cols],
      OutputType* outptrs[Base::output_tile_rows][Base::output_tile_cols]
    );
};


template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
class DepthwiseConvolution<
  OutputTileRows, OutputTileCols,
  KernelRows, KernelCols,
  StrideRows, StrideCols,
  float, float, float
> : public DepthwiseConvolutionBase<
  OutputTileRows, OutputTileCols,
  KernelRows, KernelCols,
  StrideRows, StrideCols,
  float, float, float,
  DepthwiseConvolution<
    OutputTileRows, OutputTileCols,
    KernelRows, KernelCols,
    StrideRows, StrideCols,
    float, float, float
  >
>
{
  using Base = DepthwiseConvolutionBase<
    OutputTileRows, OutputTileCols,
    KernelRows, KernelCols,
    StrideRows, StrideCols,
    float, float, float,
    DepthwiseConvolution<
      OutputTileRows, OutputTileCols,
      KernelRows, KernelCols,
      StrideRows, StrideCols,
      float, float, float
  > >;
  friend Base;
  using InputType = typename Base::InputType;
  using OutputType = typename Base::OutputType;

  public:
    DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      nck::ActivationFunction activation,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right
    );

  protected:
    template <nck::ActivationFunction Activation>
    void execute_tile(
      int n_channels,
      const void* packed_params,
      const float* inptr,
      unsigned int in_row_stride,
      unsigned int in_col_stride,
      float* outptr,
      unsigned int out_row_stride,
      unsigned int out_col_stride
    );

    template <nck::ActivationFunction Activation>
    void execute_tile(
      int n_channels,
      const void* packed_params,
      const float* inptrs[Base::inner_tile_rows][Base::inner_tile_cols],
      float* outptrs[Base::output_tile_rows][Base::output_tile_cols]
    );
};

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
class DepthwiseConvolution<
  OutputTileRows, OutputTileCols,
  KernelRows, KernelCols,
  StrideRows, StrideCols,
  float16_t, float16_t, float16_t
> : public DepthwiseConvolutionBase<
  OutputTileRows, OutputTileCols,
  KernelRows, KernelCols,
  StrideRows, StrideCols,
  float16_t, float16_t, float16_t,
  DepthwiseConvolution<
    OutputTileRows, OutputTileCols,
    KernelRows, KernelCols,
    StrideRows, StrideCols,
    float16_t, float16_t, float16_t
  >
>
{
  using Base = DepthwiseConvolutionBase<
    OutputTileRows, OutputTileCols,
    KernelRows, KernelCols,
    StrideRows, StrideCols,
    float16_t, float16_t, float16_t,
    DepthwiseConvolution<
      OutputTileRows, OutputTileCols,
      KernelRows, KernelCols,
      StrideRows, StrideCols,
      float16_t, float16_t, float16_t
  > >;
  friend Base;
  using InputType = typename Base::InputType;
  using OutputType = typename Base::OutputType;

  public:
    DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      nck::ActivationFunction activation,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right
    );

  protected:
    template <nck::ActivationFunction Activation>
    void execute_tile(
      int n_channels,
      const void* packed_params,
      const float16_t* inptr,
      unsigned int in_row_stride,
      unsigned int in_col_stride,
      float16_t* outptr,
      unsigned int out_row_stride,
      unsigned int out_col_stride
    );

    template <nck::ActivationFunction Activation>
    void execute_tile(
      int n_channels,
      const void* packed_params,
      const float16_t* inptrs[Base::inner_tile_rows][Base::inner_tile_cols],
      float16_t* outptrs[Base::output_tile_rows][Base::output_tile_cols]
    );
};
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

}  // namespace depthwise
