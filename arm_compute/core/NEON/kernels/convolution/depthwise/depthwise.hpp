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

namespace depthwise
{

class IDepthwiseConvolution
{
  public:
    virtual ~IDepthwiseConvolution() = default;
    virtual int output_size(const int dim_size, const bool padding_same) const = 0;
    virtual int output_size(
      int dim_size,
      unsigned int padding_before,
      unsigned int padding_after
    ) const = 0;

    virtual unsigned int get_window(void) const = 0;
    virtual void set_offsets(int input_offset, int weights_offset) = 0;
    virtual void run(const unsigned int start, const unsigned int stop) = 0;
};

template <
  int OutputTileRows,
  int OutputTileCols,
  int KernelRows,
  int KernelCols,
  int StrideRows,
  int StrideCols,
  typename TIn,
  typename TOut
>
class DepthwiseConvolution : public IDepthwiseConvolution
{
  public:
    typedef TIn InputType;
    typedef TOut OutputType;

    // Information about the specific convolution instance
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
     * @param[in]  n_batches Number of batches tensors.
     * @param[in]  n_input_rows Number of rows in input tensor.
     * @param[in]  n_input_cols Number of columns in input tensor.
     * @param[in]  n_channels Number of channels in input and output tensors.
     * @param[in]  padding_same True if padding is SAME, else VALID.
     * @param[in]  weights Pointer to Height x Width x Channel ordered weights.
     * @param[in]  input Pointer to NHWC ordered input tensor.
     * @param[out] output Pointer to NHWC ordered output tensor.
     */
    DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols,
      int n_channels, bool padding_same,
      const TIn* const weights,
      const TIn* const input,
      TOut* const output
    ) : DepthwiseConvolution(
      n_batches, n_input_rows, n_input_cols, n_channels, padding_same,
      weights, input, output, 0 /* column stride = default */
    )
    {
    }

    /** Create a new depthwise convolution engine.
     *
     * @param[in]  n_batches Number of batches tensors.
     * @param[in]  n_input_rows Number of rows in input tensor.
     * @param[in]  n_input_cols Number of columns in input tensor.
     * @param[in]  n_channels Number of channels in input and output tensors.
     * @param[in]  padding_top Padding to apply to top of input.
     * @param[in]  padding_left Padding to apply to left of input.
     * @param[in]  padding_bottom Padding to apply to bottom of input.
     * @param[in]  padding_right Padding to apply to right of input.
     * @param[in]  weights Pointer to Height x Width x Channel ordered weights.
     * @param[in]  input Pointer to NHWC ordered input tensor.
     * @param[out] output Pointer to NHWC ordered output tensor.
     */
    DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols,
      int n_channels,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right,
      const TIn* const weights,
      const TIn* const input,
      TOut* const output
    ) : DepthwiseConvolution(
      n_batches, n_input_rows, n_input_cols, n_channels,
      padding_top, padding_left, padding_bottom, padding_right,
      weights, input, output, 0 /* column stride = default */
    )
    {
    }

    /** Create a new depthwise convolution engine with a specified column stride.
     *
     * @param[in]  n_batches Number of batches tensors.
     * @param[in]  n_input_rows Number of rows in input tensor.
     * @param[in]  n_input_cols Number of columns in input tensor.
     * @param[in]  n_channels Number of channels in input and output tensors.
     * @param[in]  padding_same True if padding is SAME, else VALID.
     * @param[in]  weights Pointer to Height x Width x Channel ordered weights.
     * @param[in]  input Pointer to NHWC ordered input tensor.
     * @param[out] output Pointer to NHWC ordered output tensor.
     * @param[in]  col_stride Stride between columns of the weights, inputs and output tensors.
     */
    DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols,
      int n_channels, bool padding_same,
      const TIn* const weights,
      const TIn* const input,
      TOut* const output,
      const int col_stride
    ) : DepthwiseConvolution(
      n_batches, n_input_rows, n_input_cols, n_channels, padding_same,
      weights, input, output,
      col_stride, 0,    /* Weight row stride = default */
      col_stride, 0, 0, /* Input row stride, batch stride = default */
      col_stride, 0, 0  /* Output row stride, batch stride = default */
    )
    {
    }

    /** Create a new depthwise convolution engine with a specified column stride.
     *
     * @param[in]  n_batches Number of batches tensors.
     * @param[in]  n_input_rows Number of rows in input tensor.
     * @param[in]  n_input_cols Number of columns in input tensor.
     * @param[in]  n_channels Number of channels in input and output tensors.
     * @param[in]  padding_top Padding to apply to top of input.
     * @param[in]  padding_left Padding to apply to left of input.
     * @param[in]  padding_bottom Padding to apply to bottom of input.
     * @param[in]  padding_right Padding to apply to right of input.
     * @param[in]  weights Pointer to Height x Width x Channel ordered weights.
     * @param[in]  input Pointer to NHWC ordered input tensor.
     * @param[out] output Pointer to NHWC ordered output tensor.
     * @param[in]  col_stride Stride between columns of the weights, inputs and output tensors.
     */
    DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols,
      int n_channels,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right,
      const TIn* const weights,
      const TIn* const input,
      TOut* const output,
      const int col_stride
    ) : DepthwiseConvolution(
      n_batches, n_input_rows, n_input_cols, n_channels,
      padding_top, padding_left, padding_bottom, padding_right,
      weights, input, output,
      col_stride, 0,    /* Weight row stride = default */
      col_stride, 0, 0, /* Input row stride, batch stride = default */
      col_stride, 0, 0  /* Output row stride, batch stride = default */
    )
    {
    }

    /** Create a new depthwise convolution engine.
     *
     * @param[in]  n_batches Number of batches tensors.
     * @param[in]  n_input_rows Number of rows in input tensor.
     * @param[in]  n_input_cols Number of columns in input tensor.
     * @param[in]  n_channels Number of channels in input and output tensors.
     * @param[in]  padding_same True if padding is SAME, else VALID.
     * @param[in]  weights Pointer to Height x Width x Channel ordered weights.
     * @param[in]  input Pointer to NHWC ordered input tensor.
     * @param[out] output Pointer to NHWC ordered output tensor.
     * @param[in]  weight_col_stride Stride between columns of the weights (if 0, defaults appropriately).
     * @param[in]  weight_row_stride Stride between rows of the weights (if 0, defaults appropriately).
     * @param[in]  input_col_stride Stride between columns of the input tensor (if 0, defaults appropriately).
     * @param[in]  input_row_stride Stride between rows of the input tensor (if 0, defaults appropriately).
     * @param[in]  input_batch_stride Stride between batches of the input tensor (if 0, defaults appropriately).
     * @param[in]  output_col_stride Stride between columns of the output tensor (if 0, defaults appropriately).
     * @param[in]  output_row_stride Stride between rows of the output tensor (if 0, defaults appropriately).
     * @param[in]  output_batch_stride Stride between batches of the output tensor (if 0, defaults appropriately).
     */
    DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols,
      int n_channels, bool padding_same,
      const TIn* const weights,
      const TIn* const input,
      TOut* const output,
      int weight_col_stride,
      int weight_row_stride,
      int input_col_stride,
      int input_row_stride,
      int input_batch_stride,
      int output_col_stride,
      int output_row_stride,
      int output_batch_stride
    );

    /** Create a new depthwise convolution engine.
     *
     * @param[in]  n_batches Number of batches tensors.
     * @param[in]  n_input_rows Number of rows in input tensor.
     * @param[in]  n_input_cols Number of columns in input tensor.
     * @param[in]  n_channels Number of channels in input and output tensors.
     * @param[in]  padding_top Padding to apply to top of input.
     * @param[in]  padding_left Padding to apply to left of input.
     * @param[in]  padding_bottom Padding to apply to bottom of input.
     * @param[in]  padding_right Padding to apply to right of input.
     * @param[in]  weights Pointer to Height x Width x Channel ordered weights.
     * @param[in]  input Pointer to NHWC ordered input tensor.
     * @param[out] output Pointer to NHWC ordered output tensor.
     * @param[in]  weight_col_stride Stride between columns of the weights (if 0, defaults appropriately).
     * @param[in]  weight_row_stride Stride between rows of the weights (if 0, defaults appropriately).
     * @param[in]  input_col_stride Stride between columns of the input tensor (if 0, defaults appropriately).
     * @param[in]  input_row_stride Stride between rows of the input tensor (if 0, defaults appropriately).
     * @param[in]  input_batch_stride Stride between batches of the input tensor (if 0, defaults appropriately).
     * @param[in]  output_col_stride Stride between columns of the output tensor (if 0, defaults appropriately).
     * @param[in]  output_row_stride Stride between rows of the output tensor (if 0, defaults appropriately).
     * @param[in]  output_batch_stride Stride between batches of the output tensor (if 0, defaults appropriately).
     */
    DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols,
      int n_channels,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right,
      const TIn* const weights,
      const TIn* const input,
      TOut* const output,
      int weight_col_stride,
      int weight_row_stride,
      int input_col_stride,
      int input_row_stride,
      int input_batch_stride,
      int output_col_stride,
      int output_row_stride,
      int output_batch_stride
    );

    // Cannot copy or move a DepthwiseConvolution.
    DepthwiseConvolution(DepthwiseConvolution&) = delete;
    DepthwiseConvolution operator=(DepthwiseConvolution&) = delete;

    /** Get the number of output rows/columns.
     *
     * @param[in] dim_size Number of elements in the dimension (rows/columns)
     * @param[in] same_padding True if the padding is SAME, otherwise false.
     */
    static int get_output_size(int dim_size, bool padding_same);
    static int get_output_size(
      int dim_size,
      unsigned int padding_before,
      unsigned int padding_after
    );

    /** Get the number of output rows/columns.
     *
     * @param[in] dim_size Number of elements in the dimension (rows/columns)
     * @param[in] same_padding True if the padding is SAME, otherwise false.
     */
    int output_size(int dim_size, bool padding_same) const override
    {
      return DepthwiseConvolution<
        OutputTileRows,
        OutputTileCols,
        KernelRows,
        KernelCols,
        StrideRows,
        StrideCols,
        TIn, TOut
      >::get_output_size(dim_size, padding_same);
    }

    int output_size(
        int dim_size,
        unsigned int padding_before,
        unsigned int padding_after
    ) const override
    {
      return DepthwiseConvolution<
        OutputTileRows,
        OutputTileCols,
        KernelRows,
        KernelCols,
        StrideRows,
        StrideCols,
        TIn, TOut
      >::get_output_size(dim_size, padding_before, padding_after);
    }

    /** Sets quantization offsets
     *
     * @param[in] input_offset   Input offset
     * @param[in] weights_offset Weights offset
     */
     void set_offsets(int input_offset, int weights_offset) override;

    /** Get the window of work to be performed by an instance of the operator.
     */
    unsigned int get_window(void) const override;

    /** Perform a portion of the work associated with the operator.
     *
     * Will perform the window of work described by $[start, stop)$.
     *
     * @param[in] start Start of the window of work to perform.
     * @param[in] stop End of the work to perform.
     */
    void run(unsigned int start, unsigned int stop) override;

  protected:
    /** Process a tile-row of the tensors.
     */
    static void process_tile_row(
      int n_channels,
      const TIn* const weights,
      const int weight_row_stride,
      const int weight_col_stride,
      const TIn* const inptr,
      int in_row_stride,
      int in_col_stride,
      TOut* const outptr,
      int out_row_stride,
      int out_col_stride,
      int row_pad_in_top,
      int row_pad_in_left,
      int row_pad_in_bottom,
      int row_pad_out_bottom,
      int n_tiles,
      int n_input_cols,
      int n_output_cols,
      int input_offset,
      int weights_offset
    );

    // Determine the maximum (and minimum) padding values which can be applied
    // to tiles of the tensors involved in this class of convolution.
    static constexpr int max_in_pad_top = (kernel_rows - 1) / 2;
    static constexpr int min_in_pad_top = (kernel_rows - stride_rows) / 2;

    static constexpr int max_in_pad_left = (kernel_cols - 1) / 2;
    static constexpr int min_in_pad_left = (kernel_cols - stride_cols) / 2;

    static constexpr int max_in_pad_bottom = inner_tile_rows;
    static constexpr int max_in_pad_right = inner_tile_cols;
    static constexpr int max_out_pad_bottom = output_tile_rows;
    static constexpr int max_out_pad_right = output_tile_cols;

    static constexpr int n_in_pad_top_fns = (max_in_pad_top - min_in_pad_top) + 1;
    static constexpr int n_in_pad_left_fns = (max_in_pad_left - min_in_pad_left) + 1;
    static constexpr int n_in_pad_bottom_fns = max_in_pad_bottom + 1;
    static constexpr int n_in_pad_right_fns = max_in_pad_right + 1;
    static constexpr int n_out_pad_bottom_fns = max_out_pad_bottom + 1;
    static constexpr int n_out_pad_right_fns = max_out_pad_right + 1;

    /** Pointer to a function which will process a tile.
     *
     * @param[in] n_channels Number of channels.
     * @param[in] weights Pointer to Height x Width x Channels ordered weights.
     * @param[in] inptr Pointer to the top-left unpadded value of the tile.
     * @param[in] in_row_stride Stride between rows of the input tensor.
     * @param[in] in_col_stride Stride between columns of the input tensor.
     * @param[out] outptr Pointer to the top-left output value for the tile.
     * @param[in] out_row_stride Stride between rows of the output tensor.
     * @param[in] out_col_stride Stride between columns of the output tensor.
     *
     * The following parameters may be ignored if the function has been
     * specialised for specific padding constraints.
     *
     * @param[in] _in_pad_top Padding to apply to top of input tile.
     * @param[in] _in_pad_left Padding to apply to left of input tile.
     * @param[in] _in_pad_bottom Padding to apply to bottom of input tile.
     * @param[in] _in_pad_right Padding to apply to right of input tile.
     * @param[in] _out_pad_bottom Null cells at bottom of output tile.
     * @param[in] _out_pad_right Null cells at right of output tile.
     */
    typedef void (*TileFn)(
      int n_channels,
      const TIn* const weights,
      int weight_row_stride,
      int weight_col_stride,
      const TIn* const inptr,
      int in_row_stride,
      int in_col_stride,
      TOut* const outptr,
      int out_row_stride,
      int out_col_stride,
      int _in_pad_top,
      int _in_pad_left,
      int _in_pad_bottom,
      int _in_pad_right,
      int _out_pad_bottom,
      int _out_pad_right,
      int _input_offset,
      int _weights_offset
    );

    /* Arrays of methods to process tensor tiles.
     *
     * Allows dynamic dispatch to specialized implementations based on
     * different padding configurations.
     */
    static const TileFn tilefn_unpadded;
    static const TileFn tilefn_top[n_in_pad_top_fns];
    static const TileFn tilefn_left[n_in_pad_left_fns];
    static const TileFn tilefn_bottom[n_in_pad_bottom_fns][n_out_pad_bottom_fns];
    static const TileFn tilefn_right[n_in_pad_right_fns][n_out_pad_right_fns];
    static const TileFn tilefn_generic;

  private:
    // Member variables of instances of a convolution engine.
    const TIn* const _weights;
    const TIn* const _input;
    TOut* const _output;
    const int _n_batches, _n_input_rows, _n_input_cols, _n_channels,
              _n_output_rows, _n_output_cols, _n_tile_rows, _n_tile_cols;
    const unsigned int _padding_top, _padding_left, _padding_bottom, _padding_right;

    // Stride information for a convolution instance
    const int _weight_col_stride, _weight_row_stride;
    const int _input_col_stride, _input_row_stride, _input_batch_stride;
    const int _output_col_stride, _output_row_stride, _output_batch_stride;
    int _input_offset, _weights_offset;
};

}  // namespace depthwise
