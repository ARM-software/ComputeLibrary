/*
 * Copyright (c) 2018 ARM Limited.
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
    virtual unsigned int get_window(void) const = 0;
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
    static constexpr int inner_tile_rows = stride_rows * output_tile_rows + kernel_rows - 1;
    static constexpr int inner_tile_cols = stride_cols * output_tile_cols + kernel_cols - 1;

    /** Create a new depthwise convolution engine.
     *
     * @param[in] n_batches Number of batches tensors.
     * @param[in] n_input_rows Number of rows in input tensor.
     * @param[in] n_input_cols Number of columns in input tensor.
     * @param[in] n_channels Number of channels in input and output tensors.
     * @param[in] padding_same True if padding is SAME, else VALID.
     * @param[in] weights Pointer to Height x Width x Channel ordered weights.
     * @param[in] input Pointer to NHWC ordered input tensor.
     * @param[output] output Pointer to NHWC ordered output tensor.
     */
    DepthwiseConvolution(
      const int n_batches, const int n_input_rows, const int n_input_cols,
      const int n_channels, const bool padding_same,
      const TIn* const weights,
      const TIn* const input,
      TOut* const output
    );

    // Cannot copy or move a DepthwiseConvolution.
    DepthwiseConvolution(DepthwiseConvolution&) = delete;
    DepthwiseConvolution operator=(DepthwiseConvolution&) = delete;

    /** Get the number of output rows/columns.
     *
     * @param[in] dim_size Number of elements in the dimension (rows/columns)
     * @param[in] same_padding True if the padding is SAME, otherwise false.
     */
    static int get_output_size(const int dim_size, const bool padding_same);

    /** Get the number of output rows/columns.
     *
     * @param[in] dim_size Number of elements in the dimension (rows/columns)
     * @param[in] same_padding True if the padding is SAME, otherwise false.
     */
    int output_size(const int dim_size, const bool padding_same) const override
    {
        return DepthwiseConvolution<OutputTileRows,
                                    OutputTileCols,
                                    KernelRows,
                                    KernelCols,
                                    StrideRows,
                                    StrideCols,
                                    TIn,
                                    TOut>::get_output_size(dim_size, padding_same);
    }

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
    void run(const unsigned int start, const unsigned int stop) override;

  protected:
    /** Process a tile-row of the tensors.
     */
    static void process_tile_row(
      const int n_channels,
      const TIn* const weights,
      const TIn* const inptr,
      const int in_row_stride,
      const int in_col_stride,
      TOut* const outptr,
      const int out_row_stride,
      const int out_col_stride,
      const int row_pad_in_top,
      const int row_pad_in_left,
      const int row_pad_in_bottom,
      const int row_pad_out_bottom,
      const int n_tiles,
      const int n_input_cols,
      const int n_output_cols
    );

    /** Process a single tile of the tensors.
     *
     * @param[in] n_channels Number of channels.
     * @param[in] weights Pointer to Height x Width x Channels ordered weights.
     * @param[in] inptr Pointer to the top-left unpadded value of the tile.
     * @param[in] in_row_stride Stride between rows of the input tensor.
     * @param[in] in_col_stride Stride between columns of the input tensor.
     * @param[out] outptr Pointer to the top-left output value for the tile.
     * @param[in] out_row_stride Stride between rows of the output tensor.
     * @param[in] out_col_stride Stride between columns of the output tensor.
     */
    template <
      int in_pad_top, int in_pad_left, int in_pad_bottom, int in_pad_right,
      int out_pad_bottom, int out_pad_right
    >
    static void process_tile(
      const int n_channels,
      const TIn* const weights,
      const TIn* const inptr,
      const int in_row_stride,
      const int in_col_stride,
      TOut* const outptr,
      const int out_row_stride,
      const int out_col_stride
    );

    // Type of a pointer to a `process_tile` instance
    typedef void (*TileFn)(
      const int,
      const TIn* const,
      const TIn* const, const int, const int,
      TOut* const, const int, const int
    );

    // Determine the maximum padding values which can be applied to tiles of
    // the tensors involved in this class of convolution.
    static constexpr int max_in_pad_top = 2;
    static constexpr int max_in_pad_left = 2;
    static constexpr int max_in_pad_bottom = inner_tile_rows - 1;
    static constexpr int max_in_pad_right = inner_tile_cols - 1;
    static constexpr int max_out_pad_bottom = output_tile_rows;
    static constexpr int max_out_pad_right = output_tile_cols;

    /** Array of methods to process tensor tiles.
     *
     * Allows dynamic dispatch to specialized implementations based on
     * different padding configurations.
     */
    static const TileFn tile_fns[
      max_in_pad_top][max_in_pad_left][max_in_pad_bottom][max_in_pad_right][
      max_out_pad_bottom][max_out_pad_right
    ];

  private:
    // Member variables of instances of a convolution engine.
    const TIn* const _weights;
    const TIn* const _input;
    TOut* const _output;
    const int _n_batches, _n_input_rows, _n_input_cols, _n_channels,
              _n_output_rows, _n_output_cols, _n_tile_rows, _n_tile_cols;
    const bool _padding_same;
};

}  // namespace depthwise
