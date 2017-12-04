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

/** Re-order a weight tensor from [Output feature map x Input feature map x
 *  Height x Width] format to [Height x Width x Input feature map x Output
 *  feature map] format.
 */
template <typename T>
inline void ofm_ifm_h_w_to_h_w_ifm_ofm(
  const T* const in,  // Input in [Output x Input x Height x Width] form
  T* const out,       // Output in [Height x Width x Input x Output] form
  const int n_output_feature_maps,
  const int n_input_feature_maps,
  const int n_rows,
  const int n_cols,
  int in_output_feature_map_stride=0,
  int in_input_feature_map_stride=0,
  int in_row_stride=0,
  int out_row_stride=0,
  int out_col_stride=0,
  int out_input_feature_map_stride=0
);

/** Re-order a weight tensor from [Height x Width x Input feature map x Output
 *  feature map] format to [Output feature map x Input feature map x Height x
 *  Width] format.
 */
template <typename T>
inline void h_w_ifm_ofm_to_ofm_ifm_h_w(
  const T* const in,  // Input in [Height x Width x Input x Output] form
  T* const out,       // Output in [Output x Input x Height x Width] form
  const int n_rows,
  const int n_cols,
  const int n_input_feature_maps,
  const int n_output_feature_maps,
  int in_row_stride=0,
  int in_col_stride=0,
  int in_input_feature_map_stride=0,
  int out_output_feature_map_stride=0,
  int out_input_feature_map_stride=0,
  int out_row_stride=0
);


/* Re-order a tensor from NCHW format to NHWC.
 */
template <typename T>
inline void nchw_to_nhwc(
  const T* const in,
  T* const out,
  const int n_batches,
  const int n_channels,
  const int n_rows,
  const int n_cols,
  int in_batch_stride=0,
  int in_channel_stride=0,
  int in_row_stride=0,
  int out_batch_stride=0,
  int out_row_stride=0,
  int out_col_stride=0
)
{
  // Fill in the stride values
  in_row_stride = (in_row_stride) ? in_row_stride : n_cols;
  in_channel_stride = (in_channel_stride) ? in_channel_stride
                                          : n_rows * in_row_stride;
  in_batch_stride = (in_batch_stride) ? in_batch_stride
                                      : n_channels * in_channel_stride;

  out_col_stride = (out_col_stride) ? out_col_stride : n_channels;
  out_row_stride = (out_row_stride) ? out_row_stride : n_cols * out_col_stride;
  out_batch_stride = (out_batch_stride) ? out_batch_stride
                                        : n_rows * out_row_stride;

  // Perform the re-ordering
  for (int n = 0; n < n_batches; n++)
  {
    const T* const in_batch = in + n*in_batch_stride;
    T* const out_batch = out + n*out_batch_stride;

    for (int i = 0; i < n_rows; i++)
    {
      const T* const in_row = in_batch + i*in_row_stride;
      T* const out_row = out_batch + i*out_row_stride;

      for (int j = 0; j < n_cols; j++)
      {
        const T* const in_col = in_row + j;
        T* const out_col = out_row + j*out_col_stride;

        for (int c = 0; c < n_channels; c++)
        {
          const T* const in_channel = in_col + c*in_channel_stride;
          out_col[c] = *(in_channel);
        }
      }
    }
  }
}

/* Re-order a tensor from NHWC format to NCHW.
 */
template <typename T>
inline void nhwc_to_nchw(
  const T* const in,  // Input data in NHWC form
  T* const out,       // Output data in NCHW form
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels,
  int in_batch_stride=0,
  int in_row_stride=0,
  int in_col_stride=0,
  int out_batch_stride=0,
  int out_channel_stride=0,
  int out_row_stride=0
)
{
  // Fill in stride values
  in_col_stride = (in_col_stride) ? in_col_stride : n_channels;
  in_row_stride = (in_row_stride) ? in_row_stride : n_cols * in_col_stride;
  in_batch_stride = (in_batch_stride) ? in_batch_stride
                                      : n_rows * in_row_stride;

  out_row_stride = (out_row_stride) ? out_row_stride : n_cols;
  out_channel_stride = (out_channel_stride) ? out_channel_stride
                                            : n_rows * out_row_stride;
  out_batch_stride = (out_batch_stride) ? out_batch_stride
                                        : n_channels * out_channel_stride;

  // Perform the re-ordering
  // For every batch
  for (int n = 0; n < n_batches; n++)
  {
    const T* const in_batch = in + n*in_batch_stride;
    T* const out_batch = out + n*out_batch_stride;

    // For every row
    for (int i = 0; i < n_rows; i++)
    {
      const T* const in_i = in_batch + i*in_row_stride;
      T* const out_i = out_batch + i*out_row_stride;

      // For every column
      for (int j = 0; j < n_cols; j++)
      {
        const T* const in_j = in_i + j*in_col_stride;
        T* const out_j = out_i + j;

        // For every channel
        for (int c = 0; c < n_channels; c++)
        {
          const T* const in_channel = in_j + c;
          T* const out_channel = out_j + c*out_channel_stride;
          *(out_channel) = *(in_channel);
        }
      }
    }
  }
}


/*****************************************************************************/
/* Generic weight re-order implementation.
 */
template <typename T>
inline void ofm_ifm_h_w_to_h_w_ifm_ofm(
  const T* const in,  // Input in [Output x Input x Height x Width] form
  T* const out,       // Output in [Height x Width x Input x Output] form
  const int n_output_feature_maps,
  const int n_input_feature_maps,
  const int n_rows,
  const int n_cols,
  int in_output_feature_map_stride,
  int in_input_feature_map_stride,
  int in_row_stride,
  int out_row_stride,
  int out_col_stride,
  int out_input_feature_map_stride
)
{
  // Fill in stride values
  in_row_stride = (in_row_stride)
    ? in_row_stride
    : n_cols;
  in_input_feature_map_stride = (in_input_feature_map_stride)
    ? in_input_feature_map_stride
    : n_rows * in_row_stride;
  in_output_feature_map_stride = (in_output_feature_map_stride)
    ? in_output_feature_map_stride
    : n_input_feature_maps * in_input_feature_map_stride;

  out_input_feature_map_stride = (out_input_feature_map_stride)
    ? out_input_feature_map_stride
    : n_output_feature_maps;
  out_col_stride = (out_col_stride)
    ? out_col_stride
    : n_input_feature_maps * out_input_feature_map_stride;
  out_row_stride = (out_row_stride)
    ? out_row_stride
    : n_cols * out_col_stride;

  // Perform the re-ordering
  for (int i = 0; i < n_rows; i++)
  {
    const T* const in_row = in + i * in_row_stride;
    T* out_row = out + i * out_row_stride;

    for (int j = 0; j < n_cols; j++)
    {
      const T* const in_col = in_row + j;
      T* const out_col = out_row + j * out_col_stride;

      for (int ifm = 0; ifm < n_input_feature_maps; ifm++)
      {
        const T* const in_ifm = in_col + ifm * in_input_feature_map_stride;
        T* const out_ifm = out_col + ifm * out_input_feature_map_stride;

        for (int ofm = 0; ofm < n_output_feature_maps; ofm++)
        {
          const T* const in_ofm = in_ifm + ofm * in_output_feature_map_stride;
          T* const out_ofm = out_ifm + ofm;
          *(out_ofm) = *(in_ofm);
        }
      }
    }
  }
}

/*****************************************************************************/
/* Generic weight re-order implementation.
 */
template <typename T>
inline void h_w_ifm_ofm_to_ofm_ifm_h_w(
  const T* const in,  // Input in [Height x Width x Input x Output] form
  T* const out,       // Output in [Output x Input x Height x Width] form
  const int n_rows,
  const int n_cols,
  const int n_input_feature_maps,
  const int n_output_feature_maps,
  int in_row_stride,
  int in_col_stride,
  int in_input_feature_map_stride,
  int out_output_feature_map_stride,
  int out_input_feature_map_stride,
  int out_row_stride
)
{
  // Fill in the stride values
  in_input_feature_map_stride = (in_input_feature_map_stride)
    ? in_input_feature_map_stride
    : n_output_feature_maps;
  in_col_stride = (in_col_stride)
    ? in_col_stride
    : n_input_feature_maps * in_input_feature_map_stride;
  in_row_stride = (in_row_stride)
    ? in_row_stride
    : n_cols * in_col_stride;

  out_row_stride = (out_row_stride)
    ? out_row_stride
    : n_cols;
  out_input_feature_map_stride = (out_input_feature_map_stride)
    ? out_input_feature_map_stride
    : n_rows * out_row_stride;
  out_output_feature_map_stride = (out_output_feature_map_stride)
    ? out_output_feature_map_stride
    : n_input_feature_maps * out_input_feature_map_stride;

  // Perform the re-ordering
  for (int i = 0; i < n_rows; i++)
  {
    const T* const in_row = in + i * in_row_stride;
    T* const out_row = out + i * out_row_stride;

    for (int j = 0; j < n_cols; j++)
    {
      const T* const in_col = in_row + j * in_col_stride;
      T* const out_col = out_row + j;

      for (int ifm = 0; ifm < n_input_feature_maps; ifm++)
      {
        const T* const in_ifm = in_col + ifm * in_input_feature_map_stride;
        T* const out_ifm = out_col + ifm * out_input_feature_map_stride;

        for (int ofm = 0; ofm < n_output_feature_maps; ofm++)
        {
          const T* const in_ofm = in_ifm + ofm;
          T* const out_ofm = out_ifm + ofm * out_output_feature_map_stride;
          *(out_ofm) = *(in_ofm);
        }
      }
    }
  }
}

