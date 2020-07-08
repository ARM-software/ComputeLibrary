/*
 * Copyright (c) 2017 Arm Limited.
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
#ifndef DOXYGEN_SKIP_THIS
#include <cstdint>
#endif /* DOXYGEN_SKIP_THIS */
#include "arm.hpp"

namespace reorder {
/** Re-order a tensor from NCHW format to NHWC.
 *
 * @note The stride parameters are optional and are provided to allow padding in either input or output tensors.
 *
 * @param[in] in Input tensor in NCHW format.
 * @param[out] out Output tensor, to be written in NHWC format.
 * @param n_batches Number of batches in the tensors.
 * @param n_channels Number of channels in the tensors
 * @param n_rows Height of the tensor
 * @param n_cols Width of the tensor
 * @param in_batch_stride Stride over batches in the input tensor. If `0` defaults to `n_channels * in_channel_stride`.
 * @param in_channel_stride Stride over channels in the input tensor. If `0` defaults to `n_rows * in_row_stride`.
 * @param in_row_stride Stride over rows in the input tensor. If `0` defaults to `n_cols`.
 * @param out_batch_stride Stride over batches in the output tensor. If `0` defaults to `n_rows * out_row_stride`.
 * @param out_row_stride Stride over rows in the output tensor. If `0` defaults to `n_cols * out_col_stride`.
 * @param out_col_stride Stride over columns in the output tensor. If `0` defaults to `n_channels`.
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
);

/** Re-order a tensor from NHWC format to NCHW.
 *
 * @note The stride parameters are optional and are provided to allow padding in either input or output tensors.
 *
 * @param[in] in Input tensor in NHWC format.
 * @param[out] out Output tensor, to be written in NCHW format.
 * @param n_batches Number of batches in the tensors.
 * @param n_rows Height of the tensor
 * @param n_cols Width of the tensor
 * @param n_channels Number of channels in the tensors
 * @param in_batch_stride Stride over batches in the input tensor. If `0` defaults to `n_rows * in_row_stride`.
 * @param in_row_stride Stride over rows in the input tensor. If `0` defaults to `n_cols * in_col_stride`.
 * @param in_col_stride Stride over columns in the input tensor. If `0` defaults to `n_channels`.
 * @param out_batch_stride Stride over batches in the output tensor. If `0` defaults to `n_channels * out_channel_stride`.
 * @param out_channel_stride Stride over channels in the output tensor. If `0` defaults to `n_rows * out_row_stride`.
 * @param out_row_stride Stride over rows in the output tensor. If `0` defaults to `n_cols`.
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
);

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

/*****************************************************************************/
/* 32-bit implementation : NCHW -> NHWC
 */
template <>
inline void nchw_to_nhwc(
  const int32_t* const in,
  int32_t* const out,
  const int n_batches,
  const int n_channels,
  const int n_rows,
  const int n_cols,
  int in_batch_stride,
  int in_channel_stride,
  int in_row_stride,
  int out_batch_stride,
  int out_row_stride,
  int out_col_stride
)
{
  typedef int32_t T;

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

      int j = 0, j_remaining = n_cols;
#ifdef __arm_any__
      for (; j_remaining >= 4; j += 4, j_remaining -= 4)
      {
        int c = 0, c_remaining = n_channels;
        for (; c_remaining >= 4; c += 4, c_remaining -= 4)
        {
          // Read 4 channels worth of 4 columns, then zip to produce 4 columns
          // worth of 4 channels.
          int32x4_t channel_pixels[4];
          channel_pixels[0] = vld1q_s32(in_row + (c + 0)*in_channel_stride + j);
          channel_pixels[1] = vld1q_s32(in_row + (c + 1)*in_channel_stride + j);
          channel_pixels[2] = vld1q_s32(in_row + (c + 2)*in_channel_stride + j);
          channel_pixels[3] = vld1q_s32(in_row + (c + 3)*in_channel_stride + j);

          const auto zip1 = vzipq_s32(channel_pixels[0], channel_pixels[2]);
          const auto zip2 = vzipq_s32(channel_pixels[1], channel_pixels[3]);
          const auto out_0 = vzipq_s32(zip1.val[0], zip2.val[0]);
          const auto out_1 = vzipq_s32(zip1.val[1], zip2.val[1]);

          vst1q_s32(out_row + (j + 0)*out_col_stride + c, out_0.val[0]);
          vst1q_s32(out_row + (j + 1)*out_col_stride + c, out_0.val[1]);
          vst1q_s32(out_row + (j + 2)*out_col_stride + c, out_1.val[0]);
          vst1q_s32(out_row + (j + 3)*out_col_stride + c, out_1.val[1]);
        }
        for (; c_remaining; c++, c_remaining--)
        {
          for (int _j = 0; _j < 4; _j++)
          {
            const T* const in_col = in_row + j + _j;
            T* const out_col = out_row + (j + _j)*out_col_stride;
            const T* const in_channel = in_col + c*in_channel_stride;
            out_col[c] = *(in_channel);
          }
        }
      }
      for (; j_remaining >= 2; j += 2, j_remaining -= 2)
      {
        int c = 0, c_remaining = n_channels;
        for (; c_remaining >= 2; c += 2, c_remaining -= 2)
        {
          // Read 2 channels worth of 2 columns, then zip to produce 2 columns
          // worth of 2 channels.
          int32x2_t channel_pixels[2];
          channel_pixels[0] = vld1_s32(in_row + (c + 0)*in_channel_stride + j);
          channel_pixels[1] = vld1_s32(in_row + (c + 1)*in_channel_stride + j);

          const auto output = vzip_s32(channel_pixels[0], channel_pixels[1]);

          vst1_s32(out_row + (j + 0)*out_col_stride + c, output.val[0]);
          vst1_s32(out_row + (j + 1)*out_col_stride + c, output.val[1]);
        }
        for (; c_remaining; c++, c_remaining--)
        {
          for (int _j = 0; _j < 2; _j++)
          {
            const T* const in_col = in_row + j + _j;
            T* const out_col = out_row + (j + _j)*out_col_stride;
            const T* const in_channel = in_col + c*in_channel_stride;
            out_col[c] = *(in_channel);
          }
        }
      }
#endif  // __arm_any__
      for (; j_remaining; j++, j_remaining--)
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

template <>
inline void nchw_to_nhwc(
  const uint32_t* const in,
  uint32_t* const out,
  const int n_batches,
  const int n_channels,
  const int n_rows,
  const int n_cols,
  int in_batch_stride,
  int in_channel_stride,
  int in_row_stride,
  int out_batch_stride,
  int out_row_stride,
  int out_col_stride
)
{
  nchw_to_nhwc(
    reinterpret_cast<const int32_t*>(in),
    reinterpret_cast<int32_t*>(out),
    n_batches, n_channels, n_rows, n_cols,
    in_batch_stride, in_channel_stride, in_row_stride,
    out_batch_stride, out_row_stride, out_col_stride
  );
}

template <>
inline void nchw_to_nhwc(
  const float* const in,
  float* const out,
  const int n_batches,
  const int n_channels,
  const int n_rows,
  const int n_cols,
  int in_batch_stride,
  int in_channel_stride,
  int in_row_stride,
  int out_batch_stride,
  int out_row_stride,
  int out_col_stride
)
{
  nchw_to_nhwc(
    reinterpret_cast<const int32_t*>(in),
    reinterpret_cast<int32_t*>(out),
    n_batches, n_channels, n_rows, n_cols,
    in_batch_stride, in_channel_stride, in_row_stride,
    out_batch_stride, out_row_stride, out_col_stride
  );
}

/*****************************************************************************/
/* Generic implementation : NCHW -> NHWC
 */
template <typename T>
inline void nchw_to_nhwc(
  const T* const in,
  T* const out,
  const int n_batches,
  const int n_channels,
  const int n_rows,
  const int n_cols,
  int in_batch_stride,
  int in_channel_stride,
  int in_row_stride,
  int out_batch_stride,
  int out_row_stride,
  int out_col_stride
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

/*****************************************************************************/
/* 32-bit implementation : NHWC -> NCHW
 */
template <>
inline void nhwc_to_nchw(
  const int32_t* const in,  // Input data in NHWC form
  int32_t* const out,       // Output data in NCHW form
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels,
  int in_batch_stride,
  int in_row_stride,
  int in_col_stride,
  int out_batch_stride,
  int out_channel_stride,
  int out_row_stride
)
{
  typedef int32_t T;

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

      // For every column, beginning with chunks of 4
      int j = 0, j_remaining = n_cols;
#ifdef __arm_any__
      for (; j_remaining >= 4; j += 4, j_remaining -=4)
      {
        // For every channel, beginning with chunks of 4
        int c = 0, c_remaining = n_channels;
        for (; c_remaining >= 4; c += 4, c_remaining -= 4)
        {
          // Read 4 columns worth of 4 channels then zip to produce 4 channels
          // worth of 4 columns.
          int32x4_t pixel_channels[4];
          pixel_channels[0] = vld1q_s32(in_i + (j + 0)*in_col_stride + c);
          pixel_channels[1] = vld1q_s32(in_i + (j + 1)*in_col_stride + c);
          pixel_channels[2] = vld1q_s32(in_i + (j + 2)*in_col_stride + c);
          pixel_channels[3] = vld1q_s32(in_i + (j + 3)*in_col_stride + c);

          const auto zip1 = vzipq_s32(pixel_channels[0], pixel_channels[2]);
          const auto zip2 = vzipq_s32(pixel_channels[1], pixel_channels[3]);
          const auto out_0 = vzipq_s32(zip1.val[0], zip2.val[0]);
          const auto out_1 = vzipq_s32(zip1.val[1], zip2.val[1]);

          vst1q_s32(out_i + j + (c + 0)*out_channel_stride, out_0.val[0]);
          vst1q_s32(out_i + j + (c + 1)*out_channel_stride, out_0.val[1]);
          vst1q_s32(out_i + j + (c + 2)*out_channel_stride, out_1.val[0]);
          vst1q_s32(out_i + j + (c + 3)*out_channel_stride, out_1.val[1]);
        }
        for (; c_remaining; c++, c_remaining--)
        {
          for (int _j = 0; _j < 4; _j++)
          {
            const T* const in_j = in_i + (j + _j)*in_col_stride;
            T* const out_j = out_i + (j + _j);

            const T* const in_channel = in_j + c;
            T* const out_channel = out_j + c*out_channel_stride;
            *(out_channel) = *(in_channel);
          }
        }
      }
      for (; j_remaining >= 2; j += 2, j_remaining -=2)
      {
        int c = 0, c_remaining = n_channels;
        for (; c_remaining >= 2; c += 2, c_remaining -= 2)
        {
          // Read 2 columns worth of 2 channels then zip to produce 2 channels
          // worth of 2 columns.
          int32x2_t pixel_channels[2];
          pixel_channels[0] = vld1_s32(in_i + (j + 0)*in_col_stride + c);
          pixel_channels[1] = vld1_s32(in_i + (j + 1)*in_col_stride + c);

          const auto output = vzip_s32(pixel_channels[0], pixel_channels[1]);

          vst1_s32(out_i + j + (c + 0)*out_channel_stride, output.val[0]);
          vst1_s32(out_i + j + (c + 1)*out_channel_stride, output.val[1]);
        }
        for (; c_remaining; c++, c_remaining--)
        {
          for (int _j = 0; _j < 2; _j++)
          {
            const T* const in_j = in_i + (j + _j)*in_col_stride;
            T* const out_j = out_i + (j + _j);

            const T* const in_channel = in_j + c;
            T* const out_channel = out_j + c*out_channel_stride;
            *(out_channel) = *(in_channel);
          }
        }
      }
#endif  // __arm_any__
      for (; j_remaining; j++, j_remaining--)
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

template <>
inline void nhwc_to_nchw(
  const uint32_t* const in,  // Input data in NHWC form
  uint32_t* const out,       // Output data in NCHW form
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels,
  int in_batch_stride,
  int in_row_stride,
  int in_col_stride,
  int out_batch_stride,
  int out_channel_stride,
  int out_row_stride
)
{
  // Redirect to generic 32-bit implementation
  nhwc_to_nchw(
    reinterpret_cast<const int32_t*>(in),
    reinterpret_cast<int32_t*>(out),
    n_batches, n_rows, n_cols, n_channels,
    in_batch_stride, in_row_stride, in_col_stride,
    out_batch_stride, out_channel_stride, out_row_stride
  );
}

template <>
inline void nhwc_to_nchw(
  const float* const in,  // Input data in NHWC form
  float* const out,       // Output data in NCHW form
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels,
  int in_batch_stride,
  int in_row_stride,
  int in_col_stride,
  int out_batch_stride,
  int out_channel_stride,
  int out_row_stride
)
{
  // Redirect to generic 32-bit implementation
  nhwc_to_nchw(
    reinterpret_cast<const int32_t*>(in),
    reinterpret_cast<int32_t*>(out),
    n_batches, n_rows, n_cols, n_channels,
    in_batch_stride, in_row_stride, in_col_stride,
    out_batch_stride, out_channel_stride, out_row_stride
  );
}

/*****************************************************************************/
/* Generic implementation : NHWC -> NCHW
 */
template <typename T>
inline void nhwc_to_nchw(
  const T* const in,  // Input data in NHWC form
  T* const out,       // Output data in NCHW form
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels,
  int in_batch_stride,
  int in_row_stride,
  int in_col_stride,
  int out_batch_stride,
  int out_channel_stride,
  int out_row_stride
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

}  // namespace reorder
