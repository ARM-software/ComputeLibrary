/*
 * Copyright (c) 2017, 2024 Arm Limited.
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
/* 16-bit implementation : NCHW -> NHWC
 */
template <>
inline void nchw_to_nhwc(const int16_t *const in,
                         int16_t *const       out,
                         const int            n_batches,
                         const int            n_channels,
                         const int            n_rows,
                         const int            n_cols,
                         int                  in_batch_stride,
                         int                  in_channel_stride,
                         int                  in_row_stride,
                         int                  out_batch_stride,
                         int                  out_row_stride,
                         int                  out_col_stride)
{
    typedef int16_t T;

    // Fill in the stride values
    in_row_stride     = (in_row_stride) ? in_row_stride : n_cols;
    in_channel_stride = (in_channel_stride) ? in_channel_stride : n_rows * in_row_stride;
    in_batch_stride   = (in_batch_stride) ? in_batch_stride : n_channels * in_channel_stride;

    out_col_stride   = (out_col_stride) ? out_col_stride : n_channels;
    out_row_stride   = (out_row_stride) ? out_row_stride : n_cols * out_col_stride;
    out_batch_stride = (out_batch_stride) ? out_batch_stride : n_rows * out_row_stride;

    // Perform the re-ordering
    for (int n = 0; n < n_batches; n++)
    {
        const T *const in_batch  = in + n * in_batch_stride;
        T *const       out_batch = out + n * out_batch_stride;

        for (int i = 0; i < n_rows; i++)
        {
            const T *const in_row  = in_batch + i * in_row_stride;
            T *const       out_row = out_batch + i * out_row_stride;

            int j = 0, j_remaining = n_cols;
#ifdef __arm_any__
            for (; j_remaining >= 8; j += 8, j_remaining -= 8)
            {
                int c = 0, c_remaining = n_channels;
                for (; c_remaining >= 8; c += 8, c_remaining -= 8)
                {
                    // Read 8 channels worth of 8 columns, then zip to produce 8 columns
                    // worth of 8 channels.
                    int16x8_t channel_pixels[8];
                    channel_pixels[0] = vld1q_s16(in_row + (c + 0) * in_channel_stride + j);
                    channel_pixels[1] = vld1q_s16(in_row + (c + 1) * in_channel_stride + j);
                    channel_pixels[2] = vld1q_s16(in_row + (c + 2) * in_channel_stride + j);
                    channel_pixels[3] = vld1q_s16(in_row + (c + 3) * in_channel_stride + j);
                    channel_pixels[4] = vld1q_s16(in_row + (c + 4) * in_channel_stride + j);
                    channel_pixels[5] = vld1q_s16(in_row + (c + 5) * in_channel_stride + j);
                    channel_pixels[6] = vld1q_s16(in_row + (c + 6) * in_channel_stride + j);
                    channel_pixels[7] = vld1q_s16(in_row + (c + 7) * in_channel_stride + j);

                    // 0th and 4th, 1st and 5th, 2nd and 6th, 3rd and 7th channels
                    const int16x8x2_t zip1 = vzipq_s16(channel_pixels[0], channel_pixels[4]);
                    const int16x8x2_t zip2 = vzipq_s16(channel_pixels[1], channel_pixels[5]);
                    const int16x8x2_t zip3 = vzipq_s16(channel_pixels[2], channel_pixels[6]);
                    const int16x8x2_t zip4 = vzipq_s16(channel_pixels[3], channel_pixels[7]);

                    // 0th, 2nd, 4th, 6th channels
                    const int16x8x2_t zip5 = vzipq_s16(zip1.val[0], zip3.val[0]);
                    const int16x8x2_t zip6 = vzipq_s16(zip1.val[1], zip3.val[1]);

                    // 1st, 3rd, 5th, 7th channels
                    const int16x8x2_t zip7 = vzipq_s16(zip2.val[0], zip4.val[0]);
                    const int16x8x2_t zip8 = vzipq_s16(zip2.val[1], zip4.val[1]);

                    // 0th, 1st, 2nd, ..., 7th channels
                    const int16x8x2_t out_0 = vzipq_s16(zip5.val[0], zip7.val[0]);
                    const int16x8x2_t out_1 = vzipq_s16(zip5.val[1], zip7.val[1]);
                    const int16x8x2_t out_2 = vzipq_s16(zip6.val[0], zip8.val[0]);
                    const int16x8x2_t out_3 = vzipq_s16(zip6.val[1], zip8.val[1]);

                    vst1q_s16(out_row + (j + 0) * out_col_stride + c, out_0.val[0]);
                    vst1q_s16(out_row + (j + 1) * out_col_stride + c, out_0.val[1]);
                    vst1q_s16(out_row + (j + 2) * out_col_stride + c, out_1.val[0]);
                    vst1q_s16(out_row + (j + 3) * out_col_stride + c, out_1.val[1]);
                    vst1q_s16(out_row + (j + 4) * out_col_stride + c, out_2.val[0]);
                    vst1q_s16(out_row + (j + 5) * out_col_stride + c, out_2.val[1]);
                    vst1q_s16(out_row + (j + 6) * out_col_stride + c, out_3.val[0]);
                    vst1q_s16(out_row + (j + 7) * out_col_stride + c, out_3.val[1]);
                }
                for (; c_remaining; c++, c_remaining--)
                {
                    for (int _j = 0; _j < 8; _j++)
                    {
                        const T *const in_col     = in_row + j + _j;
                        T *const       out_col    = out_row + (j + _j) * out_col_stride;
                        const T *const in_channel = in_col + c * in_channel_stride;
                        out_col[c]                = *(in_channel);
                    }
                }
            }
            for (; j_remaining >= 4; j += 4, j_remaining -= 4)
            {
                int c = 0, c_remaining = n_channels;
                for (; c_remaining >= 4; c += 4, c_remaining -= 4)
                {
                    // Read 4 channels worth of 4 columns, then zip to produce 4 columns
                    // worth of 4 channels.
                    int16x4_t channel_pixels[4];
                    channel_pixels[0] = vld1_s16(in_row + (c + 0) * in_channel_stride + j);
                    channel_pixels[1] = vld1_s16(in_row + (c + 1) * in_channel_stride + j);
                    channel_pixels[2] = vld1_s16(in_row + (c + 2) * in_channel_stride + j);
                    channel_pixels[3] = vld1_s16(in_row + (c + 3) * in_channel_stride + j);

                    const int16x4x2_t zip1 = vzip_s16(channel_pixels[0], channel_pixels[2]);
                    const int16x4x2_t zip2 = vzip_s16(channel_pixels[1], channel_pixels[3]);
                    const int16x4x2_t out_0 = vzip_s16(zip1.val[0], zip2.val[0]);
                    const int16x4x2_t out_1 = vzip_s16(zip1.val[1], zip2.val[1]);

                    vst1_s16(out_row + (j + 0) * out_col_stride + c, out_0.val[0]);
                    vst1_s16(out_row + (j + 1) * out_col_stride + c, out_0.val[1]);
                    vst1_s16(out_row + (j + 2) * out_col_stride + c, out_1.val[0]);
                    vst1_s16(out_row + (j + 3) * out_col_stride + c, out_1.val[1]);
                }
                for (; c_remaining; c++, c_remaining--)
                {
                    for (int _j = 0; _j < 4; _j++)
                    {
                        const T *const in_col     = in_row + j + _j;
                        T *const       out_col    = out_row + (j + _j) * out_col_stride;
                        const T *const in_channel = in_col + c * in_channel_stride;
                        out_col[c]                = *(in_channel);
                    }
                }
            }
#endif // __arm_any__
            for (; j_remaining; j++, j_remaining--)
            {
                const T *const in_col  = in_row + j;
                T *const       out_col = out_row + j * out_col_stride;

                for (int c = 0; c < n_channels; c++)
                {
                    const T *const in_channel = in_col + c * in_channel_stride;
                    out_col[c]                = *(in_channel);
                }
            }
        }
    }
}

template <>
inline void nchw_to_nhwc(
  const uint16_t* const in,
  uint16_t* const out,
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
    reinterpret_cast<const int16_t*>(in),
    reinterpret_cast<int16_t*>(out),
    n_batches, n_channels, n_rows, n_cols,
    in_batch_stride, in_channel_stride, in_row_stride,
    out_batch_stride, out_row_stride, out_col_stride
  );
}

#ifdef ARM_COMPUTE_ENABLE_FP16
template <>
inline void nchw_to_nhwc(
  const float16_t* const in,
  float16_t* const out,
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
    reinterpret_cast<const int16_t*>(in),
    reinterpret_cast<int16_t*>(out),
    n_batches, n_channels, n_rows, n_cols,
    in_batch_stride, in_channel_stride, in_row_stride,
    out_batch_stride, out_row_stride, out_col_stride
  );
}
#endif /* ARM_COMPUTE_ENABLE_FP16 */

/*****************************************************************************/
/* 8-bit implementation : NCHW -> NHWC
 */
template <>
inline void nchw_to_nhwc(const int8_t *const in,
                         int8_t *const       out,
                         const int           n_batches,
                         const int           n_channels,
                         const int           n_rows,
                         const int           n_cols,
                         int                 in_batch_stride,
                         int                 in_channel_stride,
                         int                 in_row_stride,
                         int                 out_batch_stride,
                         int                 out_row_stride,
                         int                 out_col_stride)
{
    typedef int8_t T;

    // Fill in the stride values
    in_row_stride     = (in_row_stride) ? in_row_stride : n_cols;
    in_channel_stride = (in_channel_stride) ? in_channel_stride : n_rows * in_row_stride;
    in_batch_stride   = (in_batch_stride) ? in_batch_stride : n_channels * in_channel_stride;

    out_col_stride   = (out_col_stride) ? out_col_stride : n_channels;
    out_row_stride   = (out_row_stride) ? out_row_stride : n_cols * out_col_stride;
    out_batch_stride = (out_batch_stride) ? out_batch_stride : n_rows * out_row_stride;

    // Perform the re-ordering
    for (int n = 0; n < n_batches; n++)
    {
        const T *const in_batch  = in + n * in_batch_stride;
        T *const       out_batch = out + n * out_batch_stride;

        for (int i = 0; i < n_rows; i++)
        {
            const T *const in_row  = in_batch + i * in_row_stride;
            T *const       out_row = out_batch + i * out_row_stride;

            int j = 0, j_remaining = n_cols;
#ifdef __arm_any__
            for (; j_remaining >= 16; j += 16, j_remaining -= 16)
            {

                int c = 0, c_remaining = n_channels;
                for (; c_remaining >= 16; c += 16, c_remaining -= 16)
                {
                    // Read 16 channels worth of 16 columns, then zip to produce 16 columns
                    // worth of 16 channels.
                    int8x16_t channel_pixels[16];

                    channel_pixels[0] = vld1q_s8(in_row + (c + 0) * in_channel_stride + j);
                    channel_pixels[1] = vld1q_s8(in_row + (c + 1) * in_channel_stride + j);
                    channel_pixels[2] = vld1q_s8(in_row + (c + 2) * in_channel_stride + j);
                    channel_pixels[3] = vld1q_s8(in_row + (c + 3) * in_channel_stride + j);

                    channel_pixels[4] = vld1q_s8(in_row + (c + 4) * in_channel_stride + j);
                    channel_pixels[5] = vld1q_s8(in_row + (c + 5) * in_channel_stride + j);
                    channel_pixels[6] = vld1q_s8(in_row + (c + 6) * in_channel_stride + j);
                    channel_pixels[7] = vld1q_s8(in_row + (c + 7) * in_channel_stride + j);

                    channel_pixels[8] = vld1q_s8(in_row + (c + 8) * in_channel_stride + j);
                    channel_pixels[9] = vld1q_s8(in_row + (c + 9) * in_channel_stride + j);
                    channel_pixels[10] = vld1q_s8(in_row + (c + 10) * in_channel_stride + j);
                    channel_pixels[11] = vld1q_s8(in_row + (c + 11) * in_channel_stride + j);

                    channel_pixels[12] = vld1q_s8(in_row + (c + 12) * in_channel_stride + j);
                    channel_pixels[13] = vld1q_s8(in_row + (c + 13) * in_channel_stride + j);
                    channel_pixels[14] = vld1q_s8(in_row + (c + 14) * in_channel_stride + j);
                    channel_pixels[15] = vld1q_s8(in_row + (c + 15) * in_channel_stride + j);

                    // 0th and 8th, 1st and 9th, 2nd and 10th, 3rd and 11th channels
                    const int8x16x2_t zip1  = vzipq_s8(channel_pixels[0], channel_pixels[8]);
                    const int8x16x2_t zip2  = vzipq_s8(channel_pixels[1], channel_pixels[9]);
                    const int8x16x2_t zip3  = vzipq_s8(channel_pixels[2], channel_pixels[10]);
                    const int8x16x2_t zip4  = vzipq_s8(channel_pixels[3], channel_pixels[11]);

                    // 4th and 12th, 5th and 13th, 6th and 14th, 7th and 15th channels
                    const int8x16x2_t zip5  = vzipq_s8(channel_pixels[4], channel_pixels[12]);
                    const int8x16x2_t zip6  = vzipq_s8(channel_pixels[5], channel_pixels[13]);
                    const int8x16x2_t zip7  = vzipq_s8(channel_pixels[6], channel_pixels[14]);
                    const int8x16x2_t zip8  = vzipq_s8(channel_pixels[7], channel_pixels[15]);

                    // 0th, 4th, 8th, 12th channels
                    const int8x16x2_t zip9 = vzipq_s8(zip1.val[0], zip5.val[0]);
                    const int8x16x2_t zip10 = vzipq_s8(zip1.val[1], zip5.val[1]);

                    // 2nd, 6th, 10th, 14th channels
                    const int8x16x2_t zip11 = vzipq_s8(zip3.val[0], zip7.val[0]);
                    const int8x16x2_t zip12 = vzipq_s8(zip3.val[1], zip7.val[1]);

                    // 0th, 2nd, 4th, 6th, 8th, 10th, 12th, 14th channels
                    const int8x16x2_t zip13 = vzipq_s8(zip9.val[0], zip11.val[0]);
                    const int8x16x2_t zip14 = vzipq_s8(zip9.val[1], zip11.val[1]);
                    const int8x16x2_t zip15 = vzipq_s8(zip10.val[0], zip12.val[0]);
                    const int8x16x2_t zip16 = vzipq_s8(zip10.val[1], zip12.val[1]);

                    // 1st, 5th, 9th, 13th channels
                    const int8x16x2_t zip17 = vzipq_s8(zip2.val[0], zip6.val[0]);
                    const int8x16x2_t zip18 = vzipq_s8(zip2.val[1], zip6.val[1]);

                    // 3rd, 7th, 11th, 15th channels
                    const int8x16x2_t zip19 = vzipq_s8(zip4.val[0], zip8.val[0]);
                    const int8x16x2_t zip20 = vzipq_s8(zip4.val[1], zip8.val[1]);

                    // 1st, 3rd, 5th, 7th, 9th, 11th, 13th, 15th channels
                    const int8x16x2_t zip21 = vzipq_s8(zip17.val[0], zip19.val[0]);
                    const int8x16x2_t zip22 = vzipq_s8(zip17.val[1], zip19.val[1]);
                    const int8x16x2_t zip23 = vzipq_s8(zip18.val[0], zip20.val[0]);
                    const int8x16x2_t zip24 = vzipq_s8(zip18.val[1], zip20.val[1]);

                    // 0th, 1st, 2nd, ..., 15th channels
                    const int8x16x2_t out_0 = vzipq_s8(zip13.val[0], zip21.val[0]);
                    const int8x16x2_t out_1 = vzipq_s8(zip13.val[1], zip21.val[1]);
                    const int8x16x2_t out_2 = vzipq_s8(zip14.val[0], zip22.val[0]);
                    const int8x16x2_t out_3 = vzipq_s8(zip14.val[1], zip22.val[1]);
                    const int8x16x2_t out_4 = vzipq_s8(zip15.val[0], zip23.val[0]);
                    const int8x16x2_t out_5 = vzipq_s8(zip15.val[1], zip23.val[1]);
                    const int8x16x2_t out_6 = vzipq_s8(zip16.val[0], zip24.val[0]);
                    const int8x16x2_t out_7 = vzipq_s8(zip16.val[1], zip24.val[1]);

                    vst1q_s8(out_row + (j + 0) * out_col_stride + c, out_0.val[0]);
                    vst1q_s8(out_row + (j + 1) * out_col_stride + c, out_0.val[1]);
                    vst1q_s8(out_row + (j + 2) * out_col_stride + c, out_1.val[0]);
                    vst1q_s8(out_row + (j + 3) * out_col_stride + c, out_1.val[1]);

                    vst1q_s8(out_row + (j + 4) * out_col_stride + c, out_2.val[0]);
                    vst1q_s8(out_row + (j + 5) * out_col_stride + c, out_2.val[1]);
                    vst1q_s8(out_row + (j + 6) * out_col_stride + c, out_3.val[0]);
                    vst1q_s8(out_row + (j + 7) * out_col_stride + c, out_3.val[1]);

                    vst1q_s8(out_row + (j + 8) * out_col_stride + c, out_4.val[0]);
                    vst1q_s8(out_row + (j + 9) * out_col_stride + c, out_4.val[1]);
                    vst1q_s8(out_row + (j + 10) * out_col_stride + c, out_5.val[0]);
                    vst1q_s8(out_row + (j + 11) * out_col_stride + c, out_5.val[1]);

                    vst1q_s8(out_row + (j + 12) * out_col_stride + c, out_6.val[0]);
                    vst1q_s8(out_row + (j + 13) * out_col_stride + c, out_6.val[1]);
                    vst1q_s8(out_row + (j + 14) * out_col_stride + c, out_7.val[0]);
                    vst1q_s8(out_row + (j + 15) * out_col_stride + c, out_7.val[1]);
                }
                for (; c_remaining; c++, c_remaining--)
                {
                    for (int _j = 0; _j < 16; _j++)
                    {
                        const T *const in_col     = in_row + j + _j;
                        T *const       out_col    = out_row + (j + _j) * out_col_stride;
                        const T *const in_channel = in_col + c * in_channel_stride;
                        out_col[c]                = *(in_channel);
                    }
                }
            }
            for (; j_remaining >= 8; j += 8, j_remaining -= 8)
            {
                int c = 0, c_remaining = n_channels;
                for (; c_remaining >= 8; c += 8, c_remaining -= 8)
                {
                    // Read 8 channels worth of 8 columns, then zip to produce 8 columns
                    // worth of 8 channels.
                    int8x8_t channel_pixels[8];

                    channel_pixels[0] = vld1_s8(in_row + (c + 0) * in_channel_stride + j);
                    channel_pixels[1] = vld1_s8(in_row + (c + 1) * in_channel_stride + j);
                    channel_pixels[2] = vld1_s8(in_row + (c + 2) * in_channel_stride + j);
                    channel_pixels[3] = vld1_s8(in_row + (c + 3) * in_channel_stride + j);

                    channel_pixels[4] = vld1_s8(in_row + (c + 4) * in_channel_stride + j);
                    channel_pixels[5] = vld1_s8(in_row + (c + 5) * in_channel_stride + j);
                    channel_pixels[6] = vld1_s8(in_row + (c + 6) * in_channel_stride + j);
                    channel_pixels[7] = vld1_s8(in_row + (c + 7) * in_channel_stride + j);

                    const int8x8x2_t zip1 = vzip_s8(channel_pixels[0], channel_pixels[4]);
                    const int8x8x2_t zip2 = vzip_s8(channel_pixels[1], channel_pixels[5]);
                    const int8x8x2_t zip3 = vzip_s8(channel_pixels[2], channel_pixels[6]);
                    const int8x8x2_t zip4 = vzip_s8(channel_pixels[3], channel_pixels[7]);

                    // 0th, 2nd, 4th, 6th channels
                    const int8x8x2_t zip5 = vzip_s8(zip1.val[0], zip3.val[0]);
                    const int8x8x2_t zip6 = vzip_s8(zip1.val[1], zip3.val[1]);

                    // 1st, 3rd, 5th, 7th channels
                    const int8x8x2_t zip7 = vzip_s8(zip2.val[0], zip4.val[0]);
                    const int8x8x2_t zip8 = vzip_s8(zip2.val[1], zip4.val[1]);

                    // 0th, 1st, 2nd, ..., 7th channels
                    const int8x8x2_t out_0 = vzip_s8(zip5.val[0], zip7.val[0]);
                    const int8x8x2_t out_1 = vzip_s8(zip5.val[1], zip7.val[1]);
                    const int8x8x2_t out_2 = vzip_s8(zip6.val[0], zip8.val[0]);
                    const int8x8x2_t out_3 = vzip_s8(zip6.val[1], zip8.val[1]);

                    vst1_s8(out_row + (j + 0) * out_col_stride + c, out_0.val[0]);
                    vst1_s8(out_row + (j + 1) * out_col_stride + c, out_0.val[1]);
                    vst1_s8(out_row + (j + 2) * out_col_stride + c, out_1.val[0]);
                    vst1_s8(out_row + (j + 3) * out_col_stride + c, out_1.val[1]);

                    vst1_s8(out_row + (j + 4) * out_col_stride + c, out_2.val[0]);
                    vst1_s8(out_row + (j + 5) * out_col_stride + c, out_2.val[1]);
                    vst1_s8(out_row + (j + 6) * out_col_stride + c, out_3.val[0]);
                    vst1_s8(out_row + (j + 7) * out_col_stride + c, out_3.val[1]);
                }
                for (; c_remaining; c++, c_remaining--)
                {
                    for (int _j = 0; _j < 8; _j++)
                    {
                        const T *const in_col     = in_row + j + _j;
                        T *const       out_col    = out_row + (j + _j) * out_col_stride;
                        const T *const in_channel = in_col + c * in_channel_stride;
                        out_col[c]                = *(in_channel);
                    }
                }
            }
#endif // __arm_any__
            for (; j_remaining; j++, j_remaining--)
            {
                const T *const in_col  = in_row + j;
                T *const       out_col = out_row + j * out_col_stride;

                for (int c = 0; c < n_channels; c++)
                {
                    const T *const in_channel = in_col + c * in_channel_stride;
                    out_col[c]                = *(in_channel);
                }
            }
        }
    }
}

template <>
inline void nchw_to_nhwc(const uint8_t *const in,
                         uint8_t *const       out,
                         const int            n_batches,
                         const int            n_channels,
                         const int            n_rows,
                         const int            n_cols,
                         int                  in_batch_stride,
                         int                  in_channel_stride,
                         int                  in_row_stride,
                         int                  out_batch_stride,
                         int                  out_row_stride,
                         int                  out_col_stride)
{
    nchw_to_nhwc(reinterpret_cast<const int8_t *>(in), reinterpret_cast<int8_t *>(out), n_batches, n_channels, n_rows,
                 n_cols, in_batch_stride, in_channel_stride, in_row_stride, out_batch_stride, out_row_stride,
                 out_col_stride);
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
/* 16-bit implementation : NHWC -> NCHW
 */
template <>
inline void nhwc_to_nchw(
  const int16_t* const in,  // Input data in NHWC form
  int16_t* const out,       // Output data in NCHW form
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
  typedef int16_t T;

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

      // For every column, beginning with chunks of 8
      int j = 0, j_remaining = n_cols;
#ifdef __arm_any__
      for (; j_remaining >= 8; j += 8, j_remaining -=8)
      {
        // For every channel, beginning with chunks of 8
        int c = 0, c_remaining = n_channels;
        for (; c_remaining >= 8; c += 8, c_remaining -= 8)
        {
          // Read 8 columns worth of 8 channels then zip to produce 8 channels
          // worth of 8 columns.
          int16x8_t pixel_channels[8];

          pixel_channels[0] = vld1q_s16(in_i + (j + 0)*in_col_stride + c);
          pixel_channels[1] = vld1q_s16(in_i + (j + 1)*in_col_stride + c);
          pixel_channels[2] = vld1q_s16(in_i + (j + 2)*in_col_stride + c);
          pixel_channels[3] = vld1q_s16(in_i + (j + 3)*in_col_stride + c);
          pixel_channels[4] = vld1q_s16(in_i + (j + 4)*in_col_stride + c);
          pixel_channels[5] = vld1q_s16(in_i + (j + 5)*in_col_stride + c);
          pixel_channels[6] = vld1q_s16(in_i + (j + 6)*in_col_stride + c);
          pixel_channels[7] = vld1q_s16(in_i + (j + 7)*in_col_stride + c);

          // 0th and 4th, 1st and 5th, 2nd and 6th, 3rd and 7th columns
          const int16x8x2_t zip1 = vzipq_s16(pixel_channels[0], pixel_channels[4]);
          const int16x8x2_t zip2 = vzipq_s16(pixel_channels[1], pixel_channels[5]);
          const int16x8x2_t zip3 = vzipq_s16(pixel_channels[2], pixel_channels[6]);
          const int16x8x2_t zip4 = vzipq_s16(pixel_channels[3], pixel_channels[7]);

          // 0th, 2nd, 4th, 6th columns
          const int16x8x2_t zip5 = vzipq_s16(zip1.val[0], zip3.val[0]);
          const int16x8x2_t zip6 = vzipq_s16(zip1.val[1], zip3.val[1]);

          // 1st, 3rd, 5th, 7th columns
          const int16x8x2_t zip7 = vzipq_s16(zip2.val[0], zip4.val[0]);
          const int16x8x2_t zip8 = vzipq_s16(zip2.val[1], zip4.val[1]);

          // 0th, 1st, ..., 7th columns
          const int16x8x2_t out_0 = vzipq_s16(zip5.val[0], zip7.val[0]);
          const int16x8x2_t out_1 = vzipq_s16(zip5.val[1], zip7.val[1]);
          const int16x8x2_t out_2 = vzipq_s16(zip6.val[0], zip8.val[0]);
          const int16x8x2_t out_3 = vzipq_s16(zip6.val[1], zip8.val[1]);

          // 0th, 1st, 2nd, 3rd columns
          vst1q_s16(out_i + j + (c + 0)*out_channel_stride, out_0.val[0]);
          vst1q_s16(out_i + j + (c + 1)*out_channel_stride, out_0.val[1]);
          vst1q_s16(out_i + j + (c + 2)*out_channel_stride, out_1.val[0]);
          vst1q_s16(out_i + j + (c + 3)*out_channel_stride, out_1.val[1]);

          // 4th, 5th, 6th, 7th columns
          vst1q_s16(out_i + j + (c + 4)*out_channel_stride, out_2.val[0]);
          vst1q_s16(out_i + j + (c + 5)*out_channel_stride, out_2.val[1]);
          vst1q_s16(out_i + j + (c + 6)*out_channel_stride, out_3.val[0]);
          vst1q_s16(out_i + j + (c + 7)*out_channel_stride, out_3.val[1]);
        }
        for (; c_remaining; c++, c_remaining--)
        {
          for (int _j = 0; _j < 8; _j++)
          {
            const T* const in_j = in_i + (j + _j)*in_col_stride;
            T* const out_j = out_i + (j + _j);

            const T* const in_channel = in_j + c;
            T* const out_channel = out_j + c*out_channel_stride;
            *(out_channel) = *(in_channel);
          }
        }
      }
      for (; j_remaining >= 4; j += 4, j_remaining -=4)
      {
        int c = 0, c_remaining = n_channels;
        for (; c_remaining >= 4; c += 4, c_remaining -= 4)
        {
          // Read 4 columns worth of 4 channels then zip to produce 4 channels
          // worth of 4 columns.
          int16x4_t pixel_channels[4];

          pixel_channels[0] = vld1_s16(in_i + (j + 0)*in_col_stride + c);
          pixel_channels[1] = vld1_s16(in_i + (j + 1)*in_col_stride + c);
          pixel_channels[2] = vld1_s16(in_i + (j + 2)*in_col_stride + c);
          pixel_channels[3] = vld1_s16(in_i + (j + 3)*in_col_stride + c);

          // 0th and 2nd, 1st and 3rd columns
          const int16x4x2_t zip1 = vzip_s16(pixel_channels[0], pixel_channels[2]);
          const int16x4x2_t zip2 = vzip_s16(pixel_channels[1], pixel_channels[3]);

          // 0th, 1st, 2nd, 3rd columns
          const int16x4x2_t out_0 = vzip_s16(zip1.val[0], zip2.val[0]);
          const int16x4x2_t out_1 = vzip_s16(zip1.val[1], zip2.val[1]);

          // 0th, 1st, 2nd, 3rd columns
          vst1_s16(out_i + j + (c + 0)*out_channel_stride, out_0.val[0]);
          vst1_s16(out_i + j + (c + 1)*out_channel_stride, out_0.val[1]);
          vst1_s16(out_i + j + (c + 2)*out_channel_stride, out_1.val[0]);
          vst1_s16(out_i + j + (c + 3)*out_channel_stride, out_1.val[1]);
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
  const uint16_t* const in,  // Input data in NHWC form
  uint16_t* const out,       // Output data in NCHW form
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
  nhwc_to_nchw(
    reinterpret_cast<const int16_t*>(in),
    reinterpret_cast<int16_t*>(out),
    n_batches, n_rows, n_cols, n_channels,
    in_batch_stride, in_row_stride, in_col_stride,
    out_batch_stride, out_channel_stride, out_row_stride
  );
}

#ifdef ARM_COMPUTE_ENABLE_FP16
template <>
inline void nhwc_to_nchw(
  const float16_t* const in,  // Input data in NHWC form
  float16_t* const out,       // Output data in NCHW form
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
  nhwc_to_nchw(
    reinterpret_cast<const int16_t*>(in),
    reinterpret_cast<int16_t*>(out),
    n_batches, n_rows, n_cols, n_channels,
    in_batch_stride, in_row_stride, in_col_stride,
    out_batch_stride, out_channel_stride, out_row_stride
  );
}
#endif /* ARM_COMPUTE_ENABLE_FP16 */

template <>
inline void nhwc_to_nchw(
  const int8_t* const in,  // Input data in NHWC form
  int8_t* const out,       // Output data in NCHW form
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
  typedef int8_t T;

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

      // For every column, beginning with chunks of 16
      int j = 0, j_remaining = n_cols;
#ifdef __arm_any__
      for (; j_remaining >= 16; j += 16, j_remaining -=16)
      {
        // For every channel, beginning with chunks of 16
        int c = 0, c_remaining = n_channels;
        for (; c_remaining >= 16; c += 16, c_remaining -= 16)
        {
          // Read 16 columns worth of 16 channels then zip to produce 16 channels
          // worth of 16 columns.
          int8x16_t pixel_channels[16];

          pixel_channels[0] = vld1q_s8(in_i + (j + 0)*in_col_stride + c);
          pixel_channels[1] = vld1q_s8(in_i + (j + 1)*in_col_stride + c);
          pixel_channels[2] = vld1q_s8(in_i + (j + 2)*in_col_stride + c);
          pixel_channels[3] = vld1q_s8(in_i + (j + 3)*in_col_stride + c);

          pixel_channels[4] = vld1q_s8(in_i + (j + 4)*in_col_stride + c);
          pixel_channels[5] = vld1q_s8(in_i + (j + 5)*in_col_stride + c);
          pixel_channels[6] = vld1q_s8(in_i + (j + 6)*in_col_stride + c);
          pixel_channels[7] = vld1q_s8(in_i + (j + 7)*in_col_stride + c);

          pixel_channels[8] = vld1q_s8(in_i + (j + 8)*in_col_stride + c);
          pixel_channels[9] = vld1q_s8(in_i + (j + 9)*in_col_stride + c);
          pixel_channels[10] = vld1q_s8(in_i + (j + 10)*in_col_stride + c);
          pixel_channels[11] = vld1q_s8(in_i + (j + 11)*in_col_stride + c);

          pixel_channels[12] = vld1q_s8(in_i + (j + 12)*in_col_stride + c);
          pixel_channels[13] = vld1q_s8(in_i + (j + 13)*in_col_stride + c);
          pixel_channels[14] = vld1q_s8(in_i + (j + 14)*in_col_stride + c);
          pixel_channels[15] = vld1q_s8(in_i + (j + 15)*in_col_stride + c);

          // 0th and 8th, 1st and 9th, 2nd and 10th, 3rd and 11th columns
          const int8x16x2_t zip1 = vzipq_s8(pixel_channels[0], pixel_channels[8]);
          const int8x16x2_t zip2 = vzipq_s8(pixel_channels[1], pixel_channels[9]);
          const int8x16x2_t zip3 = vzipq_s8(pixel_channels[2], pixel_channels[10]);
          const int8x16x2_t zip4 = vzipq_s8(pixel_channels[3], pixel_channels[11]);

          // 4th and 12th, 5th and 13th, 6th and 14th, 7th and 15th columns
          const int8x16x2_t zip5 = vzipq_s8(pixel_channels[4], pixel_channels[12]);
          const int8x16x2_t zip6 = vzipq_s8(pixel_channels[5], pixel_channels[13]);
          const int8x16x2_t zip7 = vzipq_s8(pixel_channels[6], pixel_channels[14]);
          const int8x16x2_t zip8 = vzipq_s8(pixel_channels[7], pixel_channels[15]);

          // 0th, 4th, 8th, 12th columns
          const int8x16x2_t zip9 = vzipq_s8(zip1.val[0], zip5.val[0]);
          const int8x16x2_t zip10 = vzipq_s8(zip1.val[1], zip5.val[1]);

          // 2nd, 6th, 10th, 14th columns
          const int8x16x2_t zip11 = vzipq_s8(zip3.val[0], zip7.val[0]);
          const int8x16x2_t zip12 = vzipq_s8(zip3.val[1], zip7.val[1]);

          // 0th, 2nd, 4th, 6th, 8th, 10th, 12th, 14th columns
          const int8x16x2_t zip13 = vzipq_s8(zip9.val[0], zip11.val[0]);
          const int8x16x2_t zip14 = vzipq_s8(zip9.val[1], zip11.val[1]);
          const int8x16x2_t zip15 = vzipq_s8(zip10.val[0], zip12.val[0]);
          const int8x16x2_t zip16 = vzipq_s8(zip10.val[1], zip12.val[1]);

          // 1st, 5th, 9th, 13th columns
          const int8x16x2_t zip17 = vzipq_s8(zip2.val[0], zip6.val[0]);
          const int8x16x2_t zip18 = vzipq_s8(zip2.val[1], zip6.val[1]);

          // 3rd, 7th, 11th, 15th columns
          const int8x16x2_t zip19 = vzipq_s8(zip4.val[0], zip8.val[0]);
          const int8x16x2_t zip20 = vzipq_s8(zip4.val[1], zip8.val[1]);

          // 1st, 3rd, 5th, 7th, 9th, 11th, 13th, 15th columns
          const int8x16x2_t zip21 = vzipq_s8(zip17.val[0], zip19.val[0]);
          const int8x16x2_t zip22 = vzipq_s8(zip17.val[1], zip19.val[1]);
          const int8x16x2_t zip23 = vzipq_s8(zip18.val[0], zip20.val[0]);
          const int8x16x2_t zip24 = vzipq_s8(zip18.val[1], zip20.val[1]);

          // 0th, 1st, 2nd, 4th, ..., 15th columns
          const int8x16x2_t out_0 = vzipq_s8(zip13.val[0], zip21.val[0]);
          const int8x16x2_t out_1 = vzipq_s8(zip13.val[1], zip21.val[1]);
          const int8x16x2_t out_2 = vzipq_s8(zip14.val[0], zip22.val[0]);
          const int8x16x2_t out_3 = vzipq_s8(zip14.val[1], zip22.val[1]);
          const int8x16x2_t out_4 = vzipq_s8(zip15.val[0], zip23.val[0]);
          const int8x16x2_t out_5 = vzipq_s8(zip15.val[1], zip23.val[1]);
          const int8x16x2_t out_6 = vzipq_s8(zip16.val[0], zip24.val[0]);
          const int8x16x2_t out_7 = vzipq_s8(zip16.val[1], zip24.val[1]);

          // 0th, 1st, 2nd, 3rd columns
          vst1q_s8(out_i + j + (c + 0)*out_channel_stride, out_0.val[0]);
          vst1q_s8(out_i + j + (c + 1)*out_channel_stride, out_0.val[1]);
          vst1q_s8(out_i + j + (c + 2)*out_channel_stride, out_1.val[0]);
          vst1q_s8(out_i + j + (c + 3)*out_channel_stride, out_1.val[1]);

          // 4th, 5th, 6th, 7th columns
          vst1q_s8(out_i + j + (c + 4)*out_channel_stride, out_2.val[0]);
          vst1q_s8(out_i + j + (c + 5)*out_channel_stride, out_2.val[1]);
          vst1q_s8(out_i + j + (c + 6)*out_channel_stride, out_3.val[0]);
          vst1q_s8(out_i + j + (c + 7)*out_channel_stride, out_3.val[1]);

          // 8th, 9th, 10th, 11th columns
          vst1q_s8(out_i + j + (c + 8)*out_channel_stride, out_4.val[0]);
          vst1q_s8(out_i + j + (c + 9)*out_channel_stride, out_4.val[1]);
          vst1q_s8(out_i + j + (c + 10)*out_channel_stride, out_5.val[0]);
          vst1q_s8(out_i + j + (c + 11)*out_channel_stride, out_5.val[1]);

          // 12th, 13th, 14th, 15th columns
          vst1q_s8(out_i + j + (c + 12)*out_channel_stride, out_6.val[0]);
          vst1q_s8(out_i + j + (c + 13)*out_channel_stride, out_6.val[1]);
          vst1q_s8(out_i + j + (c + 14)*out_channel_stride, out_7.val[0]);
          vst1q_s8(out_i + j + (c + 15)*out_channel_stride, out_7.val[1]);

        }
        for (; c_remaining; c++, c_remaining--)
        {
          for (int _j = 0; _j < 16; _j++)
          {
            const T* const in_j = in_i + (j + _j)*in_col_stride;
            T* const out_j = out_i + (j + _j);

            const T* const in_channel = in_j + c;
            T* const out_channel = out_j + c*out_channel_stride;
            *(out_channel) = *(in_channel);
          }
        }
      }
      for (; j_remaining >= 8; j += 8, j_remaining -= 8)
      {
        int c = 0, c_remaining = n_channels;
        for (; c_remaining >= 8; c += 8, c_remaining -= 8)
        {
          // Read 8 columns worth of 8 channels then zip to produce 8 channels
          // worth of 8 columns.
          int8x8_t pixel_channels[8];

          pixel_channels[0] = vld1_s8(in_i + (j + 0)*in_col_stride + c);
          pixel_channels[1] = vld1_s8(in_i + (j + 1)*in_col_stride + c);
          pixel_channels[2] = vld1_s8(in_i + (j + 2)*in_col_stride + c);
          pixel_channels[3] = vld1_s8(in_i + (j + 3)*in_col_stride + c);

          pixel_channels[4] = vld1_s8(in_i + (j + 4)*in_col_stride + c);
          pixel_channels[5] = vld1_s8(in_i + (j + 5)*in_col_stride + c);
          pixel_channels[6] = vld1_s8(in_i + (j + 6)*in_col_stride + c);
          pixel_channels[7] = vld1_s8(in_i + (j + 7)*in_col_stride + c);

          // 0th and 4th, 1st and 5th, 2nd and 6th, 3rd and 7th columns
          const int8x8x2_t zip1 = vzip_s8(pixel_channels[0], pixel_channels[4]);
          const int8x8x2_t zip2 = vzip_s8(pixel_channels[1], pixel_channels[5]);
          const int8x8x2_t zip3 = vzip_s8(pixel_channels[2], pixel_channels[6]);
          const int8x8x2_t zip4 = vzip_s8(pixel_channels[3], pixel_channels[7]);

          // 0th, 2nd, 4th, 6th columns
          const int8x8x2_t zip5 = vzip_s8(zip1.val[0], zip3.val[0]);
          const int8x8x2_t zip6 = vzip_s8(zip1.val[1], zip3.val[1]);

          // 1st, 3rd, 5th, 7th columns
          const int8x8x2_t zip7 = vzip_s8(zip2.val[0], zip4.val[0]);
          const int8x8x2_t zip8 = vzip_s8(zip2.val[1], zip4.val[1]);

          // 0th, 1st, ..., 7th columns
          const int8x8x2_t out_0 = vzip_s8(zip5.val[0], zip7.val[0]);
          const int8x8x2_t out_1 = vzip_s8(zip5.val[1], zip7.val[1]);
          const int8x8x2_t out_2 = vzip_s8(zip6.val[0], zip8.val[0]);
          const int8x8x2_t out_3 = vzip_s8(zip6.val[1], zip8.val[1]);

          // 0th, 1st, 2nd, 3rd columns
          vst1_s8(out_i + j + (c + 0)*out_channel_stride, out_0.val[0]);
          vst1_s8(out_i + j + (c + 1)*out_channel_stride, out_0.val[1]);
          vst1_s8(out_i + j + (c + 2)*out_channel_stride, out_1.val[0]);
          vst1_s8(out_i + j + (c + 3)*out_channel_stride, out_1.val[1]);

          // 4th, 5th, 6th, 7th columns
          vst1_s8(out_i + j + (c + 4)*out_channel_stride, out_2.val[0]);
          vst1_s8(out_i + j + (c + 5)*out_channel_stride, out_2.val[1]);
          vst1_s8(out_i + j + (c + 6)*out_channel_stride, out_3.val[0]);
          vst1_s8(out_i + j + (c + 7)*out_channel_stride, out_3.val[1]);
        }
        for (; c_remaining; c++, c_remaining--)
        {
          for (int _j = 0; _j < 8; _j++)
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
  const uint8_t* const in,  // Input data in NHWC form
  uint8_t* const out,       // Output data in NCHW form
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
  nhwc_to_nchw(
    reinterpret_cast<const int8_t*>(in),
    reinterpret_cast<int8_t*>(out),
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
