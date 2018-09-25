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

#include "arm_compute/core/NEON/kernels/convolution/winograd/transforms/input.hpp"
#include "arm_compute/core/NEON/kernels/convolution/winograd/winograd_gemm.hpp"
#include "arm_compute/core/NEON/kernels/convolution/common/arm.hpp"

namespace
{

template <bool Specialized, int PadTop=0, int PadLeft=0, int PadBottom=0, int PadRight=0>
void winograd_input_transform_1x8_fp32_process_tile(
  int n_channels,
  const float* const input_base,
  const int input_row_stride,
  const int input_col_stride,
  float* const matrix_base,
  const int matrix_stride,
     const int _pad_top,
     const int _pad_left,
     const int _pad_bottom,
    const int _pad_right
)
{
  (void) input_row_stride;  // No rows over which to stride
 (void) _pad_top;  // Never any top padding
  (void) _pad_bottom;  // Never any bottom padding

  // Extract padding arguments
  const int pad_left = Specialized ? PadLeft : _pad_left;
  const int pad_right = Specialized ? PadRight : _pad_right;

  constexpr int inner_tile_cols = 8;
  const int cells_j = inner_tile_cols - pad_right;

  float *outptr = matrix_base;

  // Get pointers into the input tile
  const float *x_ptrs[inner_tile_cols];
  for (int j = pad_left, xj = 0; j < cells_j; j++, xj++)
  {
    x_ptrs[j] = input_base + xj*input_col_stride;
  }

  // Vectors used/computed in this kernel.
  float x[inner_tile_cols];
  float U[inner_tile_cols];

  for (int j = 0; j < inner_tile_cols; j++)
  {
    x[j] = 0.0f;
  }

  // Perform the Winograd input transformation for each channel in the input
  // tensor.
  int channels_remaining = n_channels;
#ifdef __arm_any__
  for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    float32x4_t x[inner_tile_cols], U[inner_tile_cols];
    for (int j = 0; j < inner_tile_cols; j++)
    {
      x[j] = vdupq_n_f32(0.0f);
    }

    // Load x
    for (int j = pad_left; j < cells_j; j++)
    {
      x[j] = vld1q_f32(x_ptrs[j]);
      x_ptrs[j] += 4;
    }

    // Compute U = x . X
    U[0] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[2], 49), x[4], -14), x[0], -36);
    U[1] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[2], 36), x[3], 13), x[4], -13), x[1], -36), x[5], -1);
    U[2] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[5], 1), x[2], 36), x[1], 36), x[4], -13), x[3], -13);
    U[3] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[3], 20), x[2], 9), x[5], -2), x[4], -10), x[1], -18);
    U[4] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[1], 18), x[2], 9), x[5], 2), x[4], -10), x[3], -20);
    U[5] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[3], 15), x[2], 4), x[5], -3), x[4], -5), x[1], -12);
    U[6] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[1], 12), x[2], 4), x[5], 3), x[4], -5), x[3], -15);
    U[7] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[7], 1), x[3], 49), x[5], -14), x[1], -36);

    // Store the transformed vector
    for (int j = 0; j < inner_tile_cols; j++)
    {
      vst1q_f32(outptr + j*matrix_stride, U[j]);
    }
    outptr += 4;
  }
  for (; channels_remaining >= 2; channels_remaining -= 2)
  {
    float32x2_t x[inner_tile_cols], U[inner_tile_cols];
    for (int j = 0; j < inner_tile_cols; j++)
    {
      x[j] = vdup_n_f32(0.0f);
    }

    // Load x
    for (int j = pad_left; j < cells_j; j++)
    {
      x[j] = vld1_f32(x_ptrs[j]);
      x_ptrs[j] += 2;
    }

    // Compute U = x . X
    U[0] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[2], 49), x[4], -14), x[0], -36);
    U[1] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[2], 36), x[3], 13), x[4], -13), x[1], -36), x[5], -1);
    U[2] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[5], 1), x[2], 36), x[1], 36), x[4], -13), x[3], -13);
    U[3] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[3], 20), x[2], 9), x[5], -2), x[4], -10), x[1], -18);
    U[4] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[1], 18), x[2], 9), x[5], 2), x[4], -10), x[3], -20);
    U[5] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[3], 15), x[2], 4), x[5], -3), x[4], -5), x[1], -12);
    U[6] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[1], 12), x[2], 4), x[5], 3), x[4], -5), x[3], -15);
    U[7] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[7], 1), x[3], 49), x[5], -14), x[1], -36);

    // Store the transformed vector
    for (int j = 0; j < inner_tile_cols; j++)
    {
      vst1_f32(outptr + j*matrix_stride, U[j]);
    }
    outptr += 2;
  }
#endif  // __arm_any__
  for (; channels_remaining; channels_remaining--)
  {
    // Load x
    for (int j = pad_left; j < cells_j; j++)
    {
      x[j] = *(x_ptrs[j]++);
    }

    // Compute U = x . X
    U[0] = x[0]*-36 + x[4]*-14 + x[2]*49 + x[6]*1;
    U[1] = x[5]*-1 + x[1]*-36 + x[4]*-13 + x[3]*13 + x[2]*36 + x[6]*1;
    U[2] = x[3]*-13 + x[4]*-13 + x[1]*36 + x[2]*36 + x[5]*1 + x[6]*1;
    U[3] = x[1]*-18 + x[4]*-10 + x[5]*-2 + x[2]*9 + x[3]*20 + x[6]*1;
    U[4] = x[3]*-20 + x[4]*-10 + x[5]*2 + x[2]*9 + x[1]*18 + x[6]*1;
    U[5] = x[1]*-12 + x[4]*-5 + x[5]*-3 + x[2]*4 + x[3]*15 + x[6]*1;
    U[6] = x[3]*-15 + x[4]*-5 + x[5]*3 + x[2]*4 + x[1]*12 + x[6]*1;
    U[7] = x[1]*-36 + x[5]*-14 + x[3]*49 + x[7]*1;

    // Store the transformed vector
    for (int j = 0; j < inner_tile_cols; j++)
    {
      *(outptr + j*matrix_stride) = U[j];
    }
    outptr++;
  }
}

}

namespace winograd
{
template <int x>
using Tiles = InputTransformImplTiles<1, x, 1, 8, float>;

/*****************************************************************************/
// 1x3 specialisations
template <>
const Tiles<3>::TileFn Tiles<3>::tilefn_generic = winograd_input_transform_1x8_fp32_process_tile<false>;

template <>
const Tiles<3>::TileFn Tiles<3>::tilefn_unpadded = winograd_input_transform_1x8_fp32_process_tile<true>;

template <>
const Tiles<3>::TileFn Tiles<3>::tilefn_left_padded[n_pad_left] = {
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 1, 0, 0>,
};

template <>
const Tiles<3>::TileFn Tiles<3>::tilefn_right_padded[n_pad_right] = {
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 1>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 2>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 3>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 4>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 5>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 6>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 7>,
};
/*****************************************************************************/

/*****************************************************************************/
// 1x5 specialisations
template <>
const Tiles<5>::TileFn Tiles<5>::tilefn_generic = winograd_input_transform_1x8_fp32_process_tile<false>;

template <>
const Tiles<5>::TileFn Tiles<5>::tilefn_unpadded = winograd_input_transform_1x8_fp32_process_tile<true>;

template <>
const Tiles<5>::TileFn Tiles<5>::tilefn_left_padded[n_pad_left] = {
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 2, 0, 0>,
};

template <>
const Tiles<5>::TileFn Tiles<5>::tilefn_right_padded[n_pad_right] = {
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 1>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 2>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 3>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 4>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 5>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 6>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 7>,
};
/*****************************************************************************/

/*****************************************************************************/
// 1x7 specialisations
template <>
const Tiles<7>::TileFn Tiles<7>::tilefn_generic = winograd_input_transform_1x8_fp32_process_tile<false>;

template <>
const Tiles<7>::TileFn Tiles<7>::tilefn_unpadded = winograd_input_transform_1x8_fp32_process_tile<true>;

template <>
const Tiles<7>::TileFn Tiles<7>::tilefn_left_padded[n_pad_left] = {
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 1, 0, 0>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 3, 0, 0>,
};

template <>
const Tiles<7>::TileFn Tiles<7>::tilefn_right_padded[n_pad_right] = {
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 1>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 2>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 3>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 4>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 5>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 6>,
  winograd_input_transform_1x8_fp32_process_tile<true, 0, 0, 0, 7>,
};
/*****************************************************************************/


template class InputTransform<1, 3, 1, 8, float>;
template class InputTransform<3, 1, 8, 1, float>;
template class InputTransform<1, 5, 1, 8, float>;
template class InputTransform<5, 1, 8, 1, float>;
template class InputTransform<1, 7, 1, 8, float>;
template class InputTransform<7, 1, 8, 1, float>;
}  // namespace winograd
