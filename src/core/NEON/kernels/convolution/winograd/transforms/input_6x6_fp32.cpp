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
void winograd_input_transform_6x6_fp32_process_tile(
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
  const int pad_top = Specialized ? PadTop : _pad_top;
  const int pad_left = Specialized ? PadLeft : _pad_left;
  const int pad_bottom = Specialized ? PadBottom : _pad_bottom;
  const int pad_right = Specialized ? PadRight : _pad_right;

 constexpr int inner_tile_rows = 6;
  constexpr int inner_tile_cols = 6;

  const int cells_i = inner_tile_rows - pad_bottom;
  const int cells_j = inner_tile_cols - pad_right;

  float *outptr = matrix_base;

  // Get pointers into the input tile
  const float *x_ptrs[inner_tile_rows][inner_tile_cols];
  for (int i = pad_top, xi = 0; i < cells_i; i++, xi++)
  {
    // Get a pointer into the row
    const float* const row_ptr = input_base + xi*input_row_stride;

    for (int j = pad_left, xj = 0; j < cells_j; j++, xj++)
    {
      x_ptrs[i][j] = row_ptr + xj*input_col_stride;
    }
  }

  // Matrices used/computed in this kernel.
  float x[inner_tile_rows][inner_tile_cols];
  float XTx[inner_tile_rows][inner_tile_cols];
  float U[inner_tile_rows][inner_tile_cols];
  for (int i = 0; i < inner_tile_rows; i++)
  {
    for (int j = 0; j < inner_tile_cols; j++)
    {
      x[i][j] = XTx[i][j] = 0.0f;
    }
  }

  // Perform the Winograd input transformation for each channel in the input
  // tensor.
  int channels_remaining = n_channels;
#ifdef __aarch64__
  for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    // Matrices used/computed in this kernel
    float32x4_t x[inner_tile_rows][inner_tile_cols];
    float32x4_t XTx[inner_tile_rows][inner_tile_cols];
    float32x4_t U[inner_tile_rows][inner_tile_cols];
    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        x[i][j] = vdupq_n_f32(0.0f);
        XTx[i][j] = vdupq_n_f32(0.0f);
      }
    }

    // Read a 6x6 tile in the Winograd domain
    for (int i = pad_top; i < cells_i; i++)
    {
      for (int j = pad_left; j < cells_j; j++)
      {
        x[i][j] = vld1q_f32(x_ptrs[i][j]);
        x_ptrs[i][j] += 4;
      }
    }

    // Compute XT . x
    for (int j = pad_left; j < cells_j; j++)
    {
      // XTx[0][j] =  4*x[0][j] + -5*x[2][j] +  1*x[4][j];
      XTx[0][j] = vmlsq_n_f32(vmlaq_n_f32(x[4][j], x[0][j], 4.0f), x[2][j], 5.0f);

      // XTx[1][j] = -4*x[1][j] + -4*x[2][j] +  1*x[3][j] +  1*x[4][j];
      XTx[1][j] = vmlsq_n_f32(vaddq_f32(x[3][j], x[4][j]), vaddq_f32(x[1][j], x[2][j]), 4.0f);

      // XTx[2][j] =  4*x[1][j] + -4*x[2][j] + -1*x[3][j] +  1*x[4][j];
      XTx[2][j] = vmlaq_n_f32(vsubq_f32(x[4][j], x[3][j]), vsubq_f32(x[1][j], x[2][j]), 4.0f);

      // XTx[3][j] = -2*x[1][j] + -1*x[2][j] +  2*x[3][j] +  1*x[4][j];
      XTx[3][j] = vmlaq_n_f32(vsubq_f32(x[4][j], x[2][j]), vsubq_f32(x[3][j], x[1][j]), 2.0f);

      // XTx[4][j] =  2*x[1][j] + -1*x[2][j] + -2*x[3][j] +  1*x[4][j];
      XTx[4][j] = vmlaq_n_f32(vsubq_f32(x[4][j], x[2][j]), vsubq_f32(x[1][j], x[3][j]), 2.0f);

      // XTx[5][j] =  4*x[1][j] + -5*x[3][j] +  1*x[5][j];
      XTx[5][j] = vmlsq_n_f32(vmlaq_n_f32(x[5][j], x[1][j], 4.0f), x[3][j], 5.0f);
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_rows; i++)
    {
      // U[i][0] =  4*XTx[i][0] + -5*XTx[i][2] +  1*XTx[i][4];
      U[i][0] = vmlsq_n_f32(vmlaq_n_f32(XTx[i][4], XTx[i][0], 4.0f), XTx[i][2], 5.0f);

      // U[i][1] = -4*XTx[i][1] + -4*XTx[i][2] +  1*XTx[i][3] +  1*XTx[i][4];
      U[i][1] = vmlsq_n_f32(vaddq_f32(XTx[i][3], XTx[i][4]), vaddq_f32(XTx[i][1], XTx[i][2]), 4.0f);

      // U[i][2] =  4*XTx[i][1] + -4*XTx[i][2] + -1*XTx[i][3] +  1*XTx[i][4];
      U[i][2] = vmlaq_n_f32(vsubq_f32(XTx[i][4], XTx[i][3]), vsubq_f32(XTx[i][1], XTx[i][2]), 4.0f);

      // U[i][3] = -2*XTx[i][1] + -1*XTx[i][2] +  2*XTx[i][3] +  1*XTx[i][4];
      U[i][3] = vmlaq_n_f32(vsubq_f32(XTx[i][4], XTx[i][2]), vsubq_f32(XTx[i][3], XTx[i][1]), 2.0f);

      // U[i][4] =  2*XTx[i][1] + -1*XTx[i][2] + -2*XTx[i][3] +  1*XTx[i][4];
      U[i][4] = vmlaq_n_f32(vsubq_f32(XTx[i][4], XTx[i][2]), vsubq_f32(XTx[i][1], XTx[i][3]), 2.0f);

      // U[i][5] =  4*XTx[i][1] + -5*XTx[i][3] +  1*XTx[i][5];
      U[i][5] = vmlsq_n_f32(vmlaq_n_f32(XTx[i][5], XTx[i][1], 4.0f), XTx[i][3], 5.0f);
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++, m++)
      {
        vst1q_f32(outptr + m*matrix_stride, U[i][j]);
      }
    }
    outptr += 4;
  }
#endif  // __aarch64__
#ifdef __arm_any__
  for (; channels_remaining >= 2; channels_remaining -= 2)
  {
    // Matrices used/computed in this kernel
    float32x2_t x[inner_tile_rows][inner_tile_cols];
    float32x2_t XTx[inner_tile_rows][inner_tile_cols];
    float32x2_t U[inner_tile_rows][inner_tile_cols];
    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        x[i][j] = vdup_n_f32(0.0f);
        XTx[i][j] = vdup_n_f32(0.0f);
      }
    }

    // Read a 6x6 tile in the Winograd domain
    for (int i = pad_top; i < cells_i; i++)
    {
      for (int j = pad_left; j < cells_j; j++)
      {
        x[i][j] = vld1_f32(x_ptrs[i][j]);
        x_ptrs[i][j] += 2;
      }
    }

    // Compute XT . x
    for (int j = pad_left; j < cells_j; j++)
    {
      // XTx[0][j] =  4*x[0][j] + -5*x[2][j] +  1*x[4][j];
      XTx[0][j] = vmls_n_f32(vmla_n_f32(x[4][j], x[0][j], 4.0f), x[2][j], 5.0f);

      // XTx[1][j] = -4*x[1][j] + -4*x[2][j] +  1*x[3][j] +  1*x[4][j];
      XTx[1][j] = vmls_n_f32(vadd_f32(x[3][j], x[4][j]), vadd_f32(x[1][j], x[2][j]), 4.0f);

      // XTx[2][j] =  4*x[1][j] + -4*x[2][j] + -1*x[3][j] +  1*x[4][j];
      XTx[2][j] = vmla_n_f32(vsub_f32(x[4][j], x[3][j]), vsub_f32(x[1][j], x[2][j]), 4.0f);

      // XTx[3][j] = -2*x[1][j] + -1*x[2][j] +  2*x[3][j] +  1*x[4][j];
      XTx[3][j] = vmla_n_f32(vsub_f32(x[4][j], x[2][j]), vsub_f32(x[3][j], x[1][j]), 2.0f);

      // XTx[4][j] =  2*x[1][j] + -1*x[2][j] + -2*x[3][j] +  1*x[4][j];
      XTx[4][j] = vmla_n_f32(vsub_f32(x[4][j], x[2][j]), vsub_f32(x[1][j], x[3][j]), 2.0f);

      // XTx[5][j] =  4*x[1][j] + -5*x[3][j] +  1*x[5][j];
      XTx[5][j] = vmls_n_f32(vmla_n_f32(x[5][j], x[1][j], 4.0f), x[3][j], 5.0f);
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_rows; i++)
    {
      // U[i][0] =  4*XTx[i][0] + -5*XTx[i][2] +  1*XTx[i][4];
      U[i][0] = vmls_n_f32(vmla_n_f32(XTx[i][4], XTx[i][0], 4.0f), XTx[i][2], 5.0f);

      // U[i][1] = -4*XTx[i][1] + -4*XTx[i][2] +  1*XTx[i][3] +  1*XTx[i][4];
      U[i][1] = vmls_n_f32(vadd_f32(XTx[i][3], XTx[i][4]), vadd_f32(XTx[i][1], XTx[i][2]), 4.0f);

      // U[i][2] =  4*XTx[i][1] + -4*XTx[i][2] + -1*XTx[i][3] +  1*XTx[i][4];
      U[i][2] = vmla_n_f32(vsub_f32(XTx[i][4], XTx[i][3]), vsub_f32(XTx[i][1], XTx[i][2]), 4.0f);

      // U[i][3] = -2*XTx[i][1] + -1*XTx[i][2] +  2*XTx[i][3] +  1*XTx[i][4];
      U[i][3] = vmla_n_f32(vsub_f32(XTx[i][4], XTx[i][2]), vsub_f32(XTx[i][3], XTx[i][1]), 2.0f);

      // U[i][4] =  2*XTx[i][1] + -1*XTx[i][2] + -2*XTx[i][3] +  1*XTx[i][4];
      U[i][4] = vmla_n_f32(vsub_f32(XTx[i][4], XTx[i][2]), vsub_f32(XTx[i][1], XTx[i][3]), 2.0f);

      // U[i][5] =  4*XTx[i][1] + -5*XTx[i][3] +  1*XTx[i][5];
      U[i][5] = vmls_n_f32(vmla_n_f32(XTx[i][5], XTx[i][1], 4.0f), XTx[i][3], 5.0f);
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++, m++)
      {
        vst1_f32(outptr + m*matrix_stride, U[i][j]);
      }
    }
    outptr += 2;
  }
#endif  // __arm_any__
  for (; channels_remaining; channels_remaining--)
  {
    // Load x
    for (int i = pad_top; i < cells_i; i++)
    {
      for (int j = pad_left; j < cells_j; j++)
      {
        x[i][j] = *(x_ptrs[i][j]++);
      }
    }

    // Compute XT . x
    for (int j = pad_left; j < cells_j; j++)
    {
      XTx[0][j] =  4*x[0][j] + -5*x[2][j] +  1*x[4][j];
      XTx[1][j] = -4*x[1][j] + -4*x[2][j] +  1*x[3][j] +  1*x[4][j];
      XTx[2][j] =  4*x[1][j] + -4*x[2][j] + -1*x[3][j] +  1*x[4][j];
      XTx[3][j] = -2*x[1][j] + -1*x[2][j] +  2*x[3][j] +  1*x[4][j];
      XTx[4][j] =  2*x[1][j] + -1*x[2][j] + -2*x[3][j] +  1*x[4][j];
      XTx[5][j] =  4*x[1][j] + -5*x[3][j] +  1*x[5][j];
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_rows; i++)
    {
      U[i][0] =  4*XTx[i][0] + -5*XTx[i][2] +  1*XTx[i][4];
      U[i][1] = -4*XTx[i][1] + -4*XTx[i][2] +  1*XTx[i][3] +  1*XTx[i][4];
      U[i][2] =  4*XTx[i][1] + -4*XTx[i][2] + -1*XTx[i][3] +  1*XTx[i][4];
      U[i][3] = -2*XTx[i][1] + -1*XTx[i][2] +  2*XTx[i][3] +  1*XTx[i][4];
      U[i][4] =  2*XTx[i][1] + -1*XTx[i][2] + -2*XTx[i][3] +  1*XTx[i][4];
      U[i][5] =  4*XTx[i][1] + -5*XTx[i][3] +  1*XTx[i][5];
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++, m++)
      {
        *(outptr + m*matrix_stride) = U[i][j];
      }
    }
    outptr++;
  }
}
}

namespace winograd
{
template <int k>
using Tiles = InputTransformImplTiles<k, k, 6, 6, float>;

template <>
const Tiles<3>::TileFn Tiles<3>::tilefn_generic = winograd_input_transform_6x6_fp32_process_tile<false>;

template <>
const Tiles<3>::TileFn Tiles<3>::tilefn_unpadded = winograd_input_transform_6x6_fp32_process_tile<true>;

template <>
const Tiles<3>::TileFn Tiles<3>::tilefn_top_padded[n_pad_top] = {
  winograd_input_transform_6x6_fp32_process_tile<true, 1, 0, 0, 0>,
};

template <>
const Tiles<3>::TileFn Tiles<3>::tilefn_left_padded[n_pad_left] = {
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 1, 0, 0>,
};

template <>
const Tiles<3>::TileFn Tiles<3>::tilefn_bottom_padded[n_pad_bottom] = {
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 1, 0>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 2, 0>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 3, 0>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 4, 0>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 5, 0>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 6, 0>,
};

template <>
const Tiles<3>::TileFn Tiles<3>::tilefn_right_padded[n_pad_right] = {
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 1>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 2>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 3>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 4>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 5>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 6>,
};

template <>
const Tiles<5>::TileFn Tiles<5>::tilefn_generic = winograd_input_transform_6x6_fp32_process_tile<false>;


template <>
const Tiles<5>::TileFn Tiles<5>::tilefn_unpadded = winograd_input_transform_6x6_fp32_process_tile<true>;


template <>
const Tiles<5>::TileFn Tiles<5>::tilefn_top_padded[n_pad_top] = {
  winograd_input_transform_6x6_fp32_process_tile<true, 2, 0, 0, 0>,
};

template <>
const Tiles<5>::TileFn Tiles<5>::tilefn_left_padded[n_pad_left] = {
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 2, 0, 0>,
};

template <>
const Tiles<5>::TileFn Tiles<5>::tilefn_bottom_padded[n_pad_bottom] = {
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 1, 0>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 2, 0>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 3, 0>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 4, 0>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 5, 0>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 6, 0>,
};

template <>
const Tiles<5>::TileFn Tiles<5>::tilefn_right_padded[n_pad_right] = {
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 1>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 2>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 3>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 4>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 5>,
  winograd_input_transform_6x6_fp32_process_tile<true, 0, 0, 0, 6>,
};

template class InputTransform<3, 3, 6, 6, float>;
template class InputTransform<5, 5, 6, 6, float>;
}
