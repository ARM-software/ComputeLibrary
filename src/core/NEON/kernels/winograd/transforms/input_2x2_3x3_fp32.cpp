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

#include "transforms/input.hpp"
#include "winograd_gemm.hpp"
#include "arm.hpp"

namespace winograd
{

using Transform = WinogradGEMM<2, 2, 3, 3>::InputTransform<float>;

/******************************************************************************
 * Cost methods for the input transform.
 * =====================================
 */
template <>
template <>
int Transform::ops_performed(const Tensor4DShape &input_shape)
{
  // NOTE: Cost in FLOPs rather than instructions or uops.
  const int tile_M = iceildiv(input_shape.n_rows, inner_tile_rows);
  const int tile_N = iceildiv(input_shape.n_cols, inner_tile_cols);
  return 16 * 16 * tile_M * tile_N * input_shape.n_channels;
}
/*****************************************************************************/

/*****************************************************************************
* F(2x2, 3x3) implies the use of a 4x4 input tile. Such tiles can require a
* variety of padding types. For example, tiles at the top and left of an image
* can require one row or column of padding on their top and left sides if the
* padding type is SAME (where X represents a padded value):
*
*      _______    _______
*     |X X X X|  |X X X X|
*     |X      |  |       |   . . .
*     |X      |  |       |
*     |X______|  |_______|
*      _______
*     |X      |             .
*     |X      |   . . .       .
*     |X      |                 .
*     |X______|
*
* For tiles near the right or bottom of the image it is more complicated.  Such
* tiles might require padding by 0 or 1 rows or columns if the padding type is
* VALID or 1 or 2 rows or columns if the padding type is SAME:
*
*      _______    _______    _______    _______
*     |X X X X|  |X X X X|  |X X X X|  |X X X X|
*     |X      |  |       |  |      X|  |    X X|
*     |X      |  |       |  |      X|  |    X X|
*     |X______|  |_______|  |______X|  |____X_X|
*      _______    _______    _______    _______
*     |X      |  |       |  |      X|  |    X X|
*     |X      |  |       |  |      X|  |    X X|
*     |X      |  |       |  |      X|  |    X X|
*     |X______|  |_______|  |______X|  |____X_X|
*      _______    _______    _______    _______
*     |X      |  |       |  |      X|  |    X X|
*     |X      |  |       |  |      X|  |    X X|
*     |X      |  |       |  |      X|  |    X X|
*     |X_X_X_X|  |X_X_X_X|  |X_X_X_X|  |X_X_X_X|
*      _______    _______    _______    _______
*     |X      |  |       |  |      X|  |    X X|
*     |X      |  |       |  |      X|  |    X X|
*     |X X X X|  |X X X X|  |X X X X|  |X X X X|
*     |X_X_X_X|  |X_X_X_X|  |X_X_X_X|  |X_X_X_X|
*
* Additional tiles are required for especially small input images.
*
* Build an array of the specialised methods that deal with each of the
* different padding combinations which may be required. These padding
* constraints are the space:
*
*     Padding top in {0, 1}
*     Padding left in {0, 1}
*     Padding bottom in {0, 1, 2}
*     Padding right in {0, 1, 2}
*/
template <>
template <>
template <int pad_top, int pad_left, int pad_bottom, int pad_right>
void Transform::process_tile(
  int n_channels,
  const float* const input_base,
  const int input_row_stride,
  const int input_col_stride,
  float* const matrix_base,
  const int matrix_stride
)
{
  constexpr int inner_tile_i = 4, inner_tile_j = 4;
  constexpr int cells_i = inner_tile_i - pad_bottom;
  constexpr int cells_j = inner_tile_i - pad_right;

  float *outptr = matrix_base;

  // Get pointers into the input tile
  const float *x_ptrs[inner_tile_i][inner_tile_j];
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
  float x[inner_tile_i][inner_tile_j];
  float XTx[inner_tile_i][inner_tile_j];
  float U[inner_tile_i][inner_tile_j];

  for (int i = 0; i < inner_tile_i; i++)
  {
    for (int j = 0; j < inner_tile_j; j++)
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
    // Matrices used/computed in this kernel.
    float32x4_t x[inner_tile_i][inner_tile_j];
    float32x4_t XTx[inner_tile_i][inner_tile_j];
    float32x4_t U[inner_tile_i][inner_tile_j];

    for (int i = 0; i < inner_tile_i; i++)
    {
      for (int j = 0; j < inner_tile_j; j++)
      {
        x[i][j] = vdupq_n_f32(0.0f);
        XTx[i][j] = vdupq_n_f32(0.0f);
      }
    }

    // Load x
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
      // XTx[0][j] = x[0][j] - x[2][j];
      XTx[0][j] = vsubq_f32(x[0][j], x[2][j]);

      // XTx[1][j] = x[1][j] + x[2][j];
      XTx[1][j] = vaddq_f32(x[1][j], x[2][j]);

      // XTx[2][j] = x[2][j] - x[1][j];
      XTx[2][j] = vsubq_f32(x[2][j], x[1][j]);

      // XTx[3][j] = x[1][j] - x[3][j];
      XTx[3][j] = vsubq_f32(x[1][j], x[3][j]);
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_i; i++)
    {
      // U[i][0] = XTx[i][0] - XTx[i][2];
      U[i][0] = vsubq_f32(XTx[i][0], XTx[i][2]);

      // U[i][1] = XTx[i][1] + XTx[i][2];
      U[i][1] = vaddq_f32(XTx[i][1], XTx[i][2]);

      // U[i][2] = XTx[i][2] - XTx[i][1];
      U[i][2] = vsubq_f32(XTx[i][2], XTx[i][1]);

      // U[i][3] = XTx[i][1] - XTx[i][3];
      U[i][3] = vsubq_f32(XTx[i][1], XTx[i][3]);
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < inner_tile_i; i++)
    {
      for (int j = 0; j < inner_tile_j; j++, m++)
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
    // Matrices used/computed in this kernel.
    float32x2_t x[inner_tile_i][inner_tile_j];
    float32x2_t XTx[inner_tile_i][inner_tile_j];
    float32x2_t U[inner_tile_i][inner_tile_j];

    for (int i = 0; i < inner_tile_i; i++)
    {
      for (int j = 0; j < inner_tile_j; j++)
      {
        x[i][j] = vdup_n_f32(0.0f);
        XTx[i][j] = vdup_n_f32(0.0f);
      }
    }

    // Load x
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
      // XTx[0][j] = x[0][j] - x[2][j];
      XTx[0][j] = vsub_f32(x[0][j], x[2][j]);

      // XTx[1][j] = x[1][j] + x[2][j];
      XTx[1][j] = vadd_f32(x[1][j], x[2][j]);

      // XTx[2][j] = x[2][j] - x[1][j];
      XTx[2][j] = vsub_f32(x[2][j], x[1][j]);

      // XTx[3][j] = x[1][j] - x[3][j];
      XTx[3][j] = vsub_f32(x[1][j], x[3][j]);
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_i; i++)
    {
      // U[i][0] = XTx[i][0] - XTx[i][2];
      U[i][0] = vsub_f32(XTx[i][0], XTx[i][2]);

      // U[i][1] = XTx[i][1] + XTx[i][2];
      U[i][1] = vadd_f32(XTx[i][1], XTx[i][2]);

      // U[i][2] = XTx[i][2] - XTx[i][1];
      U[i][2] = vsub_f32(XTx[i][2], XTx[i][1]);

      // U[i][3] = XTx[i][1] - XTx[i][3];
      U[i][3] = vsub_f32(XTx[i][1], XTx[i][3]);
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < inner_tile_i; i++)
    {
      for (int j = 0; j < inner_tile_j; j++, m++)
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
      XTx[0][j] = x[0][j] - x[2][j];
      XTx[1][j] = x[1][j] + x[2][j];
      XTx[2][j] = x[2][j] - x[1][j];
      XTx[3][j] = x[1][j] - x[3][j];
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_i; i++)
    {
      U[i][0] = XTx[i][0] - XTx[i][2];
      U[i][1] = XTx[i][1] + XTx[i][2];
      U[i][2] = XTx[i][2] - XTx[i][1];
      U[i][3] = XTx[i][1] - XTx[i][3];
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < inner_tile_i; i++)
    {
      for (int j = 0; j < inner_tile_j; j++, m++)
      {
        *(outptr + m*matrix_stride) = U[i][j];
      }
    }
    outptr++;
  }
}

template <>
template <>
const Transform::TileFn Transform::tile_fns[2][2][max_pad_bottom][max_pad_right] =
{
  {
    {
      {
        Transform::template process_tile<0, 0, 0, 0>,  // No padding
        Transform::template process_tile<0, 0, 0, 1>,  // Right
        Transform::template process_tile<0, 0, 0, 2>,  // Right
      },
      {
        Transform::template process_tile<0, 0, 1, 0>,  // Bottom
        Transform::template process_tile<0, 0, 1, 1>,  // Bottom-right
        Transform::template process_tile<0, 0, 1, 2>,  // Bottom-right
      },
      {
        Transform::template process_tile<0, 0, 2, 0>,  // Bottom
        Transform::template process_tile<0, 0, 2, 1>,  // Bottom-right
        Transform::template process_tile<0, 0, 2, 2>,  // Bottom-right
      }
    },
    {
      {
        Transform::template process_tile<0, 1, 0, 0>,  // Left
        Transform::template process_tile<0, 1, 0, 1>,  // Left AND right
        Transform::template process_tile<0, 1, 0, 2>,  // Left AND right
      },
      {
        Transform::template process_tile<0, 1, 1, 0>,  // Left-bottom
        Transform::template process_tile<0, 1, 1, 1>,  // Left, bottom AND right
        Transform::template process_tile<0, 1, 1, 2>,  // Left, bottom AND right
      },
      {
        Transform::template process_tile<0, 1, 2, 0>,  // Left-bottom
        Transform::template process_tile<0, 1, 2, 1>,  // Left, bottom AND right
        Transform::template process_tile<0, 1, 2, 2>,  // Left, bottom AND right
      }
    },
  },
  {
    {
      {
        Transform::template process_tile<1, 0, 0, 0>,  // Top
        Transform::template process_tile<1, 0, 0, 1>,  // Top-right
        Transform::template process_tile<1, 0, 0, 2>,  // Top-right
      },
      {
        Transform::template process_tile<1, 0, 1, 0>,  // Top AND bottom
        Transform::template process_tile<1, 0, 1, 1>,  // Top, bottom AND right
        Transform::template process_tile<1, 0, 1, 2>,  // Top, bottom AND right
      },
      {
        Transform::template process_tile<1, 0, 2, 0>,  // Top AND bottom
        Transform::template process_tile<1, 0, 2, 1>,  // Top, bottom AND right
        Transform::template process_tile<1, 0, 2, 2>,  // Top, bottom AND right
      }
    },
    {
      {
        Transform::template process_tile<1, 1, 0, 0>,  // Top-left
        Transform::template process_tile<1, 1, 0, 1>,  // Top, left AND right
        Transform::template process_tile<1, 1, 0, 2>,  // Top, left AND right
      },
      {
        Transform::template process_tile<1, 1, 1, 0>,  // Top, left AND bottom
        Transform::template process_tile<1, 1, 1, 1>,  // All padded
        Transform::template process_tile<1, 1, 1, 2>,  // All padded
      },
      {
        Transform::template process_tile<1, 1, 2, 0>,  // Top, left AND bottom
        Transform::template process_tile<1, 1, 2, 1>,  // All padded
        Transform::template process_tile<1, 1, 2, 2>,  // All padded
      }
    }
  }
};

template struct WinogradGEMM<2, 2, 3, 3>::InputTransform<float>;
}  // namespace winograd
