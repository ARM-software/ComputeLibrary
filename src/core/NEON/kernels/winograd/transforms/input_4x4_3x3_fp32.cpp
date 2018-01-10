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

using Transform = WinogradGEMM<4, 4, 3, 3>::InputTransform<float>;

template <>
template <>
int Transform::ops_performed(const Tensor4DShape &input_shape)
{
  // NOTE: Cost in FLOPs rather than instructions or uops.
  const int tile_M = iceildiv(input_shape.n_rows, inner_tile_rows);
  const int tile_N = iceildiv(input_shape.n_cols, inner_tile_cols);
  return 12 * 24 * tile_M * tile_N * input_shape.n_channels;
}

/* F(4x4, 3x3) implies the use of a 6x6 input tile. Such tiles can require a
* variety of padding types. For example, tiles at the top and left of an
* image can require one row or column of padding on their top and left sides
* if the padding type is SAME (where X represents a padded value):
*
*      ___________    ___________
*     |X X X X X X|  |X X X X X X|
*     |X          |  |           |
*     |X          |  |           |
*     |X          |  |           |
*     |X          |  |           |
*     |X__________|  |___________|
*      ___________
*     |X          |
*     |X          |
*     |X          |
*     |X          |
*     |X          |
*     |X__________|
*
* For tiles near the right or bottom of the image it is more complicated.
* Such tiles might require padding by 0, 1, 2 or 3 rows or columns if the
* padding type is VALID or 1, 2, 3 or 4 rows or columns if the padding
* type is SAME.
*
* Build an array of the specialised methods that deal with each of the
* different padding combinations which may be required. These padding
* constraints are the space:
*
*     Padding top in {0, 1}
*     Padding left in {0, 1}
*     Padding bottom in {0, 1, 2, 3, 4}
*     Padding right in {0, 1, 2, 3, 4}
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
  constexpr int cells_i = 6 - pad_bottom;
  constexpr int cells_j = 6 - pad_right;

  float *outptr = matrix_base;

  // Get pointers into the input tile
  const float *x_ptrs[6][6];
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
  float x[6][6], XTx[6][6], U[6][6];
  for (int i = 0; i < 6; i++)
  {
    for (int j = 0; j < 6; j++)
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
    float32x4_t x[6][6], XTx[6][6], U[6][6];
    for (int i = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++)
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
    for (int i = 0; i < 6; i++)
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
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
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
    float32x2_t x[6][6], XTx[6][6], U[6][6];
    for (int i = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++)
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
    for (int i = 0; i < 6; i++)
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
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
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
    for (int i = 0; i < 6; i++)
    {
      U[i][0] =  4*XTx[i][0] + -5*XTx[i][2] +  1*XTx[i][4];
      U[i][1] = -4*XTx[i][1] + -4*XTx[i][2] +  1*XTx[i][3] +  1*XTx[i][4];
      U[i][2] =  4*XTx[i][1] + -4*XTx[i][2] + -1*XTx[i][3] +  1*XTx[i][4];
      U[i][3] = -2*XTx[i][1] + -1*XTx[i][2] +  2*XTx[i][3] +  1*XTx[i][4];
      U[i][4] =  2*XTx[i][1] + -1*XTx[i][2] + -2*XTx[i][3] +  1*XTx[i][4];
      U[i][5] =  4*XTx[i][1] + -5*XTx[i][3] +  1*XTx[i][5];
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
      {
        *(outptr + m*matrix_stride) = U[i][j];
      }
    }
    outptr++;
  }
}

/* In the below, unusual or especially small tiles are routed via the slow
 * path whereas common or large tiles are routed through a faster path.
 */
template <>
template <>
const Transform::TileFn Transform::tile_fns[2][2][max_pad_bottom][max_pad_right] =
{
  {
    {
      {
        Transform::template process_tile<0, 0, 0, 0>,  // No padding
        Transform::template process_tile<0, 0, 0, 1>,  // Right
        Transform::template process_tile<0, 0, 0, 2>,  // "   "
        Transform::template process_tile<0, 0, 0, 3>,  // "   "
        Transform::template process_tile<0, 0, 0, 4>,  // "   "
      },
      {
        Transform::template process_tile<0, 0, 1, 0>,  // Bottom
        Transform::template process_tile<0, 0, 1, 1>,  // Bottom right
        Transform::template process_tile<0, 0, 1, 2>,  // "          "
        Transform::template process_tile<0, 0, 1, 3>,  // "          "
        Transform::template process_tile<0, 0, 1, 4>,  // "          "
      },
      {
        Transform::template process_tile<0, 0, 2, 0>,  // Bottom
        Transform::template process_tile<0, 0, 2, 1>,  // Bottom right
        Transform::template process_tile<0, 0, 2, 2>,  // "          "
        Transform::template process_tile<0, 0, 2, 3>,  // "          "
        Transform::template process_tile<0, 0, 2, 4>,  // "          "
      },
      {
        Transform::template process_tile<0, 0, 3, 0>,  // Bottom
        Transform::template process_tile<0, 0, 3, 1>,  // Bottom right
        Transform::template process_tile<0, 0, 3, 2>,  // "          "
        Transform::template process_tile<0, 0, 3, 3>,  // "          "
        Transform::template process_tile<0, 0, 3, 4>,  // "          "
      },
      {
        Transform::template process_tile<0, 0, 4, 0>,  // Bottom
        Transform::template process_tile<0, 0, 4, 1>,  // Bottom right
        Transform::template process_tile<0, 0, 4, 2>,  // "          "
        Transform::template process_tile<0, 0, 4, 3>,  // "          "
        Transform::template process_tile<0, 0, 4, 4>,  // "          "
      }
    },
    {
      {
        Transform::template process_tile<0, 1, 0, 0>,  // Left
        Transform::template process_tile<0, 1, 0, 1>,
        Transform::template process_tile<0, 1, 0, 2>,
        Transform::template process_tile<0, 1, 0, 3>,
        Transform::template process_tile<0, 1, 0, 4>,
      },
      {
        Transform::template process_tile<0, 1, 1, 0>,  // Bottom left
        Transform::template process_tile<0, 1, 1, 1>,
        Transform::template process_tile<0, 1, 1, 2>,
        Transform::template process_tile<0, 1, 1, 3>,
        Transform::template process_tile<0, 1, 1, 4>,
      },
      {
        Transform::template process_tile<0, 1, 2, 0>,  // "          "
        Transform::template process_tile<0, 1, 2, 1>,
        Transform::template process_tile<0, 1, 2, 2>,
        Transform::template process_tile<0, 1, 2, 3>,
        Transform::template process_tile<0, 1, 2, 4>,
      },
      {
        Transform::template process_tile<0, 1, 3, 0>,  // "          "
        Transform::template process_tile<0, 1, 3, 1>,
        Transform::template process_tile<0, 1, 3, 2>,
        Transform::template process_tile<0, 1, 3, 3>,
        Transform::template process_tile<0, 1, 3, 4>,
      },
      {
        Transform::template process_tile<0, 1, 4, 0>,  // "          "
        Transform::template process_tile<0, 1, 4, 1>,
        Transform::template process_tile<0, 1, 4, 2>,
        Transform::template process_tile<0, 1, 4, 3>,
        Transform::template process_tile<0, 1, 4, 4>,
      }
    }
  },
  {
    {
      {
        Transform::template process_tile<1, 0, 0, 0>,  // Top
        Transform::template process_tile<1, 0, 0, 1>,  // Top right
        Transform::template process_tile<1, 0, 0, 2>,  // "       "
        Transform::template process_tile<1, 0, 0, 3>,  // "       "
        Transform::template process_tile<1, 0, 0, 4>,  // "       "
      },
      {
        Transform::template process_tile<1, 0, 1, 0>,
        Transform::template process_tile<1, 0, 1, 1>,
        Transform::template process_tile<1, 0, 1, 2>,
        Transform::template process_tile<1, 0, 1, 3>,
        Transform::template process_tile<1, 0, 1, 4>,
      },
      {
        Transform::template process_tile<1, 0, 2, 0>,
        Transform::template process_tile<1, 0, 2, 1>,
        Transform::template process_tile<1, 0, 2, 2>,
        Transform::template process_tile<1, 0, 2, 3>,
        Transform::template process_tile<1, 0, 2, 4>,
      },
      {
        Transform::template process_tile<1, 0, 3, 0>,
        Transform::template process_tile<1, 0, 3, 1>,
        Transform::template process_tile<1, 0, 3, 2>,
        Transform::template process_tile<1, 0, 3, 3>,
        Transform::template process_tile<1, 0, 3, 4>,
      },
      {
        Transform::template process_tile<1, 0, 4, 0>,
        Transform::template process_tile<1, 0, 4, 1>,
        Transform::template process_tile<1, 0, 4, 2>,
        Transform::template process_tile<1, 0, 4, 3>,
        Transform::template process_tile<1, 0, 4, 4>,
      },
    },
    {
      {
        Transform::template process_tile<1, 1, 0, 0>,  // Top left
        Transform::template process_tile<1, 1, 0, 1>,
        Transform::template process_tile<1, 1, 0, 2>,
        Transform::template process_tile<1, 1, 0, 3>,
        Transform::template process_tile<1, 1, 0, 4>,
      },
      {
        Transform::template process_tile<1, 1, 1, 0>,
        Transform::template process_tile<1, 1, 1, 1>,
        Transform::template process_tile<1, 1, 1, 2>,
        Transform::template process_tile<1, 1, 1, 3>,
        Transform::template process_tile<1, 1, 1, 4>,
      },
      {
        Transform::template process_tile<1, 1, 2, 0>,
        Transform::template process_tile<1, 1, 2, 1>,
        Transform::template process_tile<1, 1, 2, 2>,
        Transform::template process_tile<1, 1, 2, 3>,
        Transform::template process_tile<1, 1, 2, 4>,
      },
      {
        Transform::template process_tile<1, 1, 3, 0>,
        Transform::template process_tile<1, 1, 3, 1>,
        Transform::template process_tile<1, 1, 3, 2>,
        Transform::template process_tile<1, 1, 3, 3>,
        Transform::template process_tile<1, 1, 3, 4>,
      },
      {
        Transform::template process_tile<1, 1, 4, 0>,
        Transform::template process_tile<1, 1, 4, 1>,
        Transform::template process_tile<1, 1, 4, 2>,
        Transform::template process_tile<1, 1, 4, 3>,
        Transform::template process_tile<1, 1, 4, 4>,
      }
    }
  }
};

template struct WinogradGEMM<4, 4, 3, 3>::InputTransform<float>;
}  // namespace winograd
