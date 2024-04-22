/*
 * Copyright (c) 2022 Arm Limited.
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

#include <cstddef>

namespace arm_conv {
namespace winograd {
namespace weight_transform {

void cpp_fp32_1x4_1x5(
  unsigned int n_channels,
  const float *inptr,
  size_t,  // ld_weight_row
  size_t ld_weight_col,
  float *outptr,
  size_t matrix_stride
)
{
  constexpr auto kernel_cols = 5u, inner_tile_cols = 8u;

  // For each output channel
  for (; n_channels; n_channels--)
  {
    // Matrices used and computed in this kernel
    float w[kernel_cols], V[inner_tile_cols];

    // Read weights
    for (auto j = 0u; j < kernel_cols; j++)
    {
      w[j] = *(inptr + j * ld_weight_col);
    }

    // Compute V = w WT
    V[0] = (w[0]*-1) / 36;
    V[1] = (w[1]*-1 + w[3]*-1 + w[0]*1 + w[2]*1 + w[4]*1) / 48;
    V[2] = (w[0]*1 + w[1]*1 + w[2]*1 + w[3]*1 + w[4]*1) / 48;
    V[3] = (w[0]*-1 + w[4]*-16 + w[2]*-4 + w[1]*2 + w[3]*8) / 120;
    V[4] = (w[0]*-1 + w[4]*-16 + w[3]*-8 + w[2]*-4 + w[1]*-2) / 120;
    V[5] = (w[3]*-27 + w[1]*-3 + w[2]*9 + w[4]*81 + w[0]*1) / 720;
    V[6] = (w[1]*3 + w[2]*9 + w[3]*27 + w[4]*81 + w[0]*1) / 720;
    V[7] = (w[4]*1) / 1;

    // Store the transformed weights
    for (auto  j = 0u; j < inner_tile_cols; j++)
    {
      *(outptr + j*matrix_stride) = V[j];
    }

    inptr++;
    outptr++;
  }
}

}  // namespace weight_transform
}  // namespace winograd
}  // namespace arm_conv
