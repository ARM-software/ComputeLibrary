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

void cpp_fp32_1x6_1x3(
  unsigned int n_channels,
  const float *inptr, size_t, size_t ld_weight_col,
  float *outptr, size_t matrix_stride
)
{
  for (; n_channels; n_channels--)
  {
    // Matrices used and computed in this kernel
    float w[3], V[8];

    // Read weights
    for (int j = 0; j < 3; j++)
    {
      w[j] = *(inptr + j * ld_weight_col);
    }

    // Compute V = w WT
    V[0] = (w[0]*-1) / 36.0f;
    V[1] = (w[1]*-1 + w[0]*1 + w[2]*1) / 48.0f;
    V[2] = (w[0]*1 + w[1]*1 + w[2]*1) / 48.0f;
    V[3] = (w[0]*-1 + w[2]*-4 + w[1]*2) / 120.0f;
    V[4] = (w[0]*-1 + w[2]*-4 + w[1]*-2) / 120.0f;
    V[5] = (w[1]*-3 + w[2]*9 + w[0]*1) / 720.0f;
    V[6] = (w[1]*3 + w[2]*9 + w[0]*1) / 720.0f;
    V[7] = (w[2]*1) / 1;

    // Store the transformed weights
    for (int j = 0; j < 8; j++)
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
