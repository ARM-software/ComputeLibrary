/*
 * Copyright (c) 2019 ARM Limited.
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

#include "arm.hpp"
#include "kernel.hpp"

namespace winograd
{

template <>
void WeightTransform<1, 7, 1, 8, float, float, WinogradRoots::Integers>::execute(
  const int n_output_channels,
  const int n_input_channels,
  const float* const input,  // NOTE: Data in HWIO order
  float* const output,
  const int matrix_stride,
  const int matrix_row_stride
)
{
  // Get pointers to each cell of the weight tensor
  const auto weight_col_stride = n_input_channels * n_output_channels;
  const float *inptrs[kernel_cols];
  for (int j = 0; j < kernel_cols; j++)
  {
    inptrs[j] = input + j*weight_col_stride;
  }

  // For each input channel
  for (int ic = 0; ic < n_input_channels; ic++)
  {
    float *outptr = output + ic * matrix_row_stride;

    // For each output channel
    int channels_remaining = n_output_channels;
    for (; channels_remaining; channels_remaining--)
    {
      // Matrices used and computed in this kernel
      float w[kernel_cols], V[inner_tile_cols];

      // Read weights
      for (int j = 0; j < kernel_cols; j++)
      {
        w[j] = *(inptrs[j]++);
      }

      // Compute V = w WT
      V[0] = (w[0]*-1) / 36.0f;
      V[1] = (w[1]*-1 + w[3]*-1 + w[5]*-1 + w[0]*1 + w[2]*1 + w[4]*1 + w[6]*1) / 48.0f;
      V[2] = (w[0]*1 + w[1]*1 + w[2]*1 + w[3]*1 + w[4]*1 + w[5]*1 + w[6]*1) / 48.0f;
      V[3] = (w[0]*-1 + w[6]*-64 + w[4]*-16 + w[2]*-4 + w[1]*2 + w[3]*8 + w[5]*32) / 120.0f;
      V[4] = (w[0]*-1 + w[6]*-64 + w[5]*-32 + w[4]*-16 + w[3]*-8 + w[2]*-4 + w[1]*-2) / 120.0f;
      V[5] = (w[5]*-243 + w[3]*-27 + w[1]*-3 + w[2]*9 + w[4]*81 + w[6]*729 + w[0]*1) / 720.0f;
      V[6] = (w[1]*3 + w[2]*9 + w[3]*27 + w[4]*81 + w[5]*243 + w[6]*729 + w[0]*1) / 720.0f;
      V[7] = (w[6]*1) / 1.0f;

      // Store the transformed weights
      for (int j = 0; j < inner_tile_cols; j++)
      {
        *(outptr + j*matrix_stride) = V[j];
      }
      outptr++;
    }
  }
}

template class WeightTransform<1, 7, 1, 8, float, float, WinogradRoots::Integers>;
template class WeightTransform<7, 1, 8, 1, float, float, WinogradRoots::Integers>;

}  // namespace winograd
