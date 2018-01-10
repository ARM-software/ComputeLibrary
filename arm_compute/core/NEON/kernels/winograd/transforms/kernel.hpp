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

#include "winograd_gemm.hpp"
using namespace winograd;


template <int otr, int otc, int kr, int kc>
template <typename T>
WinogradGEMM<otr, otc, kr, kc>::WeightsTransform<T>::WeightsTransform(
  const T* const input,
  T* const output,
  const int matrix_stride,      /** Stride across matrices in the output. */
  const int matrix_row_stride,  /** Stride across rows of the matrix. */
  const int n_output_channels,
  const int n_input_channels
) : inptr(input), outptr(output),
    matrix_stride(matrix_stride), matrix_row_stride(matrix_row_stride),
    n_output_channels(n_output_channels), n_input_channels(n_input_channels)
{
}


template <int otr, int otc, int kr, int kc>
template <typename T>
unsigned int WinogradGEMM<otr, otc, kr, kc>::WeightsTransform<T>::get_window() const
{
  // TODO When the weights transform supports multithreading, return the number
  // of output channels. For now we return 1 to indicate that the weights must
  // be transformed as a single block.
  // return n_output_channels;
  return 1;
}


template <int otr, int otc, int kr, int kc>
template <typename T>
void WinogradGEMM<otr, otc, kr, kc>::WeightsTransform<T>::run(
  const unsigned int start, const unsigned int stop
)
{
  // TODO When the weights transform supports multithreading call execute for a
  // portion of the output channels.
  (void) start;
  (void) stop;

  // For now, just do all of the work.
  execute(
    n_output_channels,
    n_input_channels,
    inptr,
    outptr,
    matrix_stride,
    matrix_row_stride
  );
}
