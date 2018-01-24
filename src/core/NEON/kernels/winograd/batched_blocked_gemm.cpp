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

#include "batched_blocked_gemm.hpp"
#include "gemm.hpp"
using namespace winograd;

template <const int MB, const int NB, typename TIn, typename TOut>
BatchedBlockedGemm<MB, NB, TIn, TOut>::BatchedBlockedGemm(
  const unsigned int n_gemms,
  const int M, const int K, const int N,
  const int a_matrix_stride,
  const int a_row_stride,
  const int b_matrix_stride,
  const int b_row_stride,
  const int c_matrix_stride,
  const int c_row_stride,
  const TIn* const a_ptr,
  const TIn* const b_ptr,
  TOut* const c_ptr
) : n_gemms(n_gemms), M(M), N(N), K(K),
    a_matrix_stride(a_matrix_stride),
    a_row_stride(a_row_stride),
    b_matrix_stride(b_matrix_stride),
    b_row_stride(b_row_stride),
    c_matrix_stride(c_matrix_stride),
    c_row_stride(c_row_stride),
    a_ptr(a_ptr), b_ptr(b_ptr), c_ptr(c_ptr)
{
}

template <const int MBlock, const int NBlock, typename TIn, typename TOut>
unsigned int BatchedBlockedGemm<MBlock, NBlock, TIn, TOut>::get_window() const
{
  return n_gemms;
}

template <const int MBlock, const int NBlock, typename TIn, typename TOut>
void BatchedBlockedGemm<MBlock, NBlock, TIn, TOut>::run(
  const unsigned int start, const unsigned int stop
)
{
  // Perform the specified GEMMs
  for (unsigned int i = start; i < stop; i++)
  {
    // Get pointers to the relevant matrices
    const TIn* const mtr_a = a_ptr + i*a_matrix_stride;
    const TIn* const mtr_b = b_ptr + i*b_matrix_stride;
    TOut* const mtr_c = c_ptr + i*c_matrix_stride;

    // Perform the GEMM
    BlockedGemm<MBlock, NBlock, TIn, TOut>(
      mtr_a, mtr_b, mtr_c, M, K, N,
      a_row_stride, b_row_stride, c_row_stride
    );
  }
}

template class winograd::BatchedBlockedGemm<4, 16, float, float>;

