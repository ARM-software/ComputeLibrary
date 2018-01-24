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

#pragma once

namespace winograd
{

template <const int M_BLOCK, const int N_BLOCK, typename TIn, typename TOut>
class BatchedBlockedGemm
{
  public:
    /** Create a new batched blocked GEMM operator. */
    BatchedBlockedGemm(
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
    );

    BatchedBlockedGemm(const BatchedBlockedGemm&) = delete;
    BatchedBlockedGemm operator=(const BatchedBlockedGemm&) = delete;

    /** Get a window of work performed by the operator. */
    unsigned int get_window() const;

    /** Perform a portion of the work of the operator. */
    void run(const unsigned int start, const unsigned int stop);

  private:
    const unsigned int n_gemms;
    const int M, N, K;
    const int a_matrix_stride, a_row_stride;
    const int b_matrix_stride, b_row_stride;
    const int c_matrix_stride, c_row_stride;
    const TIn* const a_ptr;
    const TIn* const b_ptr;
    TOut* const c_ptr;
};

}  // namespace winograd
