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
#include "utils.hpp"

template <typename TIn, typename TOut>
inline void Gemm(const TIn* const a, const TIn* const b, TOut *c,
          const int M, const int K, const int N,
          const int a_row_stride,
          const int b_row_stride,
          const int c_row_stride,
          const bool a_transposed=false,
          const bool b_transposed=false) {
  // Array access methods
  const auto A = [a, a_transposed, M, K, a_row_stride] (const int i, const int j) -> TIn {
    return a[(!a_transposed) ? i*a_row_stride + j : i + j*M];
  };

  const auto B = [b, b_transposed, K, N, b_row_stride] (const int i, const int j) -> TIn {
    return b[(!b_transposed) ? i*b_row_stride + j : i + j*N];
  };

  const auto C = [c, c_row_stride] (const int i, const int j) -> TOut& {
    return c[i*c_row_stride + j];
  };

  // Perform the matrix multiplication
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

template <const int M_BLOCK, const int N_BLOCK, typename TIn, typename TOut>
inline void BlockedGemm(
  const TIn* const a, const TIn* const b, TOut *c,
  const int M, const int K, const int N,
  const int a_row_stride,
  const int b_row_stride,
  const int c_row_stride
) {
  // Array access methods
  const auto A = [a, M, K, a_row_stride] (const int i, const int j) -> TIn {
    return a[i*a_row_stride + j];
  };

  const auto B = [b, K, N, b_row_stride] (const int i, const int j) -> TIn {
    return b[i*b_row_stride + j];
  };

  const auto C = [c, c_row_stride] (const int i, const int j) -> TOut& {
    return c[i*c_row_stride + j];
  };

  const int M_BLOCKS = iceildiv(M, M_BLOCK);
  const int N_BLOCKS = iceildiv(N, N_BLOCK);

  // For each block of output rows
  for (int mblock = 0; mblock < M_BLOCKS; mblock++) {
    // For each block of output columns
    for (int nblock = 0; nblock < N_BLOCKS; nblock++) {
      // Create an appropriately sized block of accumulators
      TOut accum[M_BLOCK][N_BLOCK];
      for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < N_BLOCK; j++) {
          accum[i][j] = static_cast<TOut>(0);
        }
      }

      // Perform this portion of the matrix multiply
      for (int k = 0; k < K; k++) {
        // Load elements of A
        TIn elems_a[M_BLOCK];
        for (int i = 0; i < M_BLOCK; i++) {
          elems_a[i] = A(mblock*M_BLOCK + i, k);
        }

        // Load elements of B
        TIn elems_b[N_BLOCK];
        for (int j = 0; j < N_BLOCK; j++) {
          elems_b[j] = B(k, nblock*N_BLOCK + j);
        }

        // Perform the partial matrix multiply
        for (int i = 0; i < M_BLOCK; i++) {
          for (int j = 0; j < N_BLOCK; j++) {
            accum[i][j] += elems_a[i] * elems_b[j];
          }
        }
      }

      // Store the partial product
      for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < N_BLOCK; j++) {
          C(mblock*M_BLOCK + i, nblock*N_BLOCK + j) = accum[i][j];
        }
      }
    }
  }
}

#include "gemm/a64_sgemm.hpp"
