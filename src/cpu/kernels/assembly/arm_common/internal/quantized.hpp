/*
 * Copyright (c) 2019, 2023-2026 Arm Limited.
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

#include "arm_common/internal/utils.hpp" // IndirectInputArg

namespace arm_gemm {

template<typename Tin, typename Tout>
void requantize_block_32(const Requantize32 &qp, unsigned int width, unsigned int height,
                         const Tin *input, unsigned int in_stride, Tout *output, unsigned int out_stride,
                         const int32_t *row_bias, const int32_t *col_bias, unsigned int start_col);

template<typename T>
void compute_row_sums(const Requantize32 &qp, unsigned int width, unsigned int height,
                      const T *input, unsigned int in_stride, int32_t *row_bias);

template<typename T>
void compute_col_sums(const Requantize32 &qp, unsigned int width, unsigned int height,
                      const T *input, unsigned int in_stride, int32_t *col_bias, unsigned int depth,
                      unsigned int multi, unsigned int first_col);

/** Compute raw column sums of a matrix: col_sums[n] = sum_{k} input[k * in_stride + n].
 *  Unlike compute_col_sums(), this does not apply any quantization offsets or scaling —
 *  it stores the plain integer sums for use as weight column reductions in the
 *  DequantizeFloat a_offset correction path. */
template<typename T>
void compute_raw_col_sums(unsigned int width, unsigned int height,
                          const T *input, unsigned int in_stride, int32_t *col_sums);

template<typename T>
void row_sums_indirect(size_t num_strings, const unsigned int *string_lengths, IndirectInputArg<T> A_arg,
                       size_t M, int32_t *output_ptr, const Requantize32 *qp);

template<typename T>
void dequantize_block_32(const DequantizeFloat &qp, unsigned int width, unsigned int height,
                         const int32_t* input, unsigned int in_stride, T *output, unsigned int out_stride,
                         const T *row_bias, bool not_first_pass, const Activation &act,
                         const int32_t *col_bias = nullptr, const int32_t *row_sum = nullptr,
                         int32_t k_total = 0);

} // namespace arm_gemm
