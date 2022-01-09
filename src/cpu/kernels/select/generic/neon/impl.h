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
#ifndef SRC_CORE_NEON_KERNELS_SELECT_IMPL_H
#define SRC_CORE_NEON_KERNELS_SELECT_IMPL_H

#include <cstdint>

namespace arm_compute
{
class ITensor;
class Window;

namespace cpu
{
template <typename ScalarType>
void select_op_not_same_rank(const ITensor *cond, const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template <typename ScalarType, typename VectorType>
void select_op_32(const ITensor *cond, const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template <typename ScalarType, typename VectorType>
void select_op_16(const ITensor *cond, const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template <typename ScalarType, typename VectorType>
void select_op_8(const ITensor *cond, const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template <typename ScalarType, typename VectorType>
void select_op(const ITensor *cond, const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
               const int window_step_x, const int window_start_x, const int window_end_x, const int limit, VectorType (*condition_conversion)(const uint8_t *));

} // namespace cpu
} // namespace arm_compute
#endif //SRC_CORE_NEON_KERNELS_SELECT_IMPL_H