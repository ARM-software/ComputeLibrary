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
#ifndef SRC_CORE_KERNELS_GEMMMATRIXMUL_IMPL_H
#define SRC_CORE_KERNELS_GEMMMATRIXMUL_IMPL_H
#include "arm_compute/core/Helpers.h"
#include "src/core/CPP/Validate.h"

namespace arm_compute
{
namespace cpu
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void vector_matrix_multiply_f16(const ITensor *lhs, const ITensor *rhs, ITensor *dst, const Window &window, const ThreadInfo &info, float alpha);

void matrix_matrix_multiply_f16(const ITensor *lhs, const ITensor *rhs, ITensor *dst, const Window &window, const ThreadInfo &info, float alpha);

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

void vector_matrix_multiply_f32(const ITensor *lhs, const ITensor *rhs, ITensor *dst, const Window &window, const ThreadInfo &info, float alpha);

void matrix_matrix_multiply_f32(const ITensor *lhs, const ITensor *rhs, ITensor *dst, const Window &window, const ThreadInfo &info, float alpha);

} // namespace cpu
} // namespace arm_compute
#endif //define SRC_CORE_KERNELS_GEMMMATRIXMUL_IMPL_H
