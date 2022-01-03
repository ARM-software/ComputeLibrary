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
#ifndef SRC_CORE_SVE_KERNELS_INSTANCENORM_IMPL_H
#define SRC_CORE_SVE_KERNELS_INSTANCENORM_IMPL_H
#include "arm_compute/core/Helpers.h"
namespace arm_compute
{
namespace cpu
{
template <typename T, typename AccType = T>
void instance_normalization_nchw(ITensor *input, ITensor *output, float gamma, float beta, float epsilon, const Window &window);

template <typename InputType, typename AccType = InputType>
void vector_float_sum(AccType &result, AccType &result_square, const InputType &inputs);

template <typename InputType, typename AccType = InputType>
InputType vector_float_norm(const InputType &inputs, const AccType &vec_mean, const AccType &vec_multip, const AccType &vec_beta);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
template <>
inline void vector_float_sum(float32x4_t &result, float32x4_t &result_square, const float16x8_t &inputs);

template <>
inline float16x8_t vector_float_norm(const float16x8_t &inputs, const float32x4_t &vec_mean, const float32x4_t &vec_multip, const float32x4_t &vec_beta);
#endif //defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

} // namespace cpu
} // namespace arm_compute
#endif //define SRC_CORE_SVE_KERNELS_INSTANCENORM_IMPL_H
