/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "src/cpu/CpuTypes.h"
#include "src/cpu/kernels/depthwiseconv2d/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
void neon_fp16_deptwiseconv2dnative(const ITensor *src, const ITensor *weights, const ITensor *bias,
                                    ITensor *dst, const Window &window, bool has_biases, const ConvolutionInfo &info)
{
    return run_depthwise_float<float16_t, float16_t>(src, weights, bias, dst, window, has_biases, info);
}
}
} // namespace arm_compute
#endif //__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
