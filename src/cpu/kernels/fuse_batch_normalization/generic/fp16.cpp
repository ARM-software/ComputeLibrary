/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "src/cpu/kernels/fuse_batch_normalization/generic/impl.h"

namespace arm_compute
{
namespace cpu
{
void fused_batch_normalization_conv_f16(const ITensor *conv_weights,
                                        const ITensor *conv_bias,
                                        ITensor       *fused_weights,
                                        ITensor       *fused_bias,
                                        const ITensor *bn_mean,
                                        const ITensor *bn_var,
                                        const ITensor *bn_beta,
                                        const ITensor *bn_gamma,
                                        float          epsilon,
                                        const Window  &window)
{
    return fused_batch_normalization_conv<float16_t>(conv_weights, conv_bias, fused_weights, fused_bias, bn_mean,
                                                     bn_var, bn_beta, bn_gamma, epsilon, window);
}

void fused_batch_normalization_dwc_nchw_f16(const ITensor *dwc_weights,
                                            const ITensor *dwc_bias,
                                            ITensor       *fused_weights,
                                            ITensor       *fused_bias,
                                            const ITensor *bn_mean,
                                            const ITensor *bn_var,
                                            const ITensor *bn_beta,
                                            const ITensor *bn_gamma,
                                            float          epsilon,
                                            const Window  &window)
{
    return fused_batch_normalization_dwc_nchw<float16_t>(dwc_weights, dwc_bias, fused_weights, fused_bias, bn_mean,
                                                         bn_var, bn_beta, bn_gamma, epsilon, window);
}
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
