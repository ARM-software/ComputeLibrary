/*
 * Copyright (c) 2020, 2023 Arm Limited.
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
#ifndef ACL_SRC_CORE_NEON_KERNELS_BATCHNORMALIZATION_IMPL_LIST_H
#define ACL_SRC_CORE_NEON_KERNELS_BATCHNORMALIZATION_IMPL_LIST_H

namespace arm_compute
{
namespace cpu
{
#define DECLARE_BATCH_NORMALIZATION_KERNEL(func_name)                                                        \
    void func_name(ITensor *src, ITensor *dst, const ITensor *mean, const ITensor *var, const ITensor *beta, \
                   const ITensor *gamma, float epsilon, ActivationLayerInfo &act_info, const Window &window)

DECLARE_BATCH_NORMALIZATION_KERNEL(fp16_neon_batch_normalization);
DECLARE_BATCH_NORMALIZATION_KERNEL(fp16_sve_batch_normalization);
DECLARE_BATCH_NORMALIZATION_KERNEL(fp32_neon_batch_normalization);
DECLARE_BATCH_NORMALIZATION_KERNEL(fp32_sve_batch_normalization);

#define DECLARE_BATCH_NORMALIZATION_NCHW_KERNEL(func_name)                                                         \
    void func_name(const Window &window, ITensor *input, ITensor *output, const ITensor *mean, const ITensor *var, \
                   const ITensor *beta, const ITensor *gamma, float epsilon, ActivationLayerInfo act_info)

DECLARE_BATCH_NORMALIZATION_NCHW_KERNEL(fp16_batch_normalization_nchw_non_fused);
DECLARE_BATCH_NORMALIZATION_NCHW_KERNEL(fp32_batch_normalization_nchw_non_fused);
DECLARE_BATCH_NORMALIZATION_NCHW_KERNEL(fp16_batch_normalization_nchw_non_fused_relu);
DECLARE_BATCH_NORMALIZATION_NCHW_KERNEL(fp16_batch_normalization_nchw_non_fused_brelu);
DECLARE_BATCH_NORMALIZATION_NCHW_KERNEL(fp16_batch_normalization_nchw_non_fused_lubrelu);
DECLARE_BATCH_NORMALIZATION_NCHW_KERNEL(fp32_batch_normalization_nchw_non_fused_relu);
DECLARE_BATCH_NORMALIZATION_NCHW_KERNEL(fp32_batch_normalization_nchw_non_fused_brelu);
DECLARE_BATCH_NORMALIZATION_NCHW_KERNEL(fp32_batch_normalization_nchw_non_fused_lubrelu);

#undef DECLARE_BATCH_NORMALIZATION_KERNEL
#undef DECLARE_BATCH_NORMALIZATION_NCHW_KERNEL

} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CORE_NEON_KERNELS_BATCHNORMALIZATION_IMPL_LIST_H
