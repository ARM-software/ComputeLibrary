/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_FUSE_BATCH_NORMALIZATION_GENERIC_IMPL_H
#define SRC_CORE_NEON_KERNELS_FUSE_BATCH_NORMALIZATION_GENERIC_IMPL_H

#include "arm_compute/core/Helpers.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{
template <typename T>
void fused_batch_normalization_conv(const ITensor *conv_weights, const ITensor *conv_bias, ITensor *fused_weights, ITensor *fused_bias,
                                    const ITensor *bn_mean, const ITensor *bn_var, const ITensor *bn_beta, const ITensor *bn_gamma, float epsilon, const Window &window);
}
}
#endif //SRC_CORE_NEON_KERNELS_FUSE_BATCH_NORMALIZATION_GENERIC_IMPL_H