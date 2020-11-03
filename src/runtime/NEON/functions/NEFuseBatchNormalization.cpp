/*
 * Copyright (c) 2018-2020 Arm Limited.
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

#include "arm_compute/runtime/NEON/functions/NEFuseBatchNormalization.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NEFuseBatchNormalizationKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
NEFuseBatchNormalization::~NEFuseBatchNormalization() = default;

NEFuseBatchNormalization::NEFuseBatchNormalization()
    : _fuse_bn_kernel()
{
}

void NEFuseBatchNormalization::configure(const ITensor *input_weights, const ITensor *bn_mean, const ITensor *bn_var,
                                         ITensor *fused_weights, ITensor *fused_bias,
                                         const ITensor *input_bias, const ITensor *bn_beta, const ITensor *bn_gamma,
                                         float epsilon, FuseBatchNormalizationType fbn_type)
{
    _fuse_bn_kernel = arm_compute::support::cpp14::make_unique<NEFuseBatchNormalizationKernel>();
    _fuse_bn_kernel->configure(input_weights, bn_mean, bn_var, fused_weights, fused_bias, input_bias, bn_beta, bn_gamma, epsilon, fbn_type);
}

Status NEFuseBatchNormalization::validate(const ITensorInfo *input_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                                          const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                                          const ITensorInfo *input_bias, const ITensorInfo *bn_beta, const ITensorInfo *bn_gamma,
                                          float epsilon, FuseBatchNormalizationType fbn_type)
{
    return NEFuseBatchNormalizationKernel::validate(input_weights, bn_mean, bn_var, fused_weights, fused_bias, input_bias, bn_beta, bn_gamma, epsilon, fbn_type);
}

void NEFuseBatchNormalization::run()
{
    NEScheduler::get().schedule(_fuse_bn_kernel.get(), Window::DimY);
}
} // namespace arm_compute
