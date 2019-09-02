/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#include "arm_compute/runtime/CL/functions/CLFuseBatchNormalization.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
CLFuseBatchNormalization::CLFuseBatchNormalization()
    : _fuse_bn_kernel()
{
}

void CLFuseBatchNormalization::configure(const ICLTensor *input_weights, const ICLTensor *bn_mean, const ICLTensor *bn_var,
                                         ICLTensor *fused_weights, ICLTensor *fused_bias,
                                         const ICLTensor *input_bias, const ICLTensor *bn_beta, const ICLTensor *bn_gamma,
                                         float epsilon, FuseBatchNormalizationType fbn_type)
{
    _fuse_bn_kernel.configure(input_weights, bn_mean, bn_var, fused_weights, fused_bias, input_bias, bn_beta, bn_gamma, epsilon, fbn_type);
}

Status CLFuseBatchNormalization::validate(const ITensorInfo *input_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                                          const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                                          const ITensorInfo *input_bias, const ITensorInfo *bn_beta, const ITensorInfo *bn_gamma,
                                          float epsilon, FuseBatchNormalizationType fbn_type)
{
    return CLFuseBatchNormalizationKernel::validate(input_weights, bn_mean, bn_var, fused_weights, fused_bias, input_bias, bn_beta, bn_gamma, epsilon, fbn_type);
}

void CLFuseBatchNormalization::run()
{
    CLScheduler::get().enqueue(_fuse_bn_kernel, true);
}
} // namespace arm_compute
