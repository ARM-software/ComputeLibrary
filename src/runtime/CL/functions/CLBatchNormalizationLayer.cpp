/*
 * Copyright (c) 2017-2020 Arm Limited.
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

#include "arm_compute/runtime/CL/functions/CLBatchNormalizationLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/MemorySupport.h"

#include "src/core/CL/kernels/CLBatchNormalizationLayerKernel.h"

namespace arm_compute
{
CLBatchNormalizationLayer::CLBatchNormalizationLayer()
    : _norm_kernel(support::cpp14::make_unique<CLBatchNormalizationLayerKernel>())
{
}

CLBatchNormalizationLayer::~CLBatchNormalizationLayer() = default;

void CLBatchNormalizationLayer::configure(ICLTensor *input, ICLTensor *output, const ICLTensor *mean, const ICLTensor *var, const ICLTensor *beta, const ICLTensor *gamma, float epsilon,
                                          ActivationLayerInfo act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, mean, var, beta, gamma, epsilon, act_info);
}

void CLBatchNormalizationLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const ICLTensor *mean, const ICLTensor *var, const ICLTensor *beta,
                                          const ICLTensor *gamma, float epsilon,
                                          ActivationLayerInfo act_info)
{
    _norm_kernel->configure(compile_context, input, output, mean, var, beta, gamma, epsilon, act_info);
}

Status CLBatchNormalizationLayer::validate(const ITensorInfo *input, const ITensorInfo *output,
                                           const ITensorInfo *mean, const ITensorInfo *var,
                                           const ITensorInfo *beta, const ITensorInfo *gamma,
                                           float epsilon, ActivationLayerInfo act_info)
{
    return CLBatchNormalizationLayerKernel::validate(input, output, mean, var, beta, gamma, epsilon, act_info);
}

void CLBatchNormalizationLayer::run()
{
    CLScheduler::get().enqueue(*_norm_kernel, true);
}
} // namespace arm_compute