/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLInstanceNormalizationLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLInstanceNormalizationLayerKernel.h"

namespace arm_compute
{
CLInstanceNormalizationLayer::CLInstanceNormalizationLayer(CLRuntimeContext *ctx) // NOLINT
    : _inst_norm_kernel(), _mean_var_kernel(), _mean_var_tensor(), _ctx(ctx)
{
}
CLInstanceNormalizationLayer::~CLInstanceNormalizationLayer()
{
}

void CLInstanceNormalizationLayer::configure(
    ICLTensor *input, ICLTensor *output, float gamma, float beta, float epsilon, bool use_mixed_precision)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, gamma, beta, epsilon, use_mixed_precision);
}

void CLInstanceNormalizationLayer::configure(const CLCompileContext &compile_context,
                                             ICLTensor              *input,
                                             ICLTensor              *output,
                                             float                   gamma,
                                             float                   beta,
                                             float                   epsilon,
                                             bool                    use_mixed_precision)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, gamma, beta, epsilon, use_mixed_precision);
    auto w = std::make_unique<CLComputeMeanVariance>();
    w->configure(compile_context, input, &_mean_var_tensor, use_mixed_precision);
    _mean_var_kernel = std::move(w);
    auto k           = std::make_unique<CLInstanceNormalizationLayerKernel>();
    k->configure(compile_context, input, &_mean_var_tensor, output,
                 InstanceNormalizationLayerKernelInfo(gamma, beta, epsilon, use_mixed_precision));
    _inst_norm_kernel = std::move(k);
    _mean_var_tensor.allocator()->allocate();
}

Status CLInstanceNormalizationLayer::validate(const ITensorInfo *input,
                                              const ITensorInfo *output,
                                              float              gamma,
                                              float              beta,
                                              float              epsilon,
                                              bool               use_mixed_precision)
{
    return CLInstanceNormalizationLayerKernel::validate(
        input, output, InstanceNormalizationLayerKernelInfo(gamma, beta, epsilon, use_mixed_precision));
}

void CLInstanceNormalizationLayer::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(!_inst_norm_kernel,
                             "The child class didn't set the CL kernel or function isn't configured");
    schedule_kernel_on_ctx(_ctx, _mean_var_kernel.get());
    schedule_kernel_on_ctx(_ctx, _inst_norm_kernel.get());
}

} // namespace arm_compute
