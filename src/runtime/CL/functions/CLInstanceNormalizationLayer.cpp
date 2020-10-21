/*
 * Copyright (c) 2019-2020 Arm Limited.
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

#include "arm_compute/core/Types.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLInstanceNormalizationLayerKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
CLInstanceNormalizationLayer::CLInstanceNormalizationLayer()
{
}

void CLInstanceNormalizationLayer::configure(ICLTensor *input, ICLTensor *output, float gamma, float beta, float epsilon, bool use_mixed_precision)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, gamma, beta, epsilon, use_mixed_precision);
}

void CLInstanceNormalizationLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, float gamma, float beta, float epsilon, bool use_mixed_precision)
{
    auto k = arm_compute::support::cpp14::make_unique<CLInstanceNormalizationLayerKernel>();
    k->configure(compile_context, input, output, InstanceNormalizationLayerKernelInfo(gamma, beta, epsilon, use_mixed_precision));
    _kernel = std::move(k);
}

Status CLInstanceNormalizationLayer::validate(const ITensorInfo *input, const ITensorInfo *output, float gamma, float beta, float epsilon, bool use_mixed_precision)
{
    return CLInstanceNormalizationLayerKernel::validate(input, output, InstanceNormalizationLayerKernelInfo(gamma, beta, epsilon, use_mixed_precision));
}
} // namespace arm_compute