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
#include "arm_compute/runtime/CL/functions/CLElementWiseUnaryLayer.h"

#include "arm_compute/core/CL/kernels/CLElementWiseUnaryLayerKernel.h"
#include "support/ToolchainSupport.h"

#include <utility>

namespace arm_compute
{
void CLRsqrtLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(input, output, ElementWiseUnary::RSQRT);
    _kernel = std::move(k);
}
Status CLRsqrtLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::RSQRT);
}

void CLExpLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(input, output, ElementWiseUnary::EXP);
    _kernel = std::move(k);
}
Status CLExpLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::EXP);
}

void CLNegLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(input, output, ElementWiseUnary::NEG);
    _kernel = std::move(k);
}
Status CLNegLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::NEG);
}

void CLSinLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(input, output, ElementWiseUnary::SIN);
    _kernel = std::move(k);
}
Status CLSinLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::SIN);
}

void CLAbsLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(input, output, ElementWiseUnary::ABS);
    _kernel = std::move(k);
}
Status CLAbsLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::ABS);
}
void CLLogLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(input, output, ElementWiseUnary::LOG);
    _kernel = std::move(k);
}
Status CLLogLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::LOG);
}

void CLRoundLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(input, output, ElementWiseUnary::ROUND);
    _kernel = std::move(k);
}
Status CLRoundLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::ROUND);
}

} // namespace arm_compute
