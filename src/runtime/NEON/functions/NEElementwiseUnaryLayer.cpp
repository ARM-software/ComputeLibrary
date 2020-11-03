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
#include "arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h"

#include "src/core/NEON/kernels/NEElementwiseUnaryKernel.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
void NERsqrtLayer::configure(const ITensor *input, ITensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<NEElementwiseUnaryKernel>();
    k->configure(ElementWiseUnary::RSQRT, input, output);
    _kernel = std::move(k);
}
Status NERsqrtLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return NEElementwiseUnaryKernel::validate(ElementWiseUnary::RSQRT, input, output);
}

void NEExpLayer::configure(const ITensor *input, ITensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<NEElementwiseUnaryKernel>();
    k->configure(ElementWiseUnary::EXP, input, output);
    _kernel = std::move(k);
}
Status NEExpLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return NEElementwiseUnaryKernel::validate(ElementWiseUnary::EXP, input, output);
}

void NENegLayer::configure(const ITensor *input, ITensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<NEElementwiseUnaryKernel>();
    k->configure(ElementWiseUnary::NEG, input, output);
    _kernel = std::move(k);
}
Status NENegLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return NEElementwiseUnaryKernel::validate(ElementWiseUnary::NEG, input, output);
}

void NELogLayer::configure(const ITensor *input, ITensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<NEElementwiseUnaryKernel>();
    k->configure(ElementWiseUnary::LOG, input, output);
    _kernel = std::move(k);
}
Status NELogLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return NEElementwiseUnaryKernel::validate(ElementWiseUnary::LOG, input, output);
}

void NEAbsLayer::configure(const ITensor *input, ITensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<NEElementwiseUnaryKernel>();
    k->configure(ElementWiseUnary::ABS, input, output);
    _kernel = std::move(k);
}
Status NEAbsLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return NEElementwiseUnaryKernel::validate(ElementWiseUnary::ABS, input, output);
}

void NERoundLayer::configure(const ITensor *input, ITensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<NEElementwiseUnaryKernel>();
    k->configure(ElementWiseUnary::ROUND, input, output);
    _kernel = std::move(k);
}
Status NERoundLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return NEElementwiseUnaryKernel::validate(ElementWiseUnary::ROUND, input, output);
}

void NESinLayer::configure(const ITensor *input, ITensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<NEElementwiseUnaryKernel>();
    k->configure(ElementWiseUnary::SIN, input, output);
    _kernel = std::move(k);
}
Status NESinLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return NEElementwiseUnaryKernel::validate(ElementWiseUnary::SIN, input, output);
}

} // namespace arm_compute
