/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/gpu/cl/operators/ClElementwiseUnary.h"

#include "src/gpu/cl/kernels/ClElementwiseUnaryKernel.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
namespace opencl
{
void ClRsqrt::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst);
    auto k = std::make_unique<kernels::ClElementWiseUnaryKernel>();
    k->configure(compile_context, src, dst, ElementWiseUnary::RSQRT);
    _kernel = std::move(k);
}

Status ClRsqrt::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return kernels::ClElementWiseUnaryKernel::validate(src, dst, ElementWiseUnary::RSQRT);
}

void ClExp::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst)
{
    auto k = std::make_unique<kernels::ClElementWiseUnaryKernel>();
    k->configure(compile_context, src, dst, ElementWiseUnary::EXP);
    _kernel = std::move(k);
}

Status ClExp::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return kernels::ClElementWiseUnaryKernel::validate(src, dst, ElementWiseUnary::EXP);
}

void ClNeg::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst)
{
    auto k = std::make_unique<kernels::ClElementWiseUnaryKernel>();
    k->configure(compile_context, src, dst, ElementWiseUnary::NEG);
    _kernel = std::move(k);
}

Status ClNeg::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return kernels::ClElementWiseUnaryKernel::validate(src, dst, ElementWiseUnary::NEG);
}

void ClSin::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst)
{
    auto k = std::make_unique<kernels::ClElementWiseUnaryKernel>();
    k->configure(compile_context, src, dst, ElementWiseUnary::SIN);
    _kernel = std::move(k);
}

Status ClSin::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return kernels::ClElementWiseUnaryKernel::validate(src, dst, ElementWiseUnary::SIN);
}

void ClAbs::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst)
{
    auto k = std::make_unique<kernels::ClElementWiseUnaryKernel>();
    k->configure(compile_context, src, dst, ElementWiseUnary::ABS);
    _kernel = std::move(k);
}

Status ClAbs::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return kernels::ClElementWiseUnaryKernel::validate(src, dst, ElementWiseUnary::ABS);
}

void ClLog::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst)
{
    auto k = std::make_unique<kernels::ClElementWiseUnaryKernel>();
    k->configure(compile_context, src, dst, ElementWiseUnary::LOG);
    _kernel = std::move(k);
}

Status ClLog::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return kernels::ClElementWiseUnaryKernel::validate(src, dst, ElementWiseUnary::LOG);
}

void ClRound::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst)
{
    auto k = std::make_unique<kernels::ClElementWiseUnaryKernel>();
    k->configure(compile_context, src, dst, ElementWiseUnary::ROUND);
    _kernel = std::move(k);
}

Status ClRound::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return kernels::ClElementWiseUnaryKernel::validate(src, dst, ElementWiseUnary::ROUND);
}
} // namespace opencl
} // namespace arm_compute
