/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLDepthConvertLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/ICLKernel.h"
#include "src/runtime/gpu/cl/operators/ClCast.h"

#include <utility>

namespace arm_compute
{
struct CLDepthConvertLayer::Impl
{
    const ICLTensor                *src{ nullptr };
    ICLTensor                      *dst{ nullptr };
    std::unique_ptr<opencl::ClCast> op{ nullptr };
};

CLDepthConvertLayer::CLDepthConvertLayer()
    : _impl(std::make_unique<Impl>())
{
}
CLDepthConvertLayer::CLDepthConvertLayer(CLDepthConvertLayer &&) = default;
CLDepthConvertLayer &CLDepthConvertLayer::operator=(CLDepthConvertLayer &&) = default;
CLDepthConvertLayer::~CLDepthConvertLayer()                                 = default;

void CLDepthConvertLayer::configure(const ICLTensor *input, ICLTensor *output, ConvertPolicy policy, uint32_t shift)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, policy, shift);
}

void CLDepthConvertLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_UNUSED(shift);

    _impl->src = input;
    _impl->dst = output;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_impl->src, _impl->dst);
    ARM_COMPUTE_ERROR_ON(shift != 0);

    _impl->op = std::make_unique<opencl::ClCast>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->dst->info(), policy);
}

Status CLDepthConvertLayer::validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_RETURN_ERROR_ON(shift != 0);
    return opencl::ClCast::validate(input, output, policy);
}

void CLDepthConvertLayer::run()
{
    ITensorPack pack = { { ACL_SRC, _impl->src }, { ACL_DST, _impl->dst } };
    _impl->op->run(pack);
}
} // namespace arm_compute
