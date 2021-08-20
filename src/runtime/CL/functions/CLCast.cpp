/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLCast.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClCast.h"

#include <utility>

namespace arm_compute
{
struct CLCast::Impl
{
    const ICLTensor                *src{ nullptr };
    ICLTensor                      *dst{ nullptr };
    std::unique_ptr<opencl::ClCast> op{ nullptr };
};

CLCast::CLCast()
    : _impl(std::make_unique<Impl>())
{
}
CLCast::CLCast(CLCast &&) = default;
CLCast &CLCast::operator=(CLCast &&) = default;
CLCast::~CLCast()                    = default;

void CLCast::configure(const ICLTensor *input, ICLTensor *output, ConvertPolicy policy)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, policy);
}

void CLCast::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _impl->src = input;
    _impl->dst = output;

    _impl->op = std::make_unique<opencl::ClCast>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->dst->info(), policy);
}

Status CLCast::validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy)
{
    return opencl::ClCast::validate(input, output, policy);
}

void CLCast::run()
{
    ITensorPack pack = { { ACL_SRC, _impl->src }, { ACL_DST, _impl->dst } };
    _impl->op->run(pack);
}
} // namespace arm_compute
