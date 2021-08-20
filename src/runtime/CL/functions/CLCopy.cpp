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
#include "arm_compute/runtime/CL/functions/CLCopy.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClCopy.h"

#include <utility>

namespace arm_compute
{
struct CLCopy::Impl
{
    const ICLTensor                *src{ nullptr };
    ICLTensor                      *dst{ nullptr };
    std::unique_ptr<opencl::ClCopy> op{ nullptr };
};

CLCopy::CLCopy()
    : _impl(std::make_unique<Impl>())
{
}
CLCopy::CLCopy(CLCopy &&) = default;
CLCopy &CLCopy::operator=(CLCopy &&) = default;
CLCopy::~CLCopy()                    = default;

void CLCopy::configure(ICLTensor *input, ICLTensor *output, Window *dst_window)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, dst_window);
}

void CLCopy::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, Window *dst_window)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _impl->src = input;
    _impl->dst = output;

    _impl->op = std::make_unique<opencl::ClCopy>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->dst->info(), dst_window);
}

Status CLCopy::validate(const ITensorInfo *input, const ITensorInfo *output, Window *dst_window)
{
    return opencl::ClCopy::validate(input, output, dst_window);
}

void CLCopy::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
