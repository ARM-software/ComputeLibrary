/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLLogicalNot.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "src/core/CL/kernels/CLElementWiseUnaryLayerKernel.h"

#include <utility>

namespace arm_compute
{
namespace experimental
{
void CLLogicalNot::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = std::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::LOGICAL_NOT);
    _kernel = std::move(k);
}

Status CLLogicalNot::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::LOGICAL_NOT);
}

void CLLogicalNot::run(ITensorPack &tensors)
{
    ICLOperator::run(tensors);
}
} // namespace experimental

struct CLLogicalNot::Impl
{
    const ICLTensor                            *src{ nullptr };
    ICLTensor                                  *dst{ nullptr };
    std::unique_ptr<experimental::CLLogicalNot> op{ nullptr };
};

CLLogicalNot::CLLogicalNot()
    : _impl(std::make_unique<Impl>())
{
}
CLLogicalNot::CLLogicalNot(CLLogicalNot &&) = default;
CLLogicalNot &CLLogicalNot::operator=(CLLogicalNot &&) = default;
CLLogicalNot::~CLLogicalNot()                          = default;

void CLLogicalNot::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLLogicalNot::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = std::make_unique<experimental::CLLogicalNot>();
    _impl->op->configure(compile_context, input->info(), output->info());
}

Status CLLogicalNot::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLLogicalNot::validate(input, output);
}

void CLLogicalNot::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}

} // namespace arm_compute