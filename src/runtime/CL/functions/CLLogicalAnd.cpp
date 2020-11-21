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
#include "arm_compute/runtime/CL/functions/CLLogicalAnd.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "src/core/CL/kernels/CLElementwiseOperationKernel.h"

#include <utility>

namespace arm_compute
{
namespace experimental
{
void CLLogicalAnd::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output)
{
    auto k = std::make_unique<CLLogicalBinaryKernel>();
    k->configure(compile_context, kernels::LogicalOperation::And, input1, input2, output);
    _kernel = std::move(k);
}

Status CLLogicalAnd::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return CLLogicalBinaryKernel::validate(kernels::LogicalOperation::And, input1, input2, output);
}

void CLLogicalAnd::run(ITensorPack &tensors)
{
    ICLOperator::run(tensors);
}
} // namespace experimental

struct CLLogicalAnd::Impl
{
    const ICLTensor                            *src0{ nullptr };
    const ICLTensor                            *src1{ nullptr };
    ICLTensor                                  *dst{ nullptr };
    std::unique_ptr<experimental::CLLogicalAnd> op{ nullptr };
};

CLLogicalAnd::CLLogicalAnd()
    : _impl(std::make_unique<Impl>())
{
}
CLLogicalAnd::CLLogicalAnd(CLLogicalAnd &&) = default;
CLLogicalAnd &CLLogicalAnd::operator=(CLLogicalAnd &&) = default;
CLLogicalAnd::~CLLogicalAnd()                          = default;

void CLLogicalAnd::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output);
}

void CLLogicalAnd::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output)
{
    _impl->src0 = input1;
    _impl->src1 = input2;
    _impl->dst  = output;
    _impl->op   = std::make_unique<experimental::CLLogicalAnd>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info());
}

Status CLLogicalAnd::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return experimental::CLLogicalAnd::validate(input1, input2, output);
}

void CLLogicalAnd::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}
} // namespace arm_compute
