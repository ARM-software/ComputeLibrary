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
#include "arm_compute/runtime/CL/functions/CLLogicalOr.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "src/core/CL/kernels/CLElementwiseOperationKernel.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
namespace experimental
{
void CLLogicalOr::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLLogicalBinaryKernel>();
    k->configure(compile_context, kernels::LogicalOperation::Or, input1, input2, output);
    _kernel = std::move(k);
}

Status CLLogicalOr::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return CLLogicalBinaryKernel::validate(kernels::LogicalOperation::Or, input1, input2, output);
}

void CLLogicalOr::run(ITensorPack &tensors)
{
    ICLOperator::run(tensors);
}
} // namespace experimental

struct CLLogicalOr::Impl
{
    const ICLTensor                           *src0{ nullptr };
    const ICLTensor                           *src1{ nullptr };
    ICLTensor                                 *dst{ nullptr };
    std::unique_ptr<experimental::CLLogicalOr> op{ nullptr };
};

CLLogicalOr::CLLogicalOr()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLLogicalOr::CLLogicalOr(CLLogicalOr &&) = default;
CLLogicalOr &CLLogicalOr::operator=(CLLogicalOr &&) = default;
CLLogicalOr::~CLLogicalOr()                         = default;

void CLLogicalOr::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output);
}

void CLLogicalOr::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output)
{
    _impl->src0 = input1;
    _impl->src1 = input2;
    _impl->dst  = output;
    _impl->op   = arm_compute::support::cpp14::make_unique<experimental::CLLogicalOr>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info());
}

Status CLLogicalOr::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return experimental::CLLogicalOr::validate(input1, input2, output);
}

void CLLogicalOr::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}
} // namespace arm_compute
