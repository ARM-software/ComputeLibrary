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
#include "arm_compute/core/CL/kernels/CLElementwiseOperationKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLPReluLayer.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
namespace
{
void configure_border_handler(const CLCompileContext &compile_context, CLFillBorderKernel &border_handler, BorderSize border_size, ITensorInfo *input1, ITensorInfo *input2, const ITensorInfo *output)
{
    if(output->dimension(0) > 1)
    {
        ITensorInfo *broadcasted_info = (input1->dimension(0) == 1) ? input1 : input2;

        if(broadcasted_info->dimension(0) == 1)
        {
            border_handler.configure(compile_context, broadcasted_info, border_size, BorderMode::REPLICATE);
        }
    }
}
void select_border_input(InputTensorMap &tensor_map, InputTensorMap &inputs, OutputTensorMap &outputs)
{
    if(outputs.at(TensorType::ACL_DST)->info()->dimension(0) > 1)
    {
        if(inputs.at(TensorType::ACL_SRC_1)->info()->dimension(0) == 1)
        {
            tensor_map[TensorType::ACL_SRC] = inputs.at(TensorType::ACL_SRC_1);
        }
        else
        {
            tensor_map[TensorType::ACL_SRC] = inputs.at(TensorType::ACL_SRC_0);
        }
    }
}
} // namespace

namespace experimental
{
CLPReluLayer::CLPReluLayer()
    : _border_handler()
{
}

void CLPReluLayer::configure(const CLCompileContext &compile_context, ITensorInfo *input, ITensorInfo *alpha, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::PRELU, input, alpha, output);
    _kernel = std::move(k);
    configure_border_handler(compile_context, _border_handler, _kernel->border_size(), input, alpha, output);
}

Status CLPReluLayer::validate(const ITensorInfo *input, const ITensorInfo *alpha, const ITensorInfo *output)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::PRELU, input, alpha, output);
}

void CLPReluLayer::run(InputTensorMap inputs, OutputTensorMap outputs, OperatorTensorMap workspace)
{
    InputTensorMap src;
    select_border_input(src, inputs, outputs);
    CLScheduler::get().enqueue_op(_border_handler, src, {});
    ICLOperator::run(inputs, outputs, workspace);
}
} // namespace experimental

struct CLPReluLayer::Impl
{
    const ICLTensor                            *src_0{ nullptr };
    const ICLTensor                            *src_1{ nullptr };
    ICLTensor                                  *dst{ nullptr };
    std::unique_ptr<experimental::CLPReluLayer> op{ nullptr };
};

CLPReluLayer::CLPReluLayer()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLPReluLayer::CLPReluLayer(CLPReluLayer &&) = default;
CLPReluLayer &CLPReluLayer::operator=(CLPReluLayer &&) = default;
CLPReluLayer::~CLPReluLayer()                          = default;

void CLPReluLayer::configure(ICLTensor *input, ICLTensor *alpha, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, alpha, output);
}

void CLPReluLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *alpha, ICLTensor *output)
{
    _impl->src_0 = input;
    _impl->src_1 = alpha;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::CLPReluLayer>();
    _impl->op->configure(compile_context, input->info(), alpha->info(), output->info());
}

Status CLPReluLayer::validate(const ITensorInfo *input, const ITensorInfo *alpha, const ITensorInfo *output)
{
    return experimental::CLPReluLayer::validate(input, alpha, output);
}

void CLPReluLayer::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC_0, _impl->src_0 }, { TensorType::ACL_SRC_1, _impl->src_1 } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}
} // namespace arm_compute
