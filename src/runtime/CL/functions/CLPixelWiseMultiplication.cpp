/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLPixelWiseMultiplicationKernel.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
namespace
{
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
CLPixelWiseMultiplication::CLPixelWiseMultiplication()
    : _border_handler()
{
}

void CLPixelWiseMultiplication::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, float scale,
                                          ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLPixelWiseMultiplicationKernel>();
    k->configure(compile_context, input1, input2, output, scale, overflow_policy, rounding_policy, act_info);
    _kernel = std::move(k);

    if(output->dimension(0) > 1)
    {
        ITensorInfo *broadcasted_info = (input1->dimension(0) == 1) ? input1 : input2;

        if(broadcasted_info->dimension(0) == 1)
        {
            _border_handler.configure(compile_context, broadcasted_info, _kernel->border_size(), BorderMode::REPLICATE);
        }
    }
}

Status CLPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale,
                                           ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    return CLPixelWiseMultiplicationKernel::validate(input1, input2, output, scale, overflow_policy, rounding_policy, act_info);
}

void CLPixelWiseMultiplication::run(InputTensorMap inputs, OutputTensorMap outputs, OperatorTensorMap workspace)
{
    InputTensorMap src;
    select_border_input(src, inputs, outputs);
    CLScheduler::get().enqueue_op(_border_handler, src, {});
    ICLOperator::run(inputs, outputs, workspace);
}

CLComplexPixelWiseMultiplication::CLComplexPixelWiseMultiplication()
    : _border_handler()
{
}

void CLComplexPixelWiseMultiplication::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLComplexPixelWiseMultiplicationKernel>();
    k->configure(compile_context, input1, input2, output, act_info);
    _kernel = std::move(k);

    if(output->dimension(0) > 1)
    {
        ITensorInfo *broadcasted_info = (input1->dimension(0) == 1) ? input1 : input2;

        if(broadcasted_info->dimension(0) == 1)
        {
            _border_handler.configure(compile_context, broadcasted_info, _kernel->border_size(), BorderMode::REPLICATE);
        }
    }
}

Status CLComplexPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLComplexPixelWiseMultiplicationKernel::validate(input1, input2, output, act_info);
}

void CLComplexPixelWiseMultiplication::run(InputTensorMap inputs, OutputTensorMap outputs, OperatorTensorMap workspace)
{
    InputTensorMap src;
    select_border_input(src, inputs, outputs);
    CLScheduler::get().enqueue_op(_border_handler, src, {});
    ICLOperator::run(inputs, outputs, workspace);
}
} // namespace experimental

struct CLPixelWiseMultiplication::Impl
{
    const ICLTensor                                         *src_0{ nullptr };
    const ICLTensor                                         *src_1{ nullptr };
    ICLTensor                                               *dst{ nullptr };
    std::unique_ptr<experimental::CLPixelWiseMultiplication> op{ nullptr };
};

CLPixelWiseMultiplication::CLPixelWiseMultiplication()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLPixelWiseMultiplication::CLPixelWiseMultiplication(CLPixelWiseMultiplication &&) = default;
CLPixelWiseMultiplication &CLPixelWiseMultiplication::operator=(CLPixelWiseMultiplication &&) = default;
CLPixelWiseMultiplication::~CLPixelWiseMultiplication()                                       = default;

void CLPixelWiseMultiplication::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, float scale,
                                          ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, scale, overflow_policy, rounding_policy, act_info);
}

void CLPixelWiseMultiplication::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, float scale,
                                          ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::CLPixelWiseMultiplication>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), scale, overflow_policy, rounding_policy, act_info);
}

Status CLPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale,
                                           ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    return experimental::CLPixelWiseMultiplication::validate(input1, input2, output, scale, overflow_policy, rounding_policy, act_info);
}

void CLPixelWiseMultiplication::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC_0, _impl->src_0 }, { TensorType::ACL_SRC_1, _impl->src_1 } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}

struct CLComplexPixelWiseMultiplication::Impl
{
    const ICLTensor                                                *src_0{ nullptr };
    const ICLTensor                                                *src_1{ nullptr };
    ICLTensor                                                      *dst{ nullptr };
    std::unique_ptr<experimental::CLComplexPixelWiseMultiplication> op{ nullptr };
};

CLComplexPixelWiseMultiplication::CLComplexPixelWiseMultiplication()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLComplexPixelWiseMultiplication::CLComplexPixelWiseMultiplication(CLComplexPixelWiseMultiplication &&) = default;
CLComplexPixelWiseMultiplication &CLComplexPixelWiseMultiplication::operator=(CLComplexPixelWiseMultiplication &&) = default;
CLComplexPixelWiseMultiplication::~CLComplexPixelWiseMultiplication()                                              = default;

void CLComplexPixelWiseMultiplication::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLComplexPixelWiseMultiplication::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::CLComplexPixelWiseMultiplication>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLComplexPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return experimental::CLComplexPixelWiseMultiplication::validate(input1, input2, output, act_info);
}

void CLComplexPixelWiseMultiplication::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC_0, _impl->src_0 }, { TensorType::ACL_SRC_1, _impl->src_1 } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}
} // namespace arm_compute
