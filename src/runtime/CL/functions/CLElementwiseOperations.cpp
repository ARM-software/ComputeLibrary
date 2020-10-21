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
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLElementwiseOperationKernel.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
namespace experimental
{
CLArithmeticAddition::CLArithmeticAddition()
{
}

void CLArithmeticAddition::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLSaturatedArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::ADD, input1, input2, output, policy, act_info);
    _kernel = std::move(k);
}

Status CLArithmeticAddition::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    return CLSaturatedArithmeticOperationKernel::validate(ArithmeticOperation::ADD, input1, input2, output, policy, act_info);
}

void CLArithmeticAddition::run(ITensorPack &tensors)
{
    ICLOperator::run(tensors);
}

CLArithmeticSubtraction::CLArithmeticSubtraction()
{
}
void CLArithmeticSubtraction::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, ConvertPolicy policy,
                                        const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLSaturatedArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::SUB, input1, input2, output, policy, act_info);
    _kernel = std::move(k);
}

Status CLArithmeticSubtraction::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(policy);
    return CLSaturatedArithmeticOperationKernel::validate(ArithmeticOperation::SUB, input1, input2, output, policy, act_info);
}

void CLArithmeticSubtraction::run(ITensorPack &tensors)
{
    ICLOperator::run(tensors);
}

CLArithmeticDivision::CLArithmeticDivision()
{
}

void CLArithmeticDivision::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::DIV, input1, input2, output, act_info);
    _kernel = std::move(k);
}

Status CLArithmeticDivision::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::DIV, input1, input2, output, act_info);
}

void CLArithmeticDivision::run(ITensorPack &tensors)
{
    ICLOperator::run(tensors);
}

CLElementwiseMax::CLElementwiseMax()
{
}

void CLElementwiseMax::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::MAX, input1, input2, output, act_info);
    _kernel = std::move(k);
}

Status CLElementwiseMax::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::MAX, input1, input2, output, act_info);
}

void CLElementwiseMax::run(ITensorPack &tensors)
{
    ICLOperator::run(tensors);
}

CLElementwiseMin::CLElementwiseMin()
{
}

void CLElementwiseMin::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::MIN, input1, input2, output, act_info);
    _kernel = std::move(k);
}

Status CLElementwiseMin::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::MIN, input1, input2, output, act_info);
}

void CLElementwiseMin::run(ITensorPack &tensors)
{
    ICLOperator::run(tensors);
}

CLElementwiseSquaredDiff::CLElementwiseSquaredDiff()
{
}

void CLElementwiseSquaredDiff::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::SQUARED_DIFF, input1, input2, output, act_info);
    _kernel = std::move(k);
}

Status CLElementwiseSquaredDiff::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::SQUARED_DIFF, input1, input2, output, act_info);
}

void CLElementwiseSquaredDiff::run(ITensorPack &tensors)
{
    ICLOperator::run(tensors);
}

CLElementwisePower::CLElementwisePower()
{
}

void CLElementwisePower::configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::POWER, input1, input2, output, act_info);
    _kernel = std::move(k);
}

Status CLElementwisePower::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::POWER, input1, input2, output, act_info);
}

void CLElementwisePower::run(ITensorPack &tensors)
{
    ICLOperator::run(tensors);
}
} // namespace experimental

struct CLArithmeticAddition::Impl
{
    const ICLTensor                                    *src_0{ nullptr };
    const ICLTensor                                    *src_1{ nullptr };
    ICLTensor                                          *dst{ nullptr };
    std::unique_ptr<experimental::CLArithmeticAddition> op{ nullptr };
};

CLArithmeticAddition::CLArithmeticAddition()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLArithmeticAddition::CLArithmeticAddition(CLArithmeticAddition &&) = default;
CLArithmeticAddition &CLArithmeticAddition::operator=(CLArithmeticAddition &&) = default;
CLArithmeticAddition::~CLArithmeticAddition()                                  = default;

void CLArithmeticAddition::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, policy, act_info);
}

void CLArithmeticAddition::configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ConvertPolicy policy,
                                     const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::CLArithmeticAddition>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), policy, act_info);
}

Status CLArithmeticAddition::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    return experimental::CLArithmeticAddition::validate(input1, input2, output, policy, act_info);
}

void CLArithmeticAddition::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}

struct CLArithmeticSubtraction::Impl
{
    const ICLTensor                                       *src_0{ nullptr };
    const ICLTensor                                       *src_1{ nullptr };
    ICLTensor                                             *dst{ nullptr };
    std::unique_ptr<experimental::CLArithmeticSubtraction> op{ nullptr };
};

CLArithmeticSubtraction::CLArithmeticSubtraction()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLArithmeticSubtraction::CLArithmeticSubtraction(CLArithmeticSubtraction &&) = default;
CLArithmeticSubtraction &CLArithmeticSubtraction::operator=(CLArithmeticSubtraction &&) = default;
CLArithmeticSubtraction::~CLArithmeticSubtraction()                                     = default;

void CLArithmeticSubtraction::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, policy, act_info);
}

void CLArithmeticSubtraction::configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ConvertPolicy policy,
                                        const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::CLArithmeticSubtraction>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), policy, act_info);
}

Status CLArithmeticSubtraction::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    return experimental::CLArithmeticSubtraction::validate(input1, input2, output, policy, act_info);
}

void CLArithmeticSubtraction::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}

struct CLArithmeticDivision::Impl
{
    const ICLTensor                                    *src_0{ nullptr };
    const ICLTensor                                    *src_1{ nullptr };
    ICLTensor                                          *dst{ nullptr };
    std::unique_ptr<experimental::CLArithmeticDivision> op{ nullptr };
};

CLArithmeticDivision::CLArithmeticDivision()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLArithmeticDivision::CLArithmeticDivision(CLArithmeticDivision &&) = default;
CLArithmeticDivision &CLArithmeticDivision::operator=(CLArithmeticDivision &&) = default;
CLArithmeticDivision::~CLArithmeticDivision()                                  = default;

void CLArithmeticDivision::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLArithmeticDivision::configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::CLArithmeticDivision>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLArithmeticDivision::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return experimental::CLArithmeticDivision::validate(input1, input2, output, act_info);
}

void CLArithmeticDivision::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}

struct CLElementwiseMax::Impl
{
    const ICLTensor                                *src_0{ nullptr };
    const ICLTensor                                *src_1{ nullptr };
    ICLTensor                                      *dst{ nullptr };
    std::unique_ptr<experimental::CLElementwiseMax> op{ nullptr };
};

CLElementwiseMax::CLElementwiseMax()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLElementwiseMax::CLElementwiseMax(CLElementwiseMax &&) = default;
CLElementwiseMax &CLElementwiseMax::operator=(CLElementwiseMax &&) = default;
CLElementwiseMax::~CLElementwiseMax()                              = default;

void CLElementwiseMax::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLElementwiseMax::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::CLElementwiseMax>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLElementwiseMax::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return experimental::CLElementwiseMax::validate(input1, input2, output, act_info);
}

void CLElementwiseMax::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}

struct CLElementwiseMin::Impl
{
    const ICLTensor                                *src_0{ nullptr };
    const ICLTensor                                *src_1{ nullptr };
    ICLTensor                                      *dst{ nullptr };
    std::unique_ptr<experimental::CLElementwiseMin> op{ nullptr };
};

CLElementwiseMin::CLElementwiseMin()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLElementwiseMin::CLElementwiseMin(CLElementwiseMin &&) = default;
CLElementwiseMin &CLElementwiseMin::operator=(CLElementwiseMin &&) = default;
CLElementwiseMin::~CLElementwiseMin()                              = default;

void CLElementwiseMin::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLElementwiseMin::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::CLElementwiseMin>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLElementwiseMin::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return experimental::CLElementwiseMin::validate(input1, input2, output, act_info);
}

void CLElementwiseMin::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}

struct CLElementwiseSquaredDiff::Impl
{
    const ICLTensor                                        *src_0{ nullptr };
    const ICLTensor                                        *src_1{ nullptr };
    ICLTensor                                              *dst{ nullptr };
    std::unique_ptr<experimental::CLElementwiseSquaredDiff> op{ nullptr };
};

CLElementwiseSquaredDiff::CLElementwiseSquaredDiff()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLElementwiseSquaredDiff::CLElementwiseSquaredDiff(CLElementwiseSquaredDiff &&) = default;
CLElementwiseSquaredDiff &CLElementwiseSquaredDiff::operator=(CLElementwiseSquaredDiff &&) = default;
CLElementwiseSquaredDiff::~CLElementwiseSquaredDiff()                                      = default;

void CLElementwiseSquaredDiff::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLElementwiseSquaredDiff::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::CLElementwiseSquaredDiff>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLElementwiseSquaredDiff::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return experimental::CLElementwiseSquaredDiff::validate(input1, input2, output, act_info);
}

void CLElementwiseSquaredDiff::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}

struct CLElementwisePower::Impl
{
    const ICLTensor                                  *src_0{ nullptr };
    const ICLTensor                                  *src_1{ nullptr };
    ICLTensor                                        *dst{ nullptr };
    std::unique_ptr<experimental::CLElementwisePower> op{ nullptr };
};

CLElementwisePower::CLElementwisePower()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
CLElementwisePower::CLElementwisePower(CLElementwisePower &&) = default;
CLElementwisePower &CLElementwisePower::operator=(CLElementwisePower &&) = default;
CLElementwisePower::~CLElementwisePower()                                = default;

void CLElementwisePower::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLElementwisePower::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::CLElementwisePower>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLElementwisePower::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return experimental::CLElementwisePower::validate(input1, input2, output, act_info);
}

void CLElementwisePower::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}
} // namespace arm_compute
