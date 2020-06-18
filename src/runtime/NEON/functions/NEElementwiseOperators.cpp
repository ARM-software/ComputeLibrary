/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/functions/NEElementwiseOperations.h"
#include <arm_compute/core/NEON/kernels/NEElementwiseOperationKernel.h>

#include "arm_compute/core/ITensor.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
namespace experimental
{
void NEElementwiseMax::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    auto k = arm_compute::support::cpp14::make_unique<NEArithmeticOperationKernel>();
    k->configure(ArithmeticOperation::MAX, input1, input2, output);
    _kernel = std::move(k);
}

Status NEElementwiseMax::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return NEArithmeticOperationKernel::validate(ArithmeticOperation::MAX, input1, input2, output);
}

MemoryRequirements NEElementwiseMax::workspace() const
{
    return MemoryRequirements{};
}

void NEElementwiseMin::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    auto k = arm_compute::support::cpp14::make_unique<NEArithmeticOperationKernel>();
    k->configure(ArithmeticOperation::MIN, input1, input2, output);
    _kernel = std::move(k);
}

Status NEElementwiseMin::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return NEArithmeticOperationKernel::validate(ArithmeticOperation::MIN, input1, input2, output);
}

MemoryRequirements NEElementwiseMin::workspace() const
{
    return MemoryRequirements{};
}

void NEElementwiseSquaredDiff::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    auto k = arm_compute::support::cpp14::make_unique<NEArithmeticOperationKernel>();
    k->configure(ArithmeticOperation::SQUARED_DIFF, input1, input2, output);
    _kernel = std::move(k);
}

Status NEElementwiseSquaredDiff::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return NEArithmeticOperationKernel::validate(ArithmeticOperation::SQUARED_DIFF, input1, input2, output);
}

MemoryRequirements NEElementwiseSquaredDiff::workspace() const
{
    return MemoryRequirements{};
}

void NEElementwiseDivision::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    auto k = arm_compute::support::cpp14::make_unique<NEDivisionOperationKernel>();
    k->configure(input1, input2, output);
    _kernel = std::move(k);
}

Status NEElementwiseDivision::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return NEDivisionOperationKernel::validate(input1, input2, output);
}

MemoryRequirements NEElementwiseDivision::workspace() const
{
    return MemoryRequirements{};
}

void NEElementwisePower::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    auto k = arm_compute::support::cpp14::make_unique<NEPowerOperationKernel>();
    k->configure(input1, input2, output);
    _kernel = std::move(k);
}

Status NEElementwisePower::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return NEPowerOperationKernel::validate(input1, input2, output);
}

MemoryRequirements NEElementwisePower::workspace() const
{
    return MemoryRequirements{};
}

template <ComparisonOperation COP>
void NEElementwiseComparisonStatic<COP>::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<NEComparisonOperationKernel>();
    k->configure(COP, input1, input2, output);
    _kernel = std::move(k);
}

template <ComparisonOperation COP>
Status NEElementwiseComparisonStatic<COP>::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return NEComparisonOperationKernel::validate(COP, input1, input2, output);
}

template <ComparisonOperation COP>
MemoryRequirements            NEElementwiseComparisonStatic<COP>::workspace() const
{
    return MemoryRequirements{};
}

void NEElementwiseComparison::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, ComparisonOperation op)
{
    auto k = arm_compute::support::cpp14::make_unique<NEComparisonOperationKernel>();
    k->configure(op, input1, input2, output);
    _kernel = std::move(k);
}

Status NEElementwiseComparison::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ComparisonOperation op)
{
    return NEComparisonOperationKernel::validate(op, input1, input2, output);
}

MemoryRequirements NEElementwiseComparison::workspace() const
{
    return MemoryRequirements{};
}

// Supported Specializations
template class NEElementwiseComparisonStatic<ComparisonOperation::Equal>;
template class NEElementwiseComparisonStatic<ComparisonOperation::NotEqual>;
template class NEElementwiseComparisonStatic<ComparisonOperation::Greater>;
template class NEElementwiseComparisonStatic<ComparisonOperation::GreaterEqual>;
template class NEElementwiseComparisonStatic<ComparisonOperation::Less>;
template class NEElementwiseComparisonStatic<ComparisonOperation::LessEqual>;
} // namespace experimental

struct NEElementwiseMax::Impl
{
    const ITensor                                  *src_0{ nullptr };
    const ITensor                                  *src_1{ nullptr };
    ITensor                                        *dst{ nullptr };
    std::unique_ptr<experimental::NEElementwiseMax> op{ nullptr };
};

NEElementwiseMax::NEElementwiseMax()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
NEElementwiseMax::NEElementwiseMax(NEElementwiseMax &&) = default;
NEElementwiseMax &NEElementwiseMax::operator=(NEElementwiseMax &&) = default;
NEElementwiseMax::~NEElementwiseMax()                              = default;

void NEElementwiseMax::configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::NEElementwiseMax>();
    _impl->op->configure(input1->info(), input2->info(), output->info(), act_info);
}

Status NEElementwiseMax::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return experimental::NEElementwiseMax::validate(input1, input2, output, act_info);
}

void NEElementwiseMax::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC_0, _impl->src_0 }, { TensorType::ACL_SRC_1, _impl->src_1 } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };
    _impl->op->run(src, dst, {});
}

struct NEElementwiseMin::Impl
{
    const ITensor                                  *src_0{ nullptr };
    const ITensor                                  *src_1{ nullptr };
    ITensor                                        *dst{ nullptr };
    std::unique_ptr<experimental::NEElementwiseMin> op{ nullptr };
};

NEElementwiseMin::NEElementwiseMin()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
NEElementwiseMin::NEElementwiseMin(NEElementwiseMin &&) = default;
NEElementwiseMin &NEElementwiseMin::operator=(NEElementwiseMin &&) = default;
NEElementwiseMin::~NEElementwiseMin()                              = default;

void NEElementwiseMin::configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::NEElementwiseMin>();
    _impl->op->configure(input1->info(), input2->info(), output->info(), act_info);
}

Status NEElementwiseMin::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return experimental::NEElementwiseMin::validate(input1, input2, output, act_info);
}

void NEElementwiseMin::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC_0, _impl->src_0 }, { TensorType::ACL_SRC_1, _impl->src_1 } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };
    _impl->op->run(src, dst, {});
}

struct NEElementwiseSquaredDiff::Impl
{
    const ITensor                                          *src_0{ nullptr };
    const ITensor                                          *src_1{ nullptr };
    ITensor                                                *dst{ nullptr };
    std::unique_ptr<experimental::NEElementwiseSquaredDiff> op{ nullptr };
};

NEElementwiseSquaredDiff::NEElementwiseSquaredDiff()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
NEElementwiseSquaredDiff::NEElementwiseSquaredDiff(NEElementwiseSquaredDiff &&) = default;
NEElementwiseSquaredDiff &NEElementwiseSquaredDiff::operator=(NEElementwiseSquaredDiff &&) = default;
NEElementwiseSquaredDiff::~NEElementwiseSquaredDiff()                                      = default;

void NEElementwiseSquaredDiff::configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::NEElementwiseSquaredDiff>();
    _impl->op->configure(input1->info(), input2->info(), output->info(), act_info);
}

Status NEElementwiseSquaredDiff::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return experimental::NEElementwiseSquaredDiff::validate(input1, input2, output, act_info);
}

void NEElementwiseSquaredDiff::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC_0, _impl->src_0 }, { TensorType::ACL_SRC_1, _impl->src_1 } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };
    _impl->op->run(src, dst, {});
}

struct NEElementwiseDivision::Impl
{
    const ITensor                                       *src_0{ nullptr };
    const ITensor                                       *src_1{ nullptr };
    ITensor                                             *dst{ nullptr };
    std::unique_ptr<experimental::NEElementwiseDivision> op{ nullptr };
};

NEElementwiseDivision::NEElementwiseDivision()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
NEElementwiseDivision::NEElementwiseDivision(NEElementwiseDivision &&) = default;
NEElementwiseDivision &NEElementwiseDivision::operator=(NEElementwiseDivision &&) = default;
NEElementwiseDivision::~NEElementwiseDivision()                                   = default;

void NEElementwiseDivision::configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::NEElementwiseDivision>();
    _impl->op->configure(input1->info(), input2->info(), output->info(), act_info);
}

Status NEElementwiseDivision::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return experimental::NEElementwiseDivision::validate(input1, input2, output, act_info);
}

void NEElementwiseDivision::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC_0, _impl->src_0 }, { TensorType::ACL_SRC_1, _impl->src_1 } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };
    _impl->op->run(src, dst, {});
}

struct NEElementwisePower::Impl
{
    const ITensor                                    *src_0{ nullptr };
    const ITensor                                    *src_1{ nullptr };
    ITensor                                          *dst{ nullptr };
    std::unique_ptr<experimental::NEElementwisePower> op{ nullptr };
};

NEElementwisePower::NEElementwisePower()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
NEElementwisePower::NEElementwisePower(NEElementwisePower &&) = default;
NEElementwisePower &NEElementwisePower::operator=(NEElementwisePower &&) = default;
NEElementwisePower::~NEElementwisePower()                                = default;

void NEElementwisePower::configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::NEElementwisePower>();
    _impl->op->configure(input1->info(), input2->info(), output->info(), act_info);
}

Status NEElementwisePower::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return experimental::NEElementwisePower::validate(input1, input2, output, act_info);
}

void NEElementwisePower::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC_0, _impl->src_0 }, { TensorType::ACL_SRC_1, _impl->src_1 } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };
    _impl->op->run(src, dst, {});
}

template <ComparisonOperation COP>
struct NEElementwiseComparisonStatic<COP>::Impl
{
    const ITensor                                                    *src_0{ nullptr };
    const ITensor                                                    *src_1{ nullptr };
    ITensor                                                          *dst{ nullptr };
    std::unique_ptr<experimental::NEElementwiseComparisonStatic<COP>> op{ nullptr };
};

template <ComparisonOperation COP>
NEElementwiseComparisonStatic<COP>::NEElementwiseComparisonStatic()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
template <ComparisonOperation COP>
NEElementwiseComparisonStatic<COP>::NEElementwiseComparisonStatic(NEElementwiseComparisonStatic &&) = default;
template <ComparisonOperation       COP>
NEElementwiseComparisonStatic<COP> &NEElementwiseComparisonStatic<COP>::operator=(NEElementwiseComparisonStatic &&) = default;
template <ComparisonOperation       COP>
NEElementwiseComparisonStatic<COP>::~NEElementwiseComparisonStatic() = default;

template <ComparisonOperation COP>
void NEElementwiseComparisonStatic<COP>::configure(ITensor *input1, ITensor *input2, ITensor *output)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::NEElementwiseComparisonStatic<COP>>();
    _impl->op->configure(input1->info(), input2->info(), output->info());
}

template <ComparisonOperation COP>
Status NEElementwiseComparisonStatic<COP>::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return experimental::NEElementwiseComparisonStatic<COP>::validate(input1, input2, output);
}

template <ComparisonOperation COP>
void                          NEElementwiseComparisonStatic<COP>::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC_0, _impl->src_0 }, { TensorType::ACL_SRC_1, _impl->src_1 } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };
    _impl->op->run(src, dst, {});
}

struct NEElementwiseComparison::Impl
{
    const ITensor                                         *src_0{ nullptr };
    const ITensor                                         *src_1{ nullptr };
    ITensor                                               *dst{ nullptr };
    std::unique_ptr<experimental::NEElementwiseComparison> op{ nullptr };
};

NEElementwiseComparison::NEElementwiseComparison()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
NEElementwiseComparison::NEElementwiseComparison(NEElementwiseComparison &&) = default;
NEElementwiseComparison &NEElementwiseComparison::operator=(NEElementwiseComparison &&) = default;
NEElementwiseComparison::~NEElementwiseComparison()                                     = default;

void NEElementwiseComparison::configure(ITensor *input1, ITensor *input2, ITensor *output, ComparisonOperation op)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = arm_compute::support::cpp14::make_unique<experimental::NEElementwiseComparison>();
    _impl->op->configure(input1->info(), input2->info(), output->info(), op);
}

Status NEElementwiseComparison::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ComparisonOperation op)
{
    return experimental::NEElementwiseComparison::validate(input1, input2, output, op);
}

void NEElementwiseComparison::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC_0, _impl->src_0 }, { TensorType::ACL_SRC_1, _impl->src_1 } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };
    _impl->op->run(src, dst, {});
}

// Supported Specializations
template class NEElementwiseComparisonStatic<ComparisonOperation::Equal>;
template class NEElementwiseComparisonStatic<ComparisonOperation::NotEqual>;
template class NEElementwiseComparisonStatic<ComparisonOperation::Greater>;
template class NEElementwiseComparisonStatic<ComparisonOperation::GreaterEqual>;
template class NEElementwiseComparisonStatic<ComparisonOperation::Less>;
template class NEElementwiseComparisonStatic<ComparisonOperation::LessEqual>;
} // namespace arm_compute
