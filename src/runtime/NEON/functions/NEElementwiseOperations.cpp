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
#include "arm_compute/runtime/NEON/functions/NEElementwiseOperations.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuElementwise.h"

#include <utility>

namespace arm_compute
{
struct NEElementwiseMax::Impl
{
    const ITensor                          *src_0{nullptr};
    const ITensor                          *src_1{nullptr};
    ITensor                                *dst{nullptr};
    std::unique_ptr<cpu::CpuElementwiseMax> op{nullptr};
};

NEElementwiseMax::NEElementwiseMax() : _impl(std::make_unique<Impl>())
{
}
NEElementwiseMax::NEElementwiseMax(NEElementwiseMax &&)            = default;
NEElementwiseMax &NEElementwiseMax::operator=(NEElementwiseMax &&) = default;
NEElementwiseMax::~NEElementwiseMax()                              = default;

void NEElementwiseMax::configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = std::make_unique<cpu::CpuElementwiseMax>();
    _impl->op->configure(input1->info(), input2->info(), output->info());
}

Status NEElementwiseMax::validate(const ITensorInfo         *input1,
                                  const ITensorInfo         *input2,
                                  const ITensorInfo         *output,
                                  const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return cpu::CpuElementwiseMax::validate(input1, input2, output);
}

void NEElementwiseMax::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct NEElementwiseMin::Impl
{
    const ITensor                          *src_0{nullptr};
    const ITensor                          *src_1{nullptr};
    ITensor                                *dst{nullptr};
    std::unique_ptr<cpu::CpuElementwiseMin> op{nullptr};
};

NEElementwiseMin::NEElementwiseMin() : _impl(std::make_unique<Impl>())
{
}
NEElementwiseMin::NEElementwiseMin(NEElementwiseMin &&)            = default;
NEElementwiseMin &NEElementwiseMin::operator=(NEElementwiseMin &&) = default;
NEElementwiseMin::~NEElementwiseMin()                              = default;

void NEElementwiseMin::configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = std::make_unique<cpu::CpuElementwiseMin>();
    _impl->op->configure(input1->info(), input2->info(), output->info());
}

Status NEElementwiseMin::validate(const ITensorInfo         *input1,
                                  const ITensorInfo         *input2,
                                  const ITensorInfo         *output,
                                  const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return cpu::CpuElementwiseMin::validate(input1, input2, output);
}

void NEElementwiseMin::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct NEElementwiseSquaredDiff::Impl
{
    const ITensor                                  *src_0{nullptr};
    const ITensor                                  *src_1{nullptr};
    ITensor                                        *dst{nullptr};
    std::unique_ptr<cpu::CpuElementwiseSquaredDiff> op{nullptr};
};

NEElementwiseSquaredDiff::NEElementwiseSquaredDiff() : _impl(std::make_unique<Impl>())
{
}
NEElementwiseSquaredDiff::NEElementwiseSquaredDiff(NEElementwiseSquaredDiff &&)            = default;
NEElementwiseSquaredDiff &NEElementwiseSquaredDiff::operator=(NEElementwiseSquaredDiff &&) = default;
NEElementwiseSquaredDiff::~NEElementwiseSquaredDiff()                                      = default;

void NEElementwiseSquaredDiff::configure(ITensor                   *input1,
                                         ITensor                   *input2,
                                         ITensor                   *output,
                                         const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = std::make_unique<cpu::CpuElementwiseSquaredDiff>();
    _impl->op->configure(input1->info(), input2->info(), output->info());
}

Status NEElementwiseSquaredDiff::validate(const ITensorInfo         *input1,
                                          const ITensorInfo         *input2,
                                          const ITensorInfo         *output,
                                          const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return cpu::CpuElementwiseSquaredDiff::validate(input1, input2, output);
}

void NEElementwiseSquaredDiff::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct NEElementwiseDivision::Impl
{
    const ITensor                               *src_0{nullptr};
    const ITensor                               *src_1{nullptr};
    ITensor                                     *dst{nullptr};
    std::unique_ptr<cpu::CpuElementwiseDivision> op{nullptr};
};

NEElementwiseDivision::NEElementwiseDivision() : _impl(std::make_unique<Impl>())
{
}
NEElementwiseDivision::NEElementwiseDivision(NEElementwiseDivision &&)            = default;
NEElementwiseDivision &NEElementwiseDivision::operator=(NEElementwiseDivision &&) = default;
NEElementwiseDivision::~NEElementwiseDivision()                                   = default;

void NEElementwiseDivision::configure(ITensor                   *input1,
                                      ITensor                   *input2,
                                      ITensor                   *output,
                                      const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = std::make_unique<cpu::CpuElementwiseDivision>();
    _impl->op->configure(input1->info(), input2->info(), output->info());
}

Status NEElementwiseDivision::validate(const ITensorInfo         *input1,
                                       const ITensorInfo         *input2,
                                       const ITensorInfo         *output,
                                       const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return cpu::CpuElementwiseDivision::validate(input1, input2, output);
}

void NEElementwiseDivision::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct NEElementwisePower::Impl
{
    const ITensor                            *src_0{nullptr};
    const ITensor                            *src_1{nullptr};
    ITensor                                  *dst{nullptr};
    std::unique_ptr<cpu::CpuElementwisePower> op{nullptr};
};

NEElementwisePower::NEElementwisePower() : _impl(std::make_unique<Impl>())
{
}
NEElementwisePower::NEElementwisePower(NEElementwisePower &&)            = default;
NEElementwisePower &NEElementwisePower::operator=(NEElementwisePower &&) = default;
NEElementwisePower::~NEElementwisePower()                                = default;

void NEElementwisePower::configure(ITensor                   *input1,
                                   ITensor                   *input2,
                                   ITensor                   *output,
                                   const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = std::make_unique<cpu::CpuElementwisePower>();
    _impl->op->configure(input1->info(), input2->info(), output->info());
}

Status NEElementwisePower::validate(const ITensorInfo         *input1,
                                    const ITensorInfo         *input2,
                                    const ITensorInfo         *output,
                                    const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return cpu::CpuElementwisePower::validate(input1, input2, output);
}

void NEElementwisePower::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

template <ComparisonOperation COP>
struct NEElementwiseComparisonStatic<COP>::Impl
{
    const ITensor                                            *src_0{nullptr};
    const ITensor                                            *src_1{nullptr};
    ITensor                                                  *dst{nullptr};
    std::unique_ptr<cpu::CpuElementwiseComparisonStatic<COP>> op{nullptr};
};

template <ComparisonOperation COP>
NEElementwiseComparisonStatic<COP>::NEElementwiseComparisonStatic() : _impl(std::make_unique<Impl>())
{
}
template <ComparisonOperation COP>
NEElementwiseComparisonStatic<COP>::NEElementwiseComparisonStatic(NEElementwiseComparisonStatic &&) = default;
template <ComparisonOperation COP>
NEElementwiseComparisonStatic<COP> &
NEElementwiseComparisonStatic<COP>::operator=(NEElementwiseComparisonStatic &&) = default;
template <ComparisonOperation COP>
NEElementwiseComparisonStatic<COP>::~NEElementwiseComparisonStatic() = default;

template <ComparisonOperation COP>
void NEElementwiseComparisonStatic<COP>::configure(ITensor *input1, ITensor *input2, ITensor *output)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = std::make_unique<cpu::CpuElementwiseComparisonStatic<COP>>();
    _impl->op->configure(input1->info(), input2->info(), output->info());
}

template <ComparisonOperation COP>
Status NEElementwiseComparisonStatic<COP>::validate(const ITensorInfo *input1,
                                                    const ITensorInfo *input2,
                                                    const ITensorInfo *output)
{
    return cpu::CpuElementwiseComparisonStatic<COP>::validate(input1, input2, output);
}

template <ComparisonOperation COP>
void NEElementwiseComparisonStatic<COP>::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct NEElementwiseComparison::Impl
{
    const ITensor                                 *src_0{nullptr};
    const ITensor                                 *src_1{nullptr};
    ITensor                                       *dst{nullptr};
    std::unique_ptr<cpu::CpuElementwiseComparison> op{nullptr};
};

NEElementwiseComparison::NEElementwiseComparison() : _impl(std::make_unique<Impl>())
{
}
NEElementwiseComparison::NEElementwiseComparison(NEElementwiseComparison &&)            = default;
NEElementwiseComparison &NEElementwiseComparison::operator=(NEElementwiseComparison &&) = default;
NEElementwiseComparison::~NEElementwiseComparison()                                     = default;

void NEElementwiseComparison::configure(ITensor *input1, ITensor *input2, ITensor *output, ComparisonOperation op)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = std::make_unique<cpu::CpuElementwiseComparison>();
    _impl->op->configure(input1->info(), input2->info(), output->info(), op);
}

Status NEElementwiseComparison::validate(const ITensorInfo  *input1,
                                         const ITensorInfo  *input2,
                                         const ITensorInfo  *output,
                                         ComparisonOperation op)
{
    return cpu::CpuElementwiseComparison::validate(input1, input2, output, op);
}

void NEElementwiseComparison::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

// Supported Specializations
template class NEElementwiseComparisonStatic<ComparisonOperation::Equal>;
template class NEElementwiseComparisonStatic<ComparisonOperation::NotEqual>;
template class NEElementwiseComparisonStatic<ComparisonOperation::Greater>;
template class NEElementwiseComparisonStatic<ComparisonOperation::GreaterEqual>;
template class NEElementwiseComparisonStatic<ComparisonOperation::Less>;
template class NEElementwiseComparisonStatic<ComparisonOperation::LessEqual>;
} // namespace arm_compute
