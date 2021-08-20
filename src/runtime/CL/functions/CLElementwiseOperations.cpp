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
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"

#include "src/gpu/cl/operators/ClAdd.h"
#include "src/gpu/cl/operators/ClElementwiseOperations.h"
#include "src/gpu/cl/operators/ClSub.h"

namespace arm_compute
{
struct CLArithmeticAddition::Impl
{
    const ICLTensor               *src_0{ nullptr };
    const ICLTensor               *src_1{ nullptr };
    ICLTensor                     *dst{ nullptr };
    std::unique_ptr<opencl::ClAdd> op{ nullptr };
};

CLArithmeticAddition::CLArithmeticAddition()
    : _impl(std::make_unique<Impl>())
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
    _impl->op    = std::make_unique<opencl::ClAdd>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), policy, act_info);
}

Status CLArithmeticAddition::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    return opencl::ClAdd::validate(input1, input2, output, policy, act_info);
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
    const ICLTensor               *src_0{ nullptr };
    const ICLTensor               *src_1{ nullptr };
    ICLTensor                     *dst{ nullptr };
    std::unique_ptr<opencl::ClSub> op{ nullptr };
};

CLArithmeticSubtraction::CLArithmeticSubtraction()
    : _impl(std::make_unique<Impl>())
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
    _impl->op    = std::make_unique<opencl::ClSub>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), policy, act_info);
}

Status CLArithmeticSubtraction::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    return opencl::ClSub::validate(input1, input2, output, policy, act_info);
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
    const ICLTensor                               *src_0{ nullptr };
    const ICLTensor                               *src_1{ nullptr };
    ICLTensor                                     *dst{ nullptr };
    std::unique_ptr<opencl::ClElementwiseDivision> op{ nullptr };
};

CLArithmeticDivision::CLArithmeticDivision()
    : _impl(std::make_unique<Impl>())
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
    _impl->op    = std::make_unique<opencl::ClElementwiseDivision>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLArithmeticDivision::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return opencl::ClElementwiseDivision::validate(input1, input2, output, act_info);
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
    const ICLTensor                          *src_0{ nullptr };
    const ICLTensor                          *src_1{ nullptr };
    ICLTensor                                *dst{ nullptr };
    std::unique_ptr<opencl::ClElementwiseMax> op{ nullptr };
};

CLElementwiseMax::CLElementwiseMax()
    : _impl(std::make_unique<Impl>())
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
    _impl->op    = std::make_unique<opencl::ClElementwiseMax>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLElementwiseMax::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return opencl::ClElementwiseMax::validate(input1, input2, output, act_info);
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
    const ICLTensor                          *src_0{ nullptr };
    const ICLTensor                          *src_1{ nullptr };
    ICLTensor                                *dst{ nullptr };
    std::unique_ptr<opencl::ClElementwiseMin> op{ nullptr };
};

CLElementwiseMin::CLElementwiseMin()
    : _impl(std::make_unique<Impl>())
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
    _impl->op    = std::make_unique<opencl::ClElementwiseMin>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLElementwiseMin::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return opencl::ClElementwiseMin::validate(input1, input2, output, act_info);
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
    const ICLTensor                                  *src_0{ nullptr };
    const ICLTensor                                  *src_1{ nullptr };
    ICLTensor                                        *dst{ nullptr };
    std::unique_ptr<opencl::ClElementwiseSquaredDiff> op{ nullptr };
};

CLElementwiseSquaredDiff::CLElementwiseSquaredDiff()
    : _impl(std::make_unique<Impl>())
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
    _impl->op    = std::make_unique<opencl::ClElementwiseSquaredDiff>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLElementwiseSquaredDiff::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return opencl::ClElementwiseSquaredDiff::validate(input1, input2, output, act_info);
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
    const ICLTensor                            *src_0{ nullptr };
    const ICLTensor                            *src_1{ nullptr };
    ICLTensor                                  *dst{ nullptr };
    std::unique_ptr<opencl::ClElementwisePower> op{ nullptr };
};

CLElementwisePower::CLElementwisePower()
    : _impl(std::make_unique<Impl>())
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
    _impl->op    = std::make_unique<opencl::ClElementwisePower>();
    _impl->op->configure(compile_context, input1->info(), input2->info(), output->info(), act_info);
}

Status CLElementwisePower::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return opencl::ClElementwisePower::validate(input1, input2, output, act_info);
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
