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
#include "arm_compute/runtime/CL/functions/CLElementWiseUnaryLayer.h"

#include "arm_compute/core/CL/kernels/CLElementWiseUnaryLayerKernel.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
namespace experimental
{
void CLRsqrtLayer::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::RSQRT);
    _kernel = std::move(k);
}

Status CLRsqrtLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::RSQRT);
}

MemoryRequirements CLRsqrtLayer::workspace() const
{
    return MemoryRequirements{};
}

void CLExpLayer::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::EXP);
    _kernel = std::move(k);
}

Status CLExpLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::EXP);
}

MemoryRequirements CLExpLayer::workspace() const
{
    return MemoryRequirements{};
}

void CLNegLayer::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::NEG);
    _kernel = std::move(k);
}

Status CLNegLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::NEG);
}

MemoryRequirements CLNegLayer::workspace() const
{
    return MemoryRequirements{};
}

void CLSinLayer::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::SIN);
    _kernel = std::move(k);
}

Status CLSinLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::SIN);
}

MemoryRequirements CLSinLayer::workspace() const
{
    return MemoryRequirements{};
}

void CLAbsLayer::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::ABS);
    _kernel = std::move(k);
}

Status CLAbsLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::ABS);
}

MemoryRequirements CLAbsLayer::workspace() const
{
    return MemoryRequirements{};
}

void CLLogLayer::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::LOG);
    _kernel = std::move(k);
}

Status CLLogLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::LOG);
}

MemoryRequirements CLLogLayer::workspace() const
{
    return MemoryRequirements{};
}

void CLRoundLayer::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::ROUND);
    _kernel = std::move(k);
}

Status CLRoundLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::ROUND);
}

MemoryRequirements CLRoundLayer::workspace() const
{
    return MemoryRequirements{};
}
} // namespace experimental

struct CLRsqrtLayer::Impl
{
    const ICLTensor                            *src{ nullptr };
    ICLTensor                                  *dst{ nullptr };
    std::unique_ptr<experimental::CLRsqrtLayer> op{ nullptr };
};

CLRsqrtLayer::CLRsqrtLayer()
    : _impl(support::cpp14::make_unique<Impl>())
{
}

CLRsqrtLayer::CLRsqrtLayer(CLRsqrtLayer &&) = default;
CLRsqrtLayer &CLRsqrtLayer::operator=(CLRsqrtLayer &&) = default;
CLRsqrtLayer::~CLRsqrtLayer()                          = default;

void CLRsqrtLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLRsqrtLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = arm_compute::support::cpp14::make_unique<experimental::CLRsqrtLayer>();
    _impl->op->configure(compile_context, input->info(), output->info());
}

Status CLRsqrtLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLRsqrtLayer::validate(input, output);
}

void CLRsqrtLayer::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}

struct CLExpLayer::Impl
{
    const ICLTensor                          *src{ nullptr };
    ICLTensor                                *dst{ nullptr };
    std::unique_ptr<experimental::CLExpLayer> op{ nullptr };
};

CLExpLayer::CLExpLayer()
    : _impl(support::cpp14::make_unique<Impl>())
{
}

CLExpLayer::CLExpLayer(CLExpLayer &&) = default;
CLExpLayer &CLExpLayer::operator=(CLExpLayer &&) = default;
CLExpLayer::~CLExpLayer()                        = default;

void CLExpLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLExpLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = arm_compute::support::cpp14::make_unique<experimental::CLExpLayer>();
    _impl->op->configure(compile_context, input->info(), output->info());
}

Status CLExpLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLExpLayer::validate(input, output);
}

void CLExpLayer::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}

struct CLNegLayer::Impl
{
    const ICLTensor                          *src{ nullptr };
    ICLTensor                                *dst{ nullptr };
    std::unique_ptr<experimental::CLNegLayer> op{ nullptr };
};

CLNegLayer::CLNegLayer()
    : _impl(support::cpp14::make_unique<Impl>())
{
}

CLNegLayer::CLNegLayer(CLNegLayer &&) = default;
CLNegLayer &CLNegLayer::operator=(CLNegLayer &&) = default;
CLNegLayer::~CLNegLayer()                        = default;

void CLNegLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLNegLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = arm_compute::support::cpp14::make_unique<experimental::CLNegLayer>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLNegLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLNegLayer::validate(input, output);
}

void CLNegLayer::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}

struct CLSinLayer::Impl
{
    const ICLTensor                          *src{ nullptr };
    ICLTensor                                *dst{ nullptr };
    std::unique_ptr<experimental::CLSinLayer> op{ nullptr };
};

CLSinLayer::CLSinLayer()
    : _impl(support::cpp14::make_unique<Impl>())
{
}

CLSinLayer::CLSinLayer(CLSinLayer &&) = default;
CLSinLayer &CLSinLayer::operator=(CLSinLayer &&) = default;
CLSinLayer::~CLSinLayer()                        = default;

void CLSinLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLSinLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = arm_compute::support::cpp14::make_unique<experimental::CLSinLayer>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLSinLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLSinLayer::validate(input, output);
}

void CLSinLayer::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}

struct CLAbsLayer::Impl
{
    const ICLTensor                          *src{ nullptr };
    ICLTensor                                *dst{ nullptr };
    std::unique_ptr<experimental::CLAbsLayer> op{ nullptr };
};

CLAbsLayer::CLAbsLayer()
    : _impl(support::cpp14::make_unique<Impl>())
{
}

CLAbsLayer::CLAbsLayer(CLAbsLayer &&) = default;
CLAbsLayer &CLAbsLayer::operator=(CLAbsLayer &&) = default;
CLAbsLayer::~CLAbsLayer()                        = default;

void CLAbsLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLAbsLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = arm_compute::support::cpp14::make_unique<experimental::CLAbsLayer>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLAbsLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLAbsLayer::validate(input, output);
}

void CLAbsLayer::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}

struct CLLogLayer::Impl
{
    const ICLTensor                          *src{ nullptr };
    ICLTensor                                *dst{ nullptr };
    std::unique_ptr<experimental::CLLogLayer> op{ nullptr };
};

CLLogLayer::CLLogLayer()
    : _impl(support::cpp14::make_unique<Impl>())
{
}

CLLogLayer::CLLogLayer(CLLogLayer &&) = default;
CLLogLayer &CLLogLayer::operator=(CLLogLayer &&) = default;
CLLogLayer::~CLLogLayer()                        = default;

void CLLogLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLLogLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = arm_compute::support::cpp14::make_unique<experimental::CLLogLayer>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLLogLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLLogLayer::validate(input, output);
}

void CLLogLayer::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}

struct CLRoundLayer::Impl
{
    const ICLTensor                            *src{ nullptr };
    ICLTensor                                  *dst{ nullptr };
    std::unique_ptr<experimental::CLRoundLayer> op{ nullptr };
};

CLRoundLayer::CLRoundLayer()
    : _impl(support::cpp14::make_unique<Impl>())
{
}

CLRoundLayer::CLRoundLayer(CLRoundLayer &&) = default;
CLRoundLayer &CLRoundLayer::operator=(CLRoundLayer &&) = default;
CLRoundLayer::~CLRoundLayer()                          = default;

void CLRoundLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLRoundLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = arm_compute::support::cpp14::make_unique<experimental::CLRoundLayer>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLRoundLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLRoundLayer::validate(input, output);
}

void CLRoundLayer::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}
} // namespace arm_compute
