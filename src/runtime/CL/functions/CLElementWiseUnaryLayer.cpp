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

#include "src/core/CL/kernels/CLElementWiseUnaryLayerKernel.h"

#include <utility>

namespace arm_compute
{
namespace experimental
{
void CLRsqrt::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = std::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::RSQRT);
    _kernel = std::move(k);
}

Status CLRsqrt::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::RSQRT);
}

void CLExp::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = std::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::EXP);
    _kernel = std::move(k);
}

Status CLExp::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::EXP);
}

void CLNeg::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = std::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::NEG);
    _kernel = std::move(k);
}

Status CLNeg::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::NEG);
}

void CLSin::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = std::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::SIN);
    _kernel = std::move(k);
}

Status CLSin::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::SIN);
}

void CLAbs::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = std::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::ABS);
    _kernel = std::move(k);
}

Status CLAbs::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::ABS);
}

void CLLog::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = std::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::LOG);
    _kernel = std::move(k);
}

Status CLLog::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::LOG);
}

void CLRound::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output)
{
    auto k = std::make_unique<CLElementWiseUnaryLayerKernel>();
    k->configure(compile_context, input, output, ElementWiseUnary::ROUND);
    _kernel = std::move(k);
}

Status CLRound::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::CLElementWiseUnaryLayerKernel::validate(input, output, ElementWiseUnary::ROUND);
}
} // namespace experimental

struct CLRsqrtLayer::Impl
{
    const ICLTensor                       *src{ nullptr };
    ICLTensor                             *dst{ nullptr };
    std::unique_ptr<experimental::CLRsqrt> op{ nullptr };
};

CLRsqrtLayer::CLRsqrtLayer()
    : _impl(std::make_unique<Impl>())
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
    _impl->op  = std::make_unique<experimental::CLRsqrt>();
    _impl->op->configure(compile_context, input->info(), output->info());
}

Status CLRsqrtLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLRsqrt::validate(input, output);
}

void CLRsqrtLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct CLExpLayer::Impl
{
    const ICLTensor                     *src{ nullptr };
    ICLTensor                           *dst{ nullptr };
    std::unique_ptr<experimental::CLExp> op{ nullptr };
};

CLExpLayer::CLExpLayer()
    : _impl(std::make_unique<Impl>())
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
    _impl->op  = std::make_unique<experimental::CLExp>();
    _impl->op->configure(compile_context, input->info(), output->info());
}

Status CLExpLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLExp::validate(input, output);
}

void CLExpLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct CLNegLayer::Impl
{
    const ICLTensor                     *src{ nullptr };
    ICLTensor                           *dst{ nullptr };
    std::unique_ptr<experimental::CLNeg> op{ nullptr };
};

CLNegLayer::CLNegLayer()
    : _impl(std::make_unique<Impl>())
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
    _impl->op  = std::make_unique<experimental::CLNeg>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLNegLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLNeg::validate(input, output);
}

void CLNegLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct CLSinLayer::Impl
{
    const ICLTensor                     *src{ nullptr };
    ICLTensor                           *dst{ nullptr };
    std::unique_ptr<experimental::CLSin> op{ nullptr };
};

CLSinLayer::CLSinLayer()
    : _impl(std::make_unique<Impl>())
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
    _impl->op  = std::make_unique<experimental::CLSin>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLSinLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLSin::validate(input, output);
}

void CLSinLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct CLAbsLayer::Impl
{
    const ICLTensor                     *src{ nullptr };
    ICLTensor                           *dst{ nullptr };
    std::unique_ptr<experimental::CLAbs> op{ nullptr };
};

CLAbsLayer::CLAbsLayer()
    : _impl(std::make_unique<Impl>())
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
    _impl->op  = std::make_unique<experimental::CLAbs>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLAbsLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLAbs::validate(input, output);
}

void CLAbsLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct CLLogLayer::Impl
{
    const ICLTensor                     *src{ nullptr };
    ICLTensor                           *dst{ nullptr };
    std::unique_ptr<experimental::CLLog> op{ nullptr };
};

CLLogLayer::CLLogLayer()
    : _impl(std::make_unique<Impl>())
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
    _impl->op  = std::make_unique<experimental::CLLog>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLLogLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLLog::validate(input, output);
}

void CLLogLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct CLRoundLayer::Impl
{
    const ICLTensor                       *src{ nullptr };
    ICLTensor                             *dst{ nullptr };
    std::unique_ptr<experimental::CLRound> op{ nullptr };
};

CLRoundLayer::CLRoundLayer()
    : _impl(std::make_unique<Impl>())
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
    _impl->op  = std::make_unique<experimental::CLRound>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLRoundLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return experimental::CLRound::validate(input, output);
}

void CLRoundLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
