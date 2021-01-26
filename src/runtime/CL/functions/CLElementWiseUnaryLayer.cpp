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
#include "arm_compute/runtime/CL/functions/CLElementWiseUnaryLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "src/core/CL/ICLKernel.h"
#include "src/runtime/gpu/cl/operators/ClElementwiseUnary.h"

namespace arm_compute
{
struct CLRsqrtLayer::Impl
{
    const ICLTensor                 *src{ nullptr };
    ICLTensor                       *dst{ nullptr };
    std::unique_ptr<opencl::ClRsqrt> op{ nullptr };
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
    _impl->op  = std::make_unique<opencl::ClRsqrt>();
    _impl->op->configure(compile_context, input->info(), output->info());
}

Status CLRsqrtLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return opencl::ClRsqrt::validate(input, output);
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
    const ICLTensor               *src{ nullptr };
    ICLTensor                     *dst{ nullptr };
    std::unique_ptr<opencl::ClExp> op{ nullptr };
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
    _impl->op  = std::make_unique<opencl::ClExp>();
    _impl->op->configure(compile_context, input->info(), output->info());
}

Status CLExpLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return opencl::ClExp::validate(input, output);
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
    const ICLTensor               *src{ nullptr };
    ICLTensor                     *dst{ nullptr };
    std::unique_ptr<opencl::ClNeg> op{ nullptr };
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
    _impl->op  = std::make_unique<opencl::ClNeg>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLNegLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return opencl::ClNeg::validate(input, output);
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
    const ICLTensor               *src{ nullptr };
    ICLTensor                     *dst{ nullptr };
    std::unique_ptr<opencl::ClSin> op{ nullptr };
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
    _impl->op  = std::make_unique<opencl::ClSin>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLSinLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return opencl::ClSin::validate(input, output);
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
    const ICLTensor               *src{ nullptr };
    ICLTensor                     *dst{ nullptr };
    std::unique_ptr<opencl::ClAbs> op{ nullptr };
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
    _impl->op  = std::make_unique<opencl::ClAbs>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLAbsLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return opencl::ClAbs::validate(input, output);
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
    const ICLTensor               *src{ nullptr };
    ICLTensor                     *dst{ nullptr };
    std::unique_ptr<opencl::ClLog> op{ nullptr };
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
    _impl->op  = std::make_unique<opencl::ClLog>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLLogLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return opencl::ClLog::validate(input, output);
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
    const ICLTensor                 *src{ nullptr };
    ICLTensor                       *dst{ nullptr };
    std::unique_ptr<opencl::ClRound> op{ nullptr };
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
    _impl->op  = std::make_unique<opencl::ClRound>();
    _impl->op->configure(compile_context, input->info(), output->info());
}
Status CLRoundLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return opencl::ClRound::validate(input, output);
}

void CLRoundLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
