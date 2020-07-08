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
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLActivationLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLRuntimeContext.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
namespace experimental
{
void CLActivationLayer::configure(const CLCompileContext &compile_context, ITensorInfo *input, ITensorInfo *output, ActivationLayerInfo act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLActivationLayerKernel>();
    k->configure(compile_context, input, output, act_info);
    _kernel = std::move(k);
}

Status CLActivationLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLActivationLayerKernel::validate(input, output, act_info);
}

MemoryRequirements CLActivationLayer::workspace() const
{
    return MemoryRequirements{};
}
} // namespace experimental

struct CLActivationLayer::Impl
{
    const ICLTensor                                 *src{ nullptr };
    ICLTensor                                       *dst{ nullptr };
    CLRuntimeContext                                *ctx{ nullptr };
    std::unique_ptr<experimental::CLActivationLayer> op{ nullptr };
};

CLActivationLayer::CLActivationLayer(CLRuntimeContext *ctx)
    : _impl(support::cpp14::make_unique<Impl>())
{
    _impl->ctx = ctx;
}

CLActivationLayer::CLActivationLayer(CLActivationLayer &&) = default;

CLActivationLayer &CLActivationLayer::operator=(CLActivationLayer &&) = default;

CLActivationLayer::~CLActivationLayer() = default;

void CLActivationLayer::configure(ICLTensor *input, ICLTensor *output, ActivationLayerInfo act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, act_info);
}

void CLActivationLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _impl->src = input;
    _impl->dst = output == nullptr ? input : output;

    _impl->op = arm_compute::support::cpp14::make_unique<experimental::CLActivationLayer>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->dst->info(), act_info);
}

Status CLActivationLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return experimental::CLActivationLayer::validate(input, output, act_info);
}

void CLActivationLayer::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}
} // namespace arm_compute
