/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"

#include "arm_compute/core/NEON/kernels/NEReshapeLayerKernel.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Types.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
namespace experimental
{
void NEReshapeLayer::configure(const ITensorInfo *input, ITensorInfo *output)
{
    auto k = arm_compute::support::cpp14::make_unique<NEReshapeLayerKernel>();
    k->configure(input, output);
    _kernel = std::move(k);
}

Status NEReshapeLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return arm_compute::NEReshapeLayerKernel::validate(input, output);
}

MemoryRequirements NEReshapeLayer::workspace() const
{
    return MemoryRequirements{};
}
} // namespace experimental

struct NEReshapeLayer::Impl
{
    const ITensor                                *src{ nullptr };
    ITensor                                      *dst{ nullptr };
    std::unique_ptr<experimental::NEReshapeLayer> op{ nullptr };
};

NEReshapeLayer::NEReshapeLayer()
    : _impl(support::cpp14::make_unique<Impl>())
{
}

NEReshapeLayer::NEReshapeLayer(NEReshapeLayer &&) = default;

NEReshapeLayer &NEReshapeLayer::operator=(NEReshapeLayer &&) = default;

NEReshapeLayer::~NEReshapeLayer() = default;

void NEReshapeLayer::configure(const ITensor *input, ITensor *output)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = arm_compute::support::cpp14::make_unique<experimental::NEReshapeLayer>();
    _impl->op->configure(input->info(), output->info());
}

Status NEReshapeLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(experimental::NEReshapeLayer::validate(input, output));

    return Status{};
}

void NEReshapeLayer::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}
} // namespace arm_compute
