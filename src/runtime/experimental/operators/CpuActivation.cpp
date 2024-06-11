/*
 * Copyright (c) 2024 Arm Limited.
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

#include "arm_compute/runtime/experimental/operators/CpuActivation.h"

#include "src/cpu/operators/CpuActivation.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{

struct CpuActivation::Impl
{
    std::unique_ptr<cpu::CpuActivation> op{nullptr};
};

CpuActivation::CpuActivation() : _impl(std::make_unique<Impl>())
{
    _impl->op = std::make_unique<cpu::CpuActivation>();
}

CpuActivation::~CpuActivation() = default;

void CpuActivation::configure(const ITensorInfo *src, ITensorInfo *dst, const ActivationLayerInfo &act_info)
{
    _impl->op->configure(src, dst, act_info);
}

Status CpuActivation::validate(const ITensorInfo *src, const ITensorInfo *dst, const ActivationLayerInfo &act_info)
{
    return cpu::CpuActivation::validate(src, dst, act_info);
}

void CpuActivation::run(ITensorPack &tensors)
{
    _impl->op->run(tensors);
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
