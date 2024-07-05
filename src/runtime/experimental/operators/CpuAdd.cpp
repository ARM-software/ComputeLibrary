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

#include "arm_compute/runtime/experimental/operators/CpuAdd.h"

#include "src/cpu/operators/CpuAdd.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{

struct CpuAdd::Impl
{
    std::unique_ptr<cpu::CpuAdd> op{nullptr};
};

CpuAdd::CpuAdd() : _impl(std::make_unique<Impl>())
{
    _impl->op = std::make_unique<cpu::CpuAdd>();
}

CpuAdd::~CpuAdd() = default;

void CpuAdd::configure(const ITensorInfo         *src0,
                       const ITensorInfo         *src1,
                       ITensorInfo               *dst,
                       ConvertPolicy              policy,
                       const ActivationLayerInfo &act_info)
{
    _impl->op->configure(src0, src1, dst, policy, act_info);
}

Status CpuAdd::validate(const ITensorInfo         *src0,
                        const ITensorInfo         *src1,
                        const ITensorInfo         *dst,
                        ConvertPolicy              policy,
                        const ActivationLayerInfo &act_info)
{
    return cpu::CpuAdd::validate(src0, src1, dst, policy, act_info);
}

void CpuAdd::run(ITensorPack &tensors)
{
    _impl->op->run(tensors);
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
