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
#include "arm_compute/runtime/experimental/operators/CpuMul.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuMul.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{

struct CpuMul::Impl
{
    std::unique_ptr<cpu::CpuMul> op{nullptr};
};

CpuMul::CpuMul() : _impl(std::make_unique<Impl>())
{
    _impl->op = std::make_unique<cpu::CpuMul>();
}

CpuMul::~CpuMul() = default;

Status CpuMul::validate(const ITensorInfo         *src1,
                        const ITensorInfo         *src2,
                        const ITensorInfo         *dst,
                        float                      scale,
                        ConvertPolicy              overflow_policy,
                        RoundingPolicy             rounding_policy,
                        const ActivationLayerInfo &act_info)
{
    return cpu::CpuMul::validate(src1, src2, dst, scale, overflow_policy, rounding_policy, act_info);
}

void CpuMul::configure(ITensorInfo               *src1,
                       ITensorInfo               *src2,
                       ITensorInfo               *dst,
                       float                      scale,
                       ConvertPolicy              overflow_policy,
                       RoundingPolicy             rounding_policy,
                       const ActivationLayerInfo &act_info)
{
    _impl->op->configure(src1, src2, dst, scale, overflow_policy, rounding_policy, act_info);
}

void CpuMul::run(ITensorPack &tensors)
{
    _impl->op->run(tensors);
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
