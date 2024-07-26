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

#include "arm_compute/runtime/experimental/operators/CpuGemmDirectConv2d.h"

#include "src/cpu/operators/CpuGemmDirectConv2d.h"

namespace arm_compute
{

namespace experimental
{
namespace op
{

struct CpuGemmDirectConv2d::Impl
{
    std::unique_ptr<arm_compute::cpu::CpuGemmDirectConv2d> cpu_gemm{nullptr};
};

CpuGemmDirectConv2d::CpuGemmDirectConv2d() : _impl(std::make_unique<Impl>())
{
    _impl->cpu_gemm = std::make_unique<cpu::CpuGemmDirectConv2d>();
}

CpuGemmDirectConv2d::~CpuGemmDirectConv2d() = default;

void CpuGemmDirectConv2d::configure(const ITensorInfo *src,
                                    const ITensorInfo *weights,
                                    const ITensorInfo *biases,
                                    ITensorInfo       *dst,
                                    const Conv2dInfo  &info)
{
    _impl->cpu_gemm->configure(src, weights, biases, dst, info);
}

Status CpuGemmDirectConv2d::validate(const ITensorInfo *src,
                                     const ITensorInfo *weights,
                                     const ITensorInfo *biases,
                                     const ITensorInfo *dst,
                                     const Conv2dInfo  &info)
{
    return cpu::CpuGemmDirectConv2d::validate(src, weights, biases, dst, info);
}

void CpuGemmDirectConv2d::run(ITensorPack &tensors)
{
    _impl->cpu_gemm->run(tensors);
}

void CpuGemmDirectConv2d::prepare(ITensorPack &constants)
{
    _impl->cpu_gemm->prepare(constants);
}

experimental::MemoryRequirements CpuGemmDirectConv2d::workspace() const
{
    return _impl->cpu_gemm->workspace();
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
