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
#include "src/cpu/operators/CpuWinogradConv2d.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/experimental/operators/CpuWinogradConv2d.h"

#include "support/Cast.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{
using namespace arm_compute::experimental;
using namespace arm_compute::utils::cast;

struct CpuWinogradConv2d::Impl
{
    std::unique_ptr<cpu::CpuWinogradConv2d> op{nullptr};
};

CpuWinogradConv2d::CpuWinogradConv2d() : _impl(std::make_unique<Impl>())
{
    _impl->op = std::make_unique<cpu::CpuWinogradConv2d>();
}

CpuWinogradConv2d::~CpuWinogradConv2d() = default;

void CpuWinogradConv2d::configure(const ITensorInfo         *src,
                                  const ITensorInfo         *weights,
                                  const ITensorInfo         *biases,
                                  ITensorInfo               *dst,
                                  const PadStrideInfo       &conv_info,
                                  const ActivationLayerInfo &act_info,
                                  bool                       enable_fast_math)
{
    _impl->op->configure(src, weights, biases, dst, conv_info, act_info, enable_fast_math);
}
Status CpuWinogradConv2d::validate(const ITensorInfo         *src,
                                   const ITensorInfo         *weights,
                                   const ITensorInfo         *biases,
                                   const ITensorInfo         *dst,
                                   const PadStrideInfo       &conv_info,
                                   const ActivationLayerInfo &act_info,
                                   bool                       enable_fast_math)
{
    return cpu::CpuWinogradConv2d::validate(src, weights, biases, dst, conv_info, act_info, enable_fast_math);
}

void CpuWinogradConv2d::run(ITensorPack &tensors)
{
    _impl->op->run(tensors);
}

void CpuWinogradConv2d::prepare(ITensorPack &tensors)
{
    _impl->op->prepare(tensors);
}

experimental::MemoryRequirements CpuWinogradConv2d::workspace() const
{
    return _impl->op->workspace();
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
