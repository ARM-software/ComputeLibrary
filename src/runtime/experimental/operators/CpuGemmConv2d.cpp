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

#include "arm_compute/runtime/experimental/operators/CpuGemmConv2d.h"

#include "src/cpu/operators/CpuGemmConv2d.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{

struct CpuGemmConv2d::Impl
{
    std::unique_ptr<cpu::CpuGemmConv2d> op{nullptr};
};

CpuGemmConv2d::CpuGemmConv2d() : _impl(std::make_unique<Impl>())
{
    _impl->op = std::make_unique<cpu::CpuGemmConv2d>();
}

CpuGemmConv2d::~CpuGemmConv2d() = default;

void CpuGemmConv2d::configure(const ITensorInfo         *src,
                              const ITensorInfo         *weights,
                              const ITensorInfo         *biases,
                              ITensorInfo               *dst,
                              const PadStrideInfo       &conv_info,
                              const WeightsInfo         &weights_info,
                              const Size2D              &dilation,
                              const ActivationLayerInfo &act_info,
                              bool                       enable_fast_math,
                              unsigned int               num_groups)
{
    _impl->op->configure(src, weights, biases, dst, conv_info, weights_info, dilation, act_info, enable_fast_math,
                         num_groups);
}

Status CpuGemmConv2d::validate(const ITensorInfo         *src,
                               const ITensorInfo         *weights,
                               const ITensorInfo         *biases,
                               const ITensorInfo         *output,
                               const PadStrideInfo       &conv_info,
                               const WeightsInfo         &weights_info,
                               const Size2D              &dilation,
                               const ActivationLayerInfo &act_info,
                               bool                       enable_fast_math,
                               unsigned int               num_groups)
{
    return cpu::CpuGemmConv2d::validate(src, weights, biases, output, conv_info, weights_info, dilation, act_info,
                                        enable_fast_math, num_groups);
}

Status CpuGemmConv2d::has_opt_impl(arm_compute::WeightFormat &expected_weight_format,
                                   const ITensorInfo         *src,
                                   const ITensorInfo         *weights,
                                   const ITensorInfo         *biases,
                                   const ITensorInfo         *output,
                                   const PadStrideInfo       &conv_info,
                                   const WeightsInfo         &weights_info,
                                   const Size2D              &dilation,
                                   const ActivationLayerInfo &act_info,
                                   const bool                 enable_fast_math)
{
    return cpu::CpuGemmConv2d::has_opt_impl(expected_weight_format, src, weights, biases, output, conv_info,
                                            weights_info, dilation, act_info, enable_fast_math);
}

void CpuGemmConv2d::run(ITensorPack &tensors)
{
    _impl->op->run(tensors);
}

void CpuGemmConv2d::prepare(ITensorPack &tensors)
{
    _impl->op->prepare(tensors);
}

experimental::MemoryRequirements CpuGemmConv2d::workspace() const
{
    return _impl->op->workspace();
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
