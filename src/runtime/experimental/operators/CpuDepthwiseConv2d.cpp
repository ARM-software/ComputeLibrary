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

#include "arm_compute/runtime/experimental/operators/CpuDepthwiseConv2d.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/misc/InfoHelpers.h"

#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuDepthwiseConv2d.h"

namespace arm_compute
{

namespace experimental
{
namespace op
{

struct CpuDepthwiseConv2d::Impl
{
    std::unique_ptr<cpu::CpuDepthwiseConv2d>                 conv{nullptr};
    std::unique_ptr<cpu::CpuDepthwiseConv2dAssemblyDispatch> conv_asm{nullptr};
};

CpuDepthwiseConv2d::CpuDepthwiseConv2d() : _impl(std::make_unique<Impl>())
{
    _impl->conv     = std::make_unique<cpu::CpuDepthwiseConv2d>();
    _impl->conv_asm = std::make_unique<cpu::CpuDepthwiseConv2dAssemblyDispatch>();
}

CpuDepthwiseConv2d::~CpuDepthwiseConv2d() = default;

void CpuDepthwiseConv2d::configure(ITensorInfo               *src,
                                   const ITensorInfo         *weights,
                                   const ITensorInfo         *biases,
                                   ITensorInfo               *dst,
                                   const PadStrideInfo       &conv_info,
                                   unsigned int               depth_multiplier,
                                   const ActivationLayerInfo &act_info,
                                   const Size2D              &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_LOG_PARAMS(src, weights, dst, conv_info, depth_multiplier, biases, act_info, dilation);
    ARM_COMPUTE_ERROR_THROW_ON(validate(src, weights, biases, dst, conv_info, depth_multiplier, act_info, dilation));

    ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
    _impl->conv->configure(src, weights, biases, dst, info);

    // Configure pipeline
    ActivationLayerInfo act_info_to_use            = ActivationLayerInfo();
    const bool          is_relu                    = arm_compute::utils::info_helpers::is_relu(act_info);
    const bool          is_relu6                   = arm_compute::utils::info_helpers::is_relu6(act_info);
    bool                is_activationlayer_enabled = act_info.enabled() && !(is_relu || is_relu6);

    if (!is_activationlayer_enabled)
    {
        act_info_to_use = act_info;
    }

    info = ConvolutionInfo{conv_info, depth_multiplier, act_info_to_use, dilation};
    _impl->conv_asm->configure(src, weights, biases == nullptr ? nullptr : biases, dst, info);
}

Status CpuDepthwiseConv2d::validate(const ITensorInfo         *input,
                                    const ITensorInfo         *weights,
                                    const ITensorInfo         *biases,
                                    const ITensorInfo         *output,
                                    const PadStrideInfo       &conv_info,
                                    unsigned int               depth_multiplier,
                                    const ActivationLayerInfo &act_info,
                                    const Size2D              &dilation)
{
#if !defined(__aarch64__)
    ARM_COMPUTE_UNUSED(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
    ARM_COMPUTE_RETURN_ERROR_MSG("32-bit is not supported by assembly kernels");
#endif // !defined(__aarch64__)

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_layout() == DataLayout::NCHW,
                                    "NCHW data layout is not valid for CpuDepthwiseConv2d.");

    ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};

    const DepthwiseConvolutionFunction depth_conv_func =
        cpu::CpuDepthwiseConv2d::get_depthwiseconvolution_function(input, weights, biases, output, info);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(depth_conv_func != DepthwiseConvolutionFunction::OPTIMIZED,
                                    "Only a subset of optimized configurations are valid for CpuDepthwiseConv2d.");
    return cpu::CpuDepthwiseConv2d::validate(input, weights, biases, output, info);
}

void CpuDepthwiseConv2d::run(ITensorPack &tensors)
{
    _impl->conv->run(tensors);
}

void CpuDepthwiseConv2d::prepare(ITensorPack &constants)
{
    _impl->conv->prepare(constants);
}

experimental::MemoryRequirements CpuDepthwiseConv2d::workspace() const
{
    auto mem_reqs = _impl->conv_asm->workspace();

    // We do not support permute, so we push all slots directly to what the asm kernel wants.
    for (auto &mem : mem_reqs)
    {
        mem.slot += 3;
    }
    return mem_reqs;
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
