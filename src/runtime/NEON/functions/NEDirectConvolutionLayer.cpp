/*
 * Copyright (c) 2017-2021, 2024-2025 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/profile/acl_profile.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuDirectConv2d.h"

namespace arm_compute
{
struct NEDirectConvolutionLayer::Impl
{
    MemoryGroup                           memory_group{};
    ITensor                              *src{nullptr};
    const ITensor                        *weights{nullptr};
    const ITensor                        *bias{nullptr};
    ITensor                              *dst{nullptr};
    std::unique_ptr<cpu::CpuDirectConv2d> op{nullptr};
    ITensorPack                           run_pack{};
    WorkspaceData<Tensor>                 workspace_tensors{};
};

NEDirectConvolutionLayer::NEDirectConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_manager(std::move(memory_manager)), _impl(std::make_unique<Impl>())
{
}
NEDirectConvolutionLayer::~NEDirectConvolutionLayer() = default;

void NEDirectConvolutionLayer::configure(ITensor                   *input,
                                         const ITensor             *weights,
                                         const ITensor             *bias,
                                         ITensor                   *output,
                                         const PadStrideInfo       &conv_info,
                                         const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NEDirectConvolutionLayer::configure");
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    _impl->memory_group.mappings().clear();
    _impl->src     = input;
    _impl->weights = weights;
    _impl->bias    = bias;
    _impl->dst     = output;
    _impl->op      = std::make_unique<cpu::CpuDirectConv2d>(_memory_manager);
    _impl->op->configure(input->info(), weights->info(), (bias != nullptr ? bias->info() : nullptr), output->info(),
                         conv_info, act_info);

    _impl->run_pack = {{ACL_SRC_0, input}, {ACL_SRC_1, weights}, {ACL_SRC_2, bias}, {ACL_DST, output}};

    _impl->workspace_tensors = manage_workspace<Tensor>(_impl->op->workspace(), _impl->memory_group, _impl->run_pack);
}

Status NEDirectConvolutionLayer::validate(const ITensorInfo         *input,
                                          const ITensorInfo         *weights,
                                          const ITensorInfo         *bias,
                                          const ITensorInfo         *output,
                                          const PadStrideInfo       &conv_info,
                                          const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NEDirectConvolutionLayer::validate");
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(input, weights, bias, output);
    return cpu::CpuDirectConv2d::validate(input, weights, bias, output, conv_info, act_info);
}

void NEDirectConvolutionLayer::run()
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NEDirectConvolutionLayer::run");
    _impl->op->run(_impl->run_pack);
}
} // namespace arm_compute
