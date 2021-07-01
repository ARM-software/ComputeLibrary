/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEWinogradConvolutionLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CPP/Validate.h"
#include "src/core/cpu/kernels/CpuWinogradConv2dKernel.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/cpu/operators/CpuWinogradConv2d.h"

#include "src/core/NEON/kernels/convolution/common/utils.hpp"
#include "src/core/NEON/kernels/convolution/winograd/winograd.hpp"

namespace arm_compute
{
using namespace arm_compute::experimental;

struct NEWinogradConvolutionLayer::Impl
{
    MemoryGroup                             memory_group{};
    std::unique_ptr<cpu::CpuWinogradConv2d> op{ nullptr };
    ITensorPack                             run_pack{};
    ITensorPack                             prep_pack{};
    WorkspaceData<Tensor>                   workspace{};
    experimental::MemoryRequirements        aux_mem_req{};
    const ITensor                          *original_weights{ nullptr };
    bool                                    is_prepared{ false };
    bool                                    is_activationlayer_enabled{ false };
    DataLayout                              data_layout{};
};

NEWinogradConvolutionLayer::NEWinogradConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(std::move(memory_manager));
}

NEWinogradConvolutionLayer::~NEWinogradConvolutionLayer() = default;

void NEWinogradConvolutionLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info,
                                           bool enable_fast_math)
{
    _impl->original_weights = weights;
    _impl->op               = std::make_unique<cpu::CpuWinogradConv2d>();
    _impl->op->configure(input->info(), weights->info(), biases != nullptr ? biases->info() : nullptr, output->info(), conv_info, act_info, enable_fast_math);

    _impl->aux_mem_req = _impl->op->workspace();
    _impl->run_pack    = { { ACL_SRC_0, input }, { ACL_SRC_1, weights }, { ACL_SRC_2, biases }, { ACL_DST, output } };
    _impl->prep_pack   = { { ACL_SRC_1, weights }, { ACL_SRC_2, biases } };
    _impl->workspace   = manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->prep_pack);
}

void NEWinogradConvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    _impl->op->run(_impl->run_pack);
}

Status NEWinogradConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                            const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    return cpu::CpuWinogradConv2d::validate(input, weights, biases, output, conv_info, act_info, enable_fast_math);
}

void NEWinogradConvolutionLayer::prepare()
{
    if(!_impl->is_prepared)
    {
        _impl->op->prepare(_impl->prep_pack);
        _impl->original_weights->mark_as_unused();

        // Release temporary tensors that are only used in prepare stage
        for(auto &ws : _impl->workspace)
        {
            const int slot = ws.first;
            for(auto &m : _impl->aux_mem_req)
            {
                if(m.slot == slot && m.lifetime == MemoryLifetime::Prepare)
                {
                    auto tensor = ws.second.get();
                    tensor->allocator()->free();
                    break;
                }
            }
        }

        _impl->is_prepared = true;
    }
}
} // namespace arm_compute
