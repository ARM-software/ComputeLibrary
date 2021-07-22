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
#include "arm_compute/runtime/NEON/functions/NEGEMMConvolutionLayer.h"

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/cpu/operators/CpuGemmConvolution.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
struct NEGEMMConvolutionLayer::Impl
{
    const ITensor                           *weights{ nullptr };
    std::unique_ptr<cpu::CpuGemmConvolution> op{ nullptr };
    ITensorPack                              run_pack{};
    ITensorPack                              prep_pack{};
    MemoryGroup                              memory_group{};
    IWeightsManager                         *weights_manager{ nullptr };
    MemoryRequirements                       aux_mem_req{};
    WorkspaceData<Tensor>                    workspace_tensors{};
    bool                                     is_prepared{ false };
};

NEGEMMConvolutionLayer::NEGEMMConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager, IWeightsManager *weights_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->weights_manager = weights_manager;
    _impl->memory_group    = MemoryGroup(memory_manager);
}
NEGEMMConvolutionLayer::~NEGEMMConvolutionLayer() = default;

void NEGEMMConvolutionLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                       const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    _impl->weights = weights;
    _impl->op      = std::make_unique<cpu::CpuGemmConvolution>();
    _impl->op->configure(input->info(), weights->info(), (biases != nullptr ? biases->info() : nullptr), output->info(), conv_info, weights_info, dilation, act_info, enable_fast_math, num_groups);

    _impl->run_pack =
    {
        { TensorType::ACL_SRC_0, input },
        { TensorType::ACL_SRC_1, weights },
        { TensorType::ACL_SRC_2, biases },
        { TensorType::ACL_DST, output }
    };
    _impl->prep_pack =
    {
        { TensorType::ACL_SRC_1, weights },
        { TensorType::ACL_SRC_2, biases },
    };
    _impl->aux_mem_req       = _impl->op->workspace();
    _impl->workspace_tensors = manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->prep_pack);
}

Status NEGEMMConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                        const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math, unsigned int num_groups)
{
    return cpu::CpuGemmConvolution::validate(input, weights, biases, output, conv_info, weights_info, dilation, act_info, enable_fast_math, num_groups);
}

void NEGEMMConvolutionLayer::run()
{
    prepare();
    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    _impl->op->run(_impl->run_pack);
}

void NEGEMMConvolutionLayer::prepare()
{
    if(!_impl->is_prepared)
    {
        _impl->op->prepare(_impl->prep_pack);
        auto has_reshape = std::find_if(_impl->aux_mem_req.begin(),
                                        _impl->aux_mem_req.end(),
                                        [](const MemoryInfo & m) -> bool { return m.lifetime == MemoryLifetime::Persistent; });

        if(has_reshape != std::end(_impl->aux_mem_req))
        {
            _impl->weights->mark_as_unused();
        }
        for(auto &ws : _impl->workspace_tensors)
        {
            const int slot = ws.slot;
            for(auto &m : _impl->aux_mem_req)
            {
                if(m.slot == slot && m.lifetime == MemoryLifetime::Prepare)
                {
                    auto tensor = ws.tensor.get();
                    tensor->allocator()->free();
                    break;
                }
            }
        }
        _impl->is_prepared = true;
    }
}
} // namespace arm_compute
