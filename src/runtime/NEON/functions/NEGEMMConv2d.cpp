/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGEMMConv2d.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/cpu/operators/CpuGemmDirectConv2d.h"

namespace arm_compute
{
using OperatorType = cpu::CpuGemmDirectConv2d;
using namespace arm_compute::experimental;

struct NEGEMMConv2d::Impl
{
    const ITensor                   *weights{ nullptr };
    std::unique_ptr<OperatorType>    op{ nullptr };
    ITensorPack                      run_pack{};
    ITensorPack                      prep_pack{};
    WorkspaceData<Tensor>            workspace{};
    MemoryGroup                      memory_group{};
    bool                             is_prepared{ false };
    experimental::MemoryRequirements aux_mem_req{};
};

NEGEMMConv2d::NEGEMMConv2d(const std::shared_ptr<IMemoryManager> &memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(memory_manager);
}

NEGEMMConv2d::~NEGEMMConv2d() = default;

void NEGEMMConv2d::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const Conv2dInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    _impl->weights     = weights;
    _impl->is_prepared = false;
    _impl->op          = std::make_unique<OperatorType>();

    _impl->op->configure(input->info(), weights->info(), biases != nullptr ? biases->info() : nullptr, output->info(), info);

    _impl->aux_mem_req = _impl->op->workspace();
    _impl->run_pack    = { { TensorType::ACL_SRC_0, input }, { TensorType::ACL_SRC_2, biases }, { TensorType::ACL_DST, output } };
    _impl->prep_pack   = { { TensorType::ACL_SRC_1, weights }, { TensorType::ACL_SRC_2, biases } };
    _impl->workspace   = manage_workspace<Tensor>(_impl->op->workspace(), _impl->memory_group, _impl->run_pack, _impl->prep_pack);
}

Status NEGEMMConv2d::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const Conv2dInfo &info)
{
    return OperatorType::validate(input, weights, biases, output, info);
}

void NEGEMMConv2d::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    _impl->op->run(_impl->run_pack);
}

void NEGEMMConv2d::prepare()
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
        else
        {
            _impl->run_pack.add_const_tensor(ACL_SRC_1, _impl->weights);
        }

        // Release temporary tensors that are only used in prepare stage
        for(auto &ws : _impl->workspace)
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
