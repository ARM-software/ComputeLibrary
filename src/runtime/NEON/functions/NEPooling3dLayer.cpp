/*
 * Copyright (c) 2022 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEPooling3dLayer.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuPool3d.h"

namespace arm_compute
{
struct NEPooling3dLayer::Impl
{
    const ITensor                  *src{nullptr};
    ITensor                        *dst{nullptr};
    std::unique_ptr<cpu::CpuPool3d> op{nullptr};
    MemoryGroup                     memory_group{};
    ITensorPack                     run_pack{};
    WorkspaceData<Tensor>           workspace_tensors{};
};

NEPooling3dLayer::~NEPooling3dLayer() = default;

NEPooling3dLayer::NEPooling3dLayer(std::shared_ptr<IMemoryManager> memory_manager) : _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(std::move(memory_manager));
}

void NEPooling3dLayer::configure(const ITensor *input, ITensor *output, const Pooling3dLayerInfo &pool_info)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = std::make_unique<cpu::CpuPool3d>();
    _impl->op->configure(input->info(), output->info(), pool_info);

    _impl->run_pack          = {{TensorType::ACL_SRC, _impl->src}, {TensorType::ACL_DST_0, _impl->dst}};
    _impl->workspace_tensors = manage_workspace<Tensor>(_impl->op->workspace(), _impl->memory_group, _impl->run_pack);
}

Status
NEPooling3dLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const Pooling3dLayerInfo &pool_info)
{
    return cpu::CpuPool3d::validate(input, output, pool_info);
}

void NEPooling3dLayer::run()
{
    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_impl->src, _impl->dst);
    _impl->op->run(_impl->run_pack);
}

} // namespace arm_compute
