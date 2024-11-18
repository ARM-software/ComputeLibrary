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
#include "arm_compute/runtime/NEON/functions/NEScatter.h"

#include "arm_compute/function_info/ScatterInfo.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuScatter.h"

namespace arm_compute
{
struct NEScatter::Impl
{
    const ITensor                   *src{nullptr};
    const ITensor                   *updates{nullptr};
    const ITensor                   *indices{nullptr};
    ITensor                         *dst{nullptr};
    std::unique_ptr<cpu::CpuScatter> op{nullptr};
    MemoryGroup                      memory_group{};
    ITensorPack                      run_pack{};
    WorkspaceData<Tensor>            workspace_tensors{};
};

NEScatter::NEScatter() : _impl(std::make_unique<Impl>())
{
}
NEScatter::~NEScatter() = default;

void NEScatter::configure(
    const ITensor *src, const ITensor *updates, const ITensor *indices, ITensor *dst, const ScatterInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(updates, indices, dst);
    ARM_COMPUTE_LOG_PARAMS(src, updates, indices, dst, info);

    _impl->op = std::make_unique<cpu::CpuScatter>();
    if (src)
    {
        _impl->op->configure(src->info(), updates->info(), indices->info(), dst->info(), info);
    }
    else
    {
        _impl->op->configure(nullptr, updates->info(), indices->info(), dst->info(), info);
    }

    _impl->run_pack = {
        {TensorType::ACL_SRC_0, src},
        {TensorType::ACL_SRC_1, updates},
        {TensorType::ACL_SRC_2, indices},
        {TensorType::ACL_DST_0, dst},
    };

    _impl->workspace_tensors = manage_workspace<Tensor>(_impl->op->workspace(), _impl->memory_group, _impl->run_pack);
}

Status NEScatter::validate(const ITensorInfo *src,
                           const ITensorInfo *updates,
                           const ITensorInfo *indices,
                           const ITensorInfo *dst,
                           const ScatterInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(src, updates, indices, dst);

    return cpu::CpuScatter::validate(src, updates, indices, dst, info);
}

void NEScatter::run()
{
    _impl->op->run(_impl->run_pack);
}
} // namespace arm_compute
