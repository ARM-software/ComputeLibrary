/*
 * Copyright (c) 2023 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEMatMul.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuMatMul.h"

namespace arm_compute
{
struct NEMatMul::Impl
{
    const ITensor                  *lhs{ nullptr };
    const ITensor                  *rhs{ nullptr };
    ITensor                        *output{ nullptr };
    std::unique_ptr<cpu::CpuMatMul> op{ nullptr };
    MemoryGroup                     memory_group{};
    WorkspaceData<Tensor>           workspace_tensors{};
    ITensorPack                     run_pack{};
};

NEMatMul::NEMatMul()
    : _impl(std::make_unique<Impl>())
{
}

NEMatMul::~NEMatMul() = default;

void NEMatMul::configure(ITensor *lhs, ITensor *rhs, ITensor *output, const MatMulInfo &info, const CpuMatMulSettings &settings, const ActivationLayerInfo &act_info)
{
    _impl->lhs    = lhs;
    _impl->rhs    = rhs;
    _impl->output = output;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_impl->lhs, _impl->rhs, _impl->output);
    _impl->op = std::make_unique<cpu::CpuMatMul>();
    _impl->op->configure(lhs->info(), rhs->info(), output->info(), info, settings, act_info);
    _impl->run_pack          = { { ACL_SRC_0, lhs }, { ACL_SRC_1, rhs }, { ACL_DST, output } };
    _impl->workspace_tensors = manage_workspace<Tensor>(_impl->op->workspace(), _impl->memory_group, _impl->run_pack);
}

Status NEMatMul::validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *output, const MatMulInfo &info, const CpuMatMulSettings &settings, const ActivationLayerInfo &act_info)
{
    return cpu::CpuMatMul::validate(lhs, rhs, output, info, settings, act_info);
}

void NEMatMul::run()
{
    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    _impl->op->run(_impl->run_pack);
}
} // namespace arm_compute
