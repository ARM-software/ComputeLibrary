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

#include "arm_compute/runtime/NEON/functions/NEAddMulAdd.h"

#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuAddMulAdd.h"

namespace arm_compute
{
struct NEAddMulAdd::Impl
{
    std::unique_ptr<cpu::CpuAddMulAdd> op{nullptr};
    WorkspaceData<Tensor>              workspace_tensors{};
    ITensorPack                        run_pack{};
    MemoryGroup                        memory_group{};
};

NEAddMulAdd::NEAddMulAdd(std::shared_ptr<IMemoryManager> memory_manager) : _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(std::move(memory_manager));
}

NEAddMulAdd::~NEAddMulAdd() = default;

void NEAddMulAdd::configure(ITensor                   *input1,
                            ITensor                   *input2,
                            ITensor                   *bn_mul,
                            ITensor                   *bn_add,
                            ITensor                   *add_output,
                            ITensor                   *final_output,
                            const ConvertPolicy        policy,
                            const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_LOG_PARAMS(input1, input2, bn_mul, bn_add, add_output, final_output, policy, act_info);

    _impl->op = std::make_unique<cpu::CpuAddMulAdd>();
    _impl->op->configure(input1->info(), input2->info(), bn_mul->info(), bn_add->info(),
                         add_output != nullptr ? add_output->info() : nullptr, final_output->info(), policy, act_info);

    _impl->run_pack = {
        {TensorType::ACL_SRC_0, input1}, {TensorType::ACL_SRC_1, input2},     {TensorType::ACL_SRC_2, bn_mul},
        {TensorType::ACL_SRC_3, bn_add}, {TensorType::ACL_DST_0, add_output}, {TensorType::ACL_DST_1, final_output},
    };

    _impl->workspace_tensors = manage_workspace<Tensor>(_impl->op->workspace(), _impl->memory_group, _impl->run_pack);
}

Status NEAddMulAdd::validate(const ITensorInfo         *input1,
                             const ITensorInfo         *input2,
                             const ITensorInfo         *bn_mul,
                             const ITensorInfo         *bn_add,
                             const ITensorInfo         *add_output,
                             const ITensorInfo         *final_output,
                             ConvertPolicy              policy,
                             const ActivationLayerInfo &act_info)
{
    return cpu::CpuAddMulAdd::validate(input1, input2, bn_mul, bn_add, add_output, final_output, policy, act_info);
}

void NEAddMulAdd::run()
{
    _impl->op->run(_impl->run_pack);
}
} // namespace arm_compute
