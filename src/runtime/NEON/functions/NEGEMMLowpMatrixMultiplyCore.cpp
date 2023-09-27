/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuGemmLowpMatrixMultiplyCore.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
struct NEGEMMLowpMatrixMultiplyCore::Impl
{
    const ITensor                                      *b{nullptr};
    std::unique_ptr<cpu::CpuGemmLowpMatrixMultiplyCore> op{nullptr};
    ITensorPack                                         run_pack{};
    ITensorPack                                         prep_pack{};
    MemoryGroup                                         memory_group{};
    IWeightsManager                                    *weights_manager{nullptr};
    MemoryRequirements                                  aux_mem_req{};
    WorkspaceData<Tensor>                               workspace_tensors{};
    bool                                                is_prepared{false};
};

NEGEMMLowpMatrixMultiplyCore::NEGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager,
                                                           IWeightsManager                *weights_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->weights_manager = weights_manager;
    _impl->memory_group    = MemoryGroup(memory_manager);
}
NEGEMMLowpMatrixMultiplyCore::~NEGEMMLowpMatrixMultiplyCore() = default;

void NEGEMMLowpMatrixMultiplyCore::configure(
    const ITensor *a, const ITensor *b, const ITensor *c, ITensor *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

    // Make the B matrix dynamic values.
    auto b_info_to_use = b->info()->clone();
    if (!gemm_info.reshape_b_only_on_first_run())
    {
        b_info_to_use->set_are_values_constant(false);
    }

    _impl->b  = b;
    _impl->op = std::make_unique<cpu::CpuGemmLowpMatrixMultiplyCore>();
    _impl->op->configure(a->info(), b_info_to_use.get(), (c != nullptr ? c->info() : nullptr), output->info(),
                         gemm_info);
    _impl->run_pack    = {{TensorType::ACL_SRC_0, a},
                          {TensorType::ACL_SRC_1, b},
                          {TensorType::ACL_SRC_2, c},
                          {TensorType::ACL_DST, output}};
    _impl->prep_pack   = {{TensorType::ACL_SRC_1, b}, {TensorType::ACL_SRC_2, c}};
    _impl->aux_mem_req = _impl->op->workspace();
    _impl->workspace_tensors =
        manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->prep_pack);
}

Status NEGEMMLowpMatrixMultiplyCore::validate(const ITensorInfo *a,
                                              const ITensorInfo *b,
                                              const ITensorInfo *c,
                                              const ITensorInfo *output,
                                              const GEMMInfo    &gemm_info)
{
    // Make the B matrix dynamic values.
    auto b_info_to_use = b->clone();
    if (!gemm_info.reshape_b_only_on_first_run())
    {
        b_info_to_use->set_are_values_constant(false);
    }

    return cpu::CpuGemmLowpMatrixMultiplyCore::validate(a, b_info_to_use.get(), c, output, gemm_info);
}

void NEGEMMLowpMatrixMultiplyCore::run()
{
    prepare();
    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    _impl->op->run(_impl->run_pack);
}

void NEGEMMLowpMatrixMultiplyCore::prepare()
{
    if (!_impl->is_prepared)
    {
        _impl->op->prepare(_impl->prep_pack);

        auto has_reshape =
            std::find_if(_impl->aux_mem_req.begin(), _impl->aux_mem_req.end(),
                         [](const MemoryInfo &m) -> bool { return m.lifetime == MemoryLifetime::Persistent; });

        if (has_reshape != std::end(_impl->aux_mem_req))
        {
            _impl->b->mark_as_unused();
        }

        // Release temporary tensors that are only used in prepare stage
        release_temporaries<Tensor>(_impl->aux_mem_req, _impl->workspace_tensors);
        _impl->is_prepared = true;
    }
}
} // namespace arm_compute
