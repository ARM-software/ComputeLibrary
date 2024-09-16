/*
 * Copyright (c) 2017-2021, 2023-2024 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuGEMMLowp.h"

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#include "src/core/utils/quantization/AsymmHelpers.h"
#include "src/cpu/operators/CpuGemmLowpMatrixMultiplyCore.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{
struct CpuGEMMLowp::Impl
{
    std::unique_ptr<arm_compute::cpu::CpuGemmLowpMatrixMultiplyCore> op{nullptr};
    bool                                                             is_prepared{false};
};

CpuGEMMLowp::CpuGEMMLowp() : _impl(std::make_unique<Impl>())
{
    _impl->op = std::make_unique<cpu::CpuGemmLowpMatrixMultiplyCore>();
}
CpuGEMMLowp::~CpuGEMMLowp() = default;

experimental::MemoryRequirements CpuGEMMLowp::workspace() const
{
    return _impl->op->workspace();
}

void CpuGEMMLowp::configure(
    const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, ITensorInfo *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

    // Make the B matrix dynamic values.
    auto b_info_to_use = b->clone();
    if (!gemm_info.reshape_b_only_on_first_run())
    {
        b_info_to_use->set_are_values_constant(false);
    }

    _impl->is_prepared = false;
    _impl->op->configure(a, b_info_to_use.get(), (c != nullptr ? c : nullptr), output, gemm_info);
}

Status CpuGEMMLowp::validate(const ITensorInfo *a,
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

void CpuGEMMLowp::run(ITensorPack &tensors)
{
    prepare(tensors);
    _impl->op->run(tensors);
}

void CpuGEMMLowp::prepare(ITensorPack &tensors)
{
    if (!_impl->is_prepared)
    {
        _impl->op->prepare(tensors);

        auto aux_mem_req = _impl->op->workspace();

        auto has_reshape =
            std::find_if(aux_mem_req.begin(), aux_mem_req.end(),
                         [](const MemoryInfo &m) -> bool { return m.lifetime == MemoryLifetime::Persistent; });

        if (has_reshape != std::end(aux_mem_req))
        {
            auto b = tensors.get_tensor(TensorType::ACL_SRC_1);
            b->mark_as_unused();
        }

        _impl->is_prepared = true;
    }
}
} // namespace op
} // namespace experimental
} // namespace arm_compute
