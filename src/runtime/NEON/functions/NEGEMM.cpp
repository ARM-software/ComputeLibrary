/*
 * Copyright (c) 2017-2025 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/core/CPP/Validate.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuDynamicGemm.h"
#include "src/cpu/operators/CpuGemm.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace
{
inline bool is_dynamic(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *d)
{
    return a->is_dynamic() || b->is_dynamic() || (c && c->is_dynamic()) || d->is_dynamic();
}

inline bool is_dynamic(const ITensor *a, const ITensor *b, const ITensor *c, const ITensor *d)
{
    return is_dynamic(a->info(), b->info(), (c != nullptr) ? c->info() : nullptr, d->info());
}

std::unique_ptr<cpu::ICpuOperator> make_and_config_op(const ITensorInfo *a,
                                                      const ITensorInfo *b,
                                                      const ITensorInfo *c,
                                                      ITensorInfo       *d,
                                                      float              alpha,
                                                      float              beta,
                                                      const GEMMInfo    &gemm_info)
{
    // Make the B matrix dynamic values.
    auto b_info_to_use = b->clone();
    if (!gemm_info.reshape_b_only_on_first_run())
    {
        b_info_to_use->set_are_values_constant(false);
    }

    std::unique_ptr<cpu::ICpuOperator> op;
    if (is_dynamic(a, b, c, d))
    {
        auto op_typed = std::make_unique<cpu::CpuDynamicGemm>();
        op_typed->configure(a, b_info_to_use.get(), c, d, alpha, beta, gemm_info);
        op = std::move(op_typed);
    }
    else
    {
        auto op_typed = std::make_unique<cpu::CpuGemm>();
        op_typed->configure(a, b_info_to_use.get(), c, d, alpha, beta, gemm_info);
        op = std::move(op_typed);
    }

    return op;
}
} // namespace

struct NEGEMM::Impl
{
    MemoryGroup      memory_group{};
    IWeightsManager *weights_manager{nullptr};

    std::unique_ptr<cpu::ICpuOperator> op{nullptr};

    const ITensor *original_b{nullptr};
    bool           is_prepared{false};
    bool           is_dynamic{false};

    ITensorPack                      run_pack{};
    ITensorPack                      prep_pack{};
    WorkspaceData<Tensor>            workspace{};
    experimental::MemoryRequirements aux_mem_req{};
};

NEGEMM::NEGEMM(std::shared_ptr<IMemoryManager> memory_manager, IWeightsManager *weights_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group    = MemoryGroup(std::move(memory_manager));
    _impl->weights_manager = weights_manager;
}

NEGEMM::~NEGEMM() = default;

void NEGEMM::configure(const ITensor  *a,
                       const ITensor  *b,
                       const ITensor  *c,
                       ITensor        *d,
                       float           alpha,
                       float           beta,
                       const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, d);

    _impl->is_dynamic = is_dynamic(a, b, c, d);

    if (_impl->is_dynamic)
    {
        ARM_COMPUTE_ERROR_THROW_ON(cpu::CpuDynamicGemm::validate(
            a->info(), b->info(), (c != nullptr) ? c->info() : nullptr, d->info(), alpha, beta, gemm_info));
    }
    else
    {
        ARM_COMPUTE_ERROR_THROW_ON(cpu::CpuGemm::validate(a->info(), b->info(), (c != nullptr) ? c->info() : nullptr,
                                                          d->info(), alpha, beta, gemm_info));
    }

    // Check if we need to reshape the matrix B only on the first run
    _impl->is_prepared = false;
    _impl->memory_group.mappings().clear();
    _impl->original_b = b;

    _impl->op = make_and_config_op(a->info(), b->info(), (c != nullptr) ? c->info() : nullptr, d->info(), alpha, beta,
                                   gemm_info);
    _impl->run_pack  = {{ACL_SRC_0, a}, {ACL_SRC_1, b}, {ACL_SRC_2, c}, {ACL_DST, d}};
    _impl->prep_pack = {{ACL_SRC_1, b}, {ACL_SRC_2, c}};
    if (_impl->is_dynamic)
    {
        // This first acquisition is not so much about the tensor sizes because
        // they are reallocated in run(), but rather about which tensors will
        // be managed in the workspace.
        _impl->aux_mem_req = _impl->op->workspace_dynamic(_impl->run_pack);
    }
    else
    {
        _impl->aux_mem_req = _impl->op->workspace();
    }
    _impl->workspace = manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack,
                                                _impl->prep_pack, /* allocate_now */ false);
}

Status NEGEMM::validate(const ITensorInfo *a,
                        const ITensorInfo *b,
                        const ITensorInfo *c,
                        const ITensorInfo *output,
                        float              alpha,
                        float              beta,
                        const GEMMInfo    &gemm_info)
{
    // Make the B matrix dynamic values.
    auto b_to_use = b->clone();
    if (!gemm_info.reshape_b_only_on_first_run())
    {
        b_to_use->set_are_values_constant(false);
    }

    if (is_dynamic(a, b, c, output))
    {
        return cpu::CpuDynamicGemm::validate(a, b_to_use.get(), c, output, alpha, beta, gemm_info);
    }
    else
    {
        return cpu::CpuGemm::validate(a, b_to_use.get(), c, output, alpha, beta, gemm_info);
    }
}

Status NEGEMM::has_opt_impl(arm_compute::WeightFormat &expected_weight_format,
                            const ITensorInfo         *a,
                            const ITensorInfo         *b,
                            const ITensorInfo         *c,
                            const ITensorInfo         *output,
                            float                      alpha,
                            float                      beta,
                            const GEMMInfo            &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha, beta);
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(a, b, c, output);

    return cpu::CpuGemm::has_opt_impl(expected_weight_format, a, b, c, output, gemm_info);
}

void NEGEMM::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    _impl->op->run(_impl->run_pack);
}

void NEGEMM::prepare()
{
    if (_impl->is_dynamic)
    {
        _impl->aux_mem_req = _impl->op->workspace_dynamic(_impl->run_pack);
        reallocate_tensors(_impl->aux_mem_req, _impl->workspace);
    }
    else if (!_impl->is_prepared)
    {
        allocate_tensors(_impl->aux_mem_req, _impl->workspace);
        _impl->op->prepare(_impl->prep_pack);

        auto has_reshape =
            std::find_if(_impl->aux_mem_req.begin(), _impl->aux_mem_req.end(),
                         [](const MemoryInfo &m) -> bool { return m.lifetime == MemoryLifetime::Persistent; });

        if (has_reshape != std::end(_impl->aux_mem_req))
        {
            _impl->original_b->mark_as_unused();
        }
        else
        {
            _impl->run_pack.add_const_tensor(ACL_SRC_1, _impl->original_b);
        }

        // Release temporary tensors that are only used in prepare stage
        release_temporaries<Tensor>(_impl->aux_mem_req, _impl->workspace);
        _impl->is_prepared = true;
    }
}
} // namespace arm_compute
