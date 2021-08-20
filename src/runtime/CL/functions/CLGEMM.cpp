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
#include "arm_compute/runtime/CL/functions/CLGEMM.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/gpu/cl/operators/ClGemm.h"

namespace arm_compute
{
using namespace arm_compute::experimental;
using OperatorType = opencl::ClGemm;

struct CLGEMM::Impl
{
    const ICLTensor              *b{ nullptr };
    std::unique_ptr<OperatorType> op{ nullptr };
    MemoryGroup                   memory_group{};
    IWeightsManager              *weights_manager{ nullptr };
    ITensorPack                   run_pack{};
    ITensorPack                   prep_pack{};
    MemoryRequirements            aux_mem_req{};
    WorkspaceData<CLTensor>       workspace_tensors{};
    bool                          is_prepared{ false };
};

CLGEMM::CLGEMM(std::shared_ptr<IMemoryManager> memory_manager, IWeightsManager *weights_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group    = MemoryGroup(memory_manager);
    _impl->weights_manager = weights_manager;
}

CLGEMM::~CLGEMM() = default;

void CLGEMM::configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), a, b, c, output, alpha, beta, gemm_info);
}

void CLGEMM::configure(const CLCompileContext &compile_context, const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

    _impl->b           = b;
    _impl->op          = std::make_unique<OperatorType>();
    _impl->is_prepared = gemm_info.retain_internal_weights();

    _impl->op->configure(compile_context, a->info(), b->info(), c != nullptr ? c->info() : nullptr, output->info(), alpha, beta, gemm_info);
    _impl->aux_mem_req = _impl->op->workspace();

    // Manage/allocate auxilairy tensors
    if(_impl->is_prepared)
    {
        _impl->run_pack.add_const_tensor(ACL_SRC_0, a);
        _impl->run_pack.add_tensor(ACL_DST, output);
    }
    else
    {
        _impl->run_pack  = { { ACL_SRC_0, a }, { ACL_SRC_2, c }, { ACL_DST, output } };
        _impl->prep_pack = { { ACL_SRC_1, _impl->b } };

        _impl->workspace_tensors = manage_workspace<CLTensor>(_impl->op->workspace(), _impl->memory_group, _impl->run_pack, _impl->prep_pack);
    }
}

Status CLGEMM::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    return OperatorType::validate(a, b, c, output, alpha, beta, gemm_info);
}

void CLGEMM::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);

    _impl->op->run(_impl->run_pack);
}

void CLGEMM::prepare()
{
    if(!_impl->is_prepared)
    {
        _impl->op->prepare(_impl->prep_pack);

        auto has_reshape = std::find_if(_impl->aux_mem_req.begin(),
                                        _impl->aux_mem_req.end(),
                                        [](const MemoryInfo & m) -> bool { return m.lifetime == MemoryLifetime::Persistent; });

        if(has_reshape != std::end(_impl->aux_mem_req))
        {
            _impl->b->mark_as_unused();
        }
        else
        {
            // Pack the B matrix to be used as the underlying GEMM performs no reshapes
            _impl->run_pack.add_const_tensor(ACL_SRC_1, _impl->b);
        }
        _impl->is_prepared = true;
    }
}
} // namespace arm_compute
