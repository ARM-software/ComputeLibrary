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
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "src/core/helpers/MemoryHelpers.h"

#include "src/runtime/gpu/cl/operators/ClGemmLowpMatrixMultiplyCore.h"

namespace arm_compute
{
using namespace arm_compute::experimental;
using OperatorType = opencl::ClGemmLowpMatrixMultiplyCore;

struct CLGEMMLowpMatrixMultiplyCore::Impl
{
    const ICLTensor              *b{ nullptr };
    std::unique_ptr<OperatorType> op{ nullptr };
    MemoryGroup                   memory_group{};
    ITensorPack                   run_pack{};
    MemoryRequirements            aux_mem_req{};
    WorkspaceData<CLTensor>       workspace_tensors{};
    bool                          is_prepared{ false };
};

CLGEMMLowpMatrixMultiplyCore::CLGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(memory_manager);
}

CLGEMMLowpMatrixMultiplyCore::~CLGEMMLowpMatrixMultiplyCore() = default;

void CLGEMMLowpMatrixMultiplyCore::configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, const GEMMInfo &gemm_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), a, b, c, output, gemm_info);
}

void CLGEMMLowpMatrixMultiplyCore::configure(const CLCompileContext &compile_context, const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

    _impl->b           = b;
    _impl->op          = std::make_unique<OperatorType>();
    _impl->is_prepared = gemm_info.retain_internal_weights();

    _impl->op->configure(compile_context, a->info(), b->info(), c != nullptr ? c->info() : nullptr, output->info(), gemm_info);
    _impl->aux_mem_req = _impl->op->workspace();

    // Manage/allocate auxilairy tensors
    if(_impl->is_prepared)
    {
        _impl->run_pack.add_const_tensor(ACL_SRC_0, a);
        _impl->run_pack.add_tensor(ACL_DST, output);
    }
    else
    {
        _impl->run_pack          = { { ACL_SRC_0, a }, { ACL_SRC_1, _impl->b }, { ACL_SRC_2, c }, { ACL_DST, output } };
        _impl->workspace_tensors = manage_workspace<CLTensor>(_impl->op->workspace(), _impl->memory_group, _impl->run_pack, _impl->run_pack);
    }
}

Status CLGEMMLowpMatrixMultiplyCore::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info)
{
    return OperatorType::validate(a, b, c, output, gemm_info);
}

void CLGEMMLowpMatrixMultiplyCore::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);

    _impl->op->run(_impl->run_pack);
}

void CLGEMMLowpMatrixMultiplyCore::prepare()
{
    if(!_impl->is_prepared)
    {
        _impl->op->prepare(_impl->run_pack);

        // Release temporary tensors that are only used in prepare stage
        release_temporaries(_impl->aux_mem_req, _impl->workspace_tensors);

        _impl->is_prepared = true;
    }
}
} // namespace arm_compute
