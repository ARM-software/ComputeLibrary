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
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/cpu/kernels/CpuSoftmaxKernel.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/helpers/SoftmaxHelpers.h"
#include "src/runtime/cpu/operators/CpuSoftmax.h"

namespace arm_compute
{
template <bool IS_LOG>
struct NESoftmaxLayerGeneric<IS_LOG>::Impl
{
    const ITensor                                  *src{ nullptr };
    ITensor                                        *dst{ nullptr };
    Tensor                                          max{ nullptr };
    std::unique_ptr<cpu::CpuSoftmaxGeneric<IS_LOG>> op{ nullptr };
    MemoryGroup                                     memory_group{};
    ITensorPack                                     run_pack{};
    WorkspaceData<Tensor>                           workspace_tensors{};
};

template <bool IS_LOG>
NESoftmaxLayerGeneric<IS_LOG>::NESoftmaxLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(std::move(memory_manager));
}

template <bool IS_LOG>
NESoftmaxLayerGeneric<IS_LOG>::NESoftmaxLayerGeneric(NESoftmaxLayerGeneric &&) = default;
template <bool                 IS_LOG>
NESoftmaxLayerGeneric<IS_LOG> &NESoftmaxLayerGeneric<IS_LOG>::operator=(NESoftmaxLayerGeneric &&) = default;
template <bool                 IS_LOG>
NESoftmaxLayerGeneric<IS_LOG>::~NESoftmaxLayerGeneric() = default;

template <bool IS_LOG>
void NESoftmaxLayerGeneric<IS_LOG>::configure(ITensor *input, ITensor *output, float beta, int32_t axis)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _impl->src = input;
    _impl->dst = output;
    _impl->op  = std::make_unique<cpu::CpuSoftmaxGeneric<IS_LOG>>();
    _impl->op->configure(input->info(), output->info(), beta, axis);

    _impl->run_pack          = { { TensorType::ACL_SRC, _impl->src }, { TensorType::ACL_DST, _impl->dst } };
    _impl->workspace_tensors = manage_workspace<Tensor>(_impl->op->workspace(), _impl->memory_group, _impl->run_pack);
}

template <bool IS_LOG>
Status NESoftmaxLayerGeneric<IS_LOG>::validate(const ITensorInfo *input, const ITensorInfo *output, float beta, int32_t axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuSoftmaxGeneric<IS_LOG>::validate(input, output, beta, axis));
    return Status{};
}

template <bool IS_LOG>
void           NESoftmaxLayerGeneric<IS_LOG>::run()
{
    // Acquire all the temporaries
    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_impl->src, _impl->dst);
    _impl->op->run(_impl->run_pack);
}

template class NESoftmaxLayerGeneric<false>;
template class NESoftmaxLayerGeneric<true>;

} // namespace arm_compute
