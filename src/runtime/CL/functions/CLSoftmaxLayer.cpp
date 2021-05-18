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
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "src/core/gpu/cl/kernels/ClSoftmaxKernel.h"
#include "src/runtime/gpu/cl/operators/ClPermute.h"
#include "src/runtime/gpu/cl/operators/ClSoftmax.h"

namespace arm_compute
{
using OperatorType = opencl::ClSoftmax;

template <bool IS_LOG>
struct CLSoftmaxLayerGeneric<IS_LOG>::Impl
{
    const ICLTensor              *src{ nullptr };
    ICLTensor                    *dst{ nullptr };
    std::unique_ptr<OperatorType> op{ nullptr };
    MemoryGroup                   memory_group{};
    std::vector<std::pair<TensorType, std::unique_ptr<CLTensor>>> workspace_tensors{};
};

template <bool IS_LOG>
CLSoftmaxLayerGeneric<IS_LOG>::CLSoftmaxLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(std::move(memory_manager));
}

template <bool IS_LOG>
CLSoftmaxLayerGeneric<IS_LOG>::~CLSoftmaxLayerGeneric() = default;

template <bool IS_LOG>
void CLSoftmaxLayerGeneric<IS_LOG>::configure(const ICLTensor *input, ICLTensor *output, float beta, int32_t axis)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, beta, axis);
}

template <bool IS_LOG>
void CLSoftmaxLayerGeneric<IS_LOG>::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, float beta, int32_t axis)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = std::make_unique<OperatorType>();

    SoftmaxKernelInfo softmax_info{ beta, IS_LOG, input->info()->data_type(), axis };
    _impl->op->configure(compile_context, *input->info(), *output->info(), softmax_info);
    allocate_workspace();
}

template <bool IS_LOG>
Status CLSoftmaxLayerGeneric<IS_LOG>::validate(const ITensorInfo *input, const ITensorInfo *output, float beta, int32_t axis)
{
    SoftmaxKernelInfo softmax_info{ beta, IS_LOG, input->data_type(), axis };
    return OperatorType::validate(*input, *output, softmax_info);
}

template <bool IS_LOG>
void           CLSoftmaxLayerGeneric<IS_LOG>::allocate_workspace()
{
    const auto memory_requirements = _impl->op->workspace();
    std::for_each(memory_requirements.begin(), memory_requirements.end(), [this](const experimental::MemoryInfo & memory_info)
    {
        auto tensor_info = TensorInfo{ TensorShape(memory_info.size), 1, DataType::U8 };
        _impl->workspace_tensors.emplace_back(memory_info.type, std::make_unique<CLTensor>());
        auto tensor = _impl->workspace_tensors.back().second.get();
        ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);
        tensor->allocator()->init(tensor_info);
        _impl->memory_group.manage(tensor);
    });

    std::for_each(_impl->workspace_tensors.begin(), _impl->workspace_tensors.end(), [](std::pair<TensorType, std::unique_ptr<CLTensor>> &wt)
    {
        auto tensor = wt.second.get();
        tensor->allocator()->allocate();
    });
}

template <bool IS_LOG>
void           CLSoftmaxLayerGeneric<IS_LOG>::run()
{
    // Acquire all the temporaries
    MemoryGroupResourceScope scope_mg(_impl->memory_group);

    ARM_COMPUTE_ERROR_ON_NULLPTR(_impl->src, _impl->dst);

    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    std::for_each(_impl->workspace_tensors.begin(), _impl->workspace_tensors.end(), [&pack](std::pair<TensorType, std::unique_ptr<CLTensor>> &wt)
    {
        auto tensor = wt.second.get();
        ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);
        pack.add_tensor(wt.first, tensor);
    });

    _impl->op->run(pack);
}

template class CLSoftmaxLayerGeneric<false>;
template class CLSoftmaxLayerGeneric<true>;

} // namespace arm_compute
