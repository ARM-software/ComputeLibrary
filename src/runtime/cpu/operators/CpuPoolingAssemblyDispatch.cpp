/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/runtime/cpu/operators/CpuPoolingAssemblyDispatch.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/CPP/Validate.h"
#include "src/core/cpu/kernels/CpuPoolingAssemblyWrapperKernel.h"

namespace arm_compute
{
namespace cpu
{
CpuPoolingAssemblyDispatch::CpuPoolingAssemblyDispatch(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _workspace(),
      _is_global_pooling_layer(false)
{
}

CpuPoolingAssemblyDispatch::~CpuPoolingAssemblyDispatch() = default;

void CpuPoolingAssemblyDispatch::configure(const ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &info)
{
    const CPUInfo     &ci          = NEScheduler::get().cpu_info();
    const unsigned int num_threads = NEScheduler::get().num_threads();

    // If we don't support a combination of data types, silently return: it is the caller's responsibility to check if configure() was successful via is_configured()
    if(!CpuPoolingAssemblyDispatch::validate(src, dst, info))
    {
        return;
    }

    auto pooling_wrapper = std::make_unique<kernels::CpuPoolingAssemblyWrapperKernel>();
    ARM_COMPUTE_ERROR_ON(pooling_wrapper == nullptr);
    pooling_wrapper->configure(src, dst, info, ci);

    // Check if we have Global Pooling Layer
    _is_global_pooling_layer = (src->dimension(2) == info.pool_size.width) && (src->dimension(1) == info.pool_size.height);

    // Allocate workspace based on kernel's memory requirements
    constexpr size_t alignment      = 4096;
    const size_t     workspace_size = pooling_wrapper->get_working_size(num_threads);
    _workspace.allocator()->init(TensorInfo(TensorShape{ (workspace_size + alignment /* FIXME: remove alignment after COMPMID-1088 */) }, 1, DataType::S8), alignment);
    _memory_group.manage(&_workspace);
    _workspace.allocator()->allocate();

    _kernel = std::move(pooling_wrapper);
}

Status CpuPoolingAssemblyDispatch::validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &info)
{
    return kernels::CpuPoolingAssemblyWrapperKernel::validate(src, dst, info);
}

bool CpuPoolingAssemblyDispatch::is_configured() const
{
    return _kernel != nullptr;
}

void CpuPoolingAssemblyDispatch::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No srcs provided");

    tensors.add_tensor(TensorType::ACL_DST_1, &_workspace);

    if(_is_global_pooling_layer)
    {
        NEScheduler::get().schedule_op(_kernel.get(), Window::DimX, _kernel->window(), tensors);
    }
    else
    {
        NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
    }
}
} // namespace cpu
} // namespace arm_compute
