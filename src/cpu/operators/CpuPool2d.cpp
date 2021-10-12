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
#include "src/cpu/operators/CpuPool2d.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/common/utils/Log.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "src/cpu/kernels/CpuPool2dKernel.h"
#include "src/cpu/kernels/internal/CpuPool2dAssemblyWrapperKernel.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace cpu
{
CpuPool2d::CpuPool2d()
    : _pooling_layer_kernel(),
      _asm_glue(),
      _is_global_pooling_layer(false),
      _data_layout(DataLayout::NCHW),
      _aux_mem(1)
{
}

CpuPool2d::~CpuPool2d() = default;

void CpuPool2d::configure(ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &pool_info, ITensorInfo *indices)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst, pool_info, indices);

    // Check if we can run assembly kernels. Currently, indices are not supported by those kernels
    const bool run_optimised = bool(kernels::CpuPool2dAssemblyWrapperKernel::validate(src, dst, pool_info)) && (indices == nullptr);

    // Get data layout
    _data_layout = pool_info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : pool_info.data_layout;

    // Check if we have Global Pooling Layer
    const unsigned int idx_width  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const unsigned int idx_height = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    _is_global_pooling_layer      = (src->dimension(idx_width) == pool_info.pool_size.width) && (src->dimension(idx_height) == pool_info.pool_size.height);

    if(run_optimised)
    {
        const CPUInfo     &ci          = NEScheduler::get().cpu_info();
        const unsigned int num_threads = NEScheduler::get().num_threads();

        auto pooling_wrapper = std::make_unique<kernels::CpuPool2dAssemblyWrapperKernel>();
        ARM_COMPUTE_ERROR_ON(pooling_wrapper == nullptr);
        pooling_wrapper->configure(src, dst, pool_info, ci);

        // Get kernel's memory requirements
        constexpr size_t alignment      = 4096;
        const size_t     workspace_size = pooling_wrapper->get_working_size(num_threads);
        _aux_mem[0]                     = MemoryInfo(TensorType::ACL_INT_0, MemoryLifetime::Temporary, workspace_size, alignment);

        _asm_glue = std::move(pooling_wrapper);
    }
    else
    {
        // Configure pooling kernel
        auto k = std::make_unique<kernels::CpuPool2dKernel>();
        k->configure(src, dst, pool_info, indices);
        _pooling_layer_kernel = std::move(k);
    }
}

Status CpuPool2d::validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    const bool run_optimised = bool(kernels::CpuPool2dAssemblyWrapperKernel::validate(src, dst, pool_info)) && (indices == nullptr);

    if(run_optimised)
    {
        return Status{};
    }

    return kernels::CpuPool2dKernel::validate(src, dst, pool_info, indices);
}

void CpuPool2d::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No tensors provided");

    if(_asm_glue)
    {
        const auto hints = (_is_global_pooling_layer) ? Window::DimX : Window::DimY;
        NEScheduler::get().schedule_op(_asm_glue.get(), hints, _asm_glue->window(), tensors);
    }
    else
    {
        switch(_data_layout)
        {
            case DataLayout::NCHW:
                NEScheduler::get().schedule_op(_pooling_layer_kernel.get(), _is_global_pooling_layer ? Window::DimZ : Window::DimY, _pooling_layer_kernel->window(), tensors);
                break;
            case DataLayout::NHWC:
                NEScheduler::get().schedule_op(_pooling_layer_kernel.get(), Window::DimX, _pooling_layer_kernel->window(), tensors);
                break;
            default:
                ARM_COMPUTE_ERROR("Data layout not supported");
        }
    }
}

experimental::MemoryRequirements CpuPool2d::workspace() const
{
    return _aux_mem;
}
} // namespace cpu
} // namespace arm_compute
