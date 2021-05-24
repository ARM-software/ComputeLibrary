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
#include "src/runtime/cpu/operators/CpuPool2d.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "src/core/cpu/kernels/CpuPool2dKernel.h"
#include "src/core/cpu/kernels/internal/CpuPool2dAssemblyWrapperKernel.h"

namespace arm_compute
{
namespace cpu
{
CpuPool2d::CpuPool2d()
    : _pooling_layer_kernel(),
      _border_handler(),
      _asm_glue(),
      _is_global_pooling_layer(false),
      _data_layout(DataLayout::NCHW),
      _mem_req()
{
}

CpuPool2d::~CpuPool2d() = default;

void CpuPool2d::configure(ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &pool_info, ITensorInfo *indices)
{
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
        _mem_req.push_back({ TensorType::ACL_INT_0, workspace_size, alignment });

        _asm_glue = std::move(pooling_wrapper);
    }
    else
    {
        // Configure pooling kernel
        auto k = std::make_unique<kernels::CpuPool2dKernel>();
        k->configure(src, dst, pool_info, indices);
        _pooling_layer_kernel = std::move(k);

        switch(_data_layout)
        {
            case DataLayout::NCHW:
            {
                // Configure border depending on operation required (quantize border in case of asymmetric data_type)
                BorderMode border_mode = (!indices && pool_info.pool_type == PoolingType::MAX) ? BorderMode::REPLICATE : BorderMode::CONSTANT;
                PixelValue zero_value((indices) ? std::numeric_limits<int>::min() : 0.f);
                if(is_data_type_quantized_asymmetric(src->data_type()) && !pool_info.exclude_padding)
                {
                    zero_value = PixelValue(0, src->data_type(), src->quantization_info());
                }
                auto b = std::make_unique<NEFillBorderKernel>();
                b->configure(src, _pooling_layer_kernel->border_size(), border_mode, zero_value);
                _border_handler = std::move(b);
                break;
            }
            case DataLayout::NHWC:
                break;
            default:
                ARM_COMPUTE_ERROR("Data layout not supported");
        }
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
                // Fill border
                NEScheduler::get().schedule_op(_border_handler.get(), Window::DimY, _border_handler->window(), tensors);

                // Run pooling layer
                NEScheduler::get().schedule_op(_pooling_layer_kernel.get(), _is_global_pooling_layer ? Window::DimZ : Window::DimY, _pooling_layer_kernel->window(), tensors);
                break;
            case DataLayout::NHWC:
                // Run pooling layer
                NEScheduler::get().schedule_op(_pooling_layer_kernel.get(), Window::DimX, _pooling_layer_kernel->window(), tensors);
                break;
            default:
                ARM_COMPUTE_ERROR("Data layout not supported");
        }
    }
}

experimental::MemoryRequirements CpuPool2d::workspace() const
{
    return _mem_req;
}
} // namespace cpu
} // namespace arm_compute
