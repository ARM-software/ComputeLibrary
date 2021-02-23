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
#include "src/runtime/cpu/operators/CpuPooling.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "src/core/cpu/kernels/CpuPoolingKernel.h"

namespace arm_compute
{
namespace cpu
{
CpuPooling::CpuPooling(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_manager(std::move(memory_manager)), _pooling_layer_kernel(), _border_handler(), _asm_glue(), _is_global_pooling_layer(false), _data_layout(DataLayout::NCHW)
{
}

CpuPooling::~CpuPooling() = default;

void CpuPooling::configure(ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &pool_info, ITensorInfo *indices)
{
    // Check if we can run assembly kernels. Currently, indices are not supported by those kernels
    const bool run_optimised = bool(CpuPoolingAssemblyDispatch::validate(input, output, pool_info)) && (indices == nullptr);

    if(run_optimised)
    {
        _asm_glue = std::make_unique<CpuPoolingAssemblyDispatch>(_memory_manager);
        _asm_glue->configure(input, output, pool_info);
        ARM_COMPUTE_ERROR_ON(!_asm_glue->is_configured());
    }
    else
    {
        // Check if we have Global Pooling Layer
        _is_global_pooling_layer = (input->dimension(0) == pool_info.pool_size.width) && (input->dimension(1) == pool_info.pool_size.height);

        // Get data layout
        _data_layout = pool_info.data_layout == DataLayout::UNKNOWN ? input->data_layout() : pool_info.data_layout;

        // Configure pooling kernel
        auto k = std::make_unique<kernels::CpuPoolingKernel>();
        k->configure(input, output, pool_info, indices);
        _pooling_layer_kernel = std::move(k);

        switch(_data_layout)
        {
            case DataLayout::NCHW:
            {
                // Configure border depending on operation required (quantize border in case of asymmetric data_type)
                BorderMode border_mode = (!indices && pool_info.pool_type == PoolingType::MAX) ? BorderMode::REPLICATE : BorderMode::CONSTANT;
                PixelValue zero_value((indices) ? std::numeric_limits<int>::min() : 0.f);
                if(is_data_type_quantized_asymmetric(input->data_type()) && !pool_info.exclude_padding)
                {
                    zero_value = PixelValue(0, input->data_type(), input->quantization_info());
                }
                auto b = std::make_unique<NEFillBorderKernel>();
                b->configure(input, _pooling_layer_kernel->border_size(), border_mode, zero_value);
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

Status CpuPooling::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    const bool run_optimised = bool(CpuPoolingAssemblyDispatch::validate(input, output, pool_info)) && (indices == nullptr);

    if(run_optimised)
    {
        return Status{};
    }

    return kernels::CpuPoolingKernel::validate(input, output, pool_info, indices);
}

void CpuPooling::run(ITensorPack &tensors)
{
    if(_asm_glue && _asm_glue->is_configured())
    {
        _asm_glue->run(tensors);
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
} // namespace cpu
} // namespace arm_compute
