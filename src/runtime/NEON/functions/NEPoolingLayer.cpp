/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "src/core/NEON/kernels/NEPoolingLayerKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
NEPoolingLayer::~NEPoolingLayer() = default;

NEPoolingLayer::NEPoolingLayer()
    : _pooling_layer_kernel(), _border_handler(), _is_global_pooling_layer(false), _data_layout(DataLayout::NCHW)
{
}

void NEPoolingLayer::configure(ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info, ITensor *indices)
{
    // Check if we have Global Pooling Layer
    _is_global_pooling_layer = (input->info()->dimension(0) == pool_info.pool_size.width) && (input->info()->dimension(1) == pool_info.pool_size.height);

    // Get data layout
    _data_layout = pool_info.data_layout == DataLayout::UNKNOWN ? input->info()->data_layout() : pool_info.data_layout;

    // Configure pooling kernel
    _pooling_layer_kernel = arm_compute::support::cpp14::make_unique<NEPoolingLayerKernel>();
    _pooling_layer_kernel->configure(input, output, pool_info, indices);

    switch(_data_layout)
    {
        case DataLayout::NCHW:
        {
            // Configure border depending on operation required (quantize border in case of asymmetric data_type)
            BorderMode border_mode = (!indices && pool_info.pool_type == PoolingType::MAX) ? BorderMode::REPLICATE : BorderMode::CONSTANT;
            PixelValue zero_value((indices) ? std::numeric_limits<int>::min() : 0.f);
            if(is_data_type_quantized_asymmetric(input->info()->data_type()) && !pool_info.exclude_padding)
            {
                zero_value = PixelValue(0, input->info()->data_type(), input->info()->quantization_info());
            }
            _border_handler = arm_compute::support::cpp14::make_unique<NEFillBorderKernel>();
            _border_handler->configure(input, _pooling_layer_kernel->border_size(), border_mode, zero_value);
            break;
        }
        case DataLayout::NHWC:
            break;
        default:
            ARM_COMPUTE_ERROR("Data layout not supported");
    }
}

Status NEPoolingLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    return NEPoolingLayerKernel::validate(input, output, pool_info, indices);
}

void NEPoolingLayer::run()
{
    switch(_data_layout)
    {
        case DataLayout::NCHW:
            // Fill border
            NEScheduler::get().schedule(_border_handler.get(), Window::DimY);

            // Run pooling layer
            NEScheduler::get().schedule(_pooling_layer_kernel.get(), _is_global_pooling_layer ? Window::DimZ : Window::DimY);
            break;
        case DataLayout::NHWC:
            // Run pooling layer
            NEScheduler::get().schedule(_pooling_layer_kernel.get(), Window::DimX);
            break;
        default:
            ARM_COMPUTE_ERROR("Data layout not supported");
    }
}

} // namespace arm_compute
