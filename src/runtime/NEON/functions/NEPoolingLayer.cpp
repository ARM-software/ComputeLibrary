/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#include "support/ToolchainSupport.h"

using namespace arm_compute;

NEPoolingLayer::NEPoolingLayer()
    : _pooling_layer_kernel(), _border_handler(), _is_global_pooling_layer(false)
{
}

void NEPoolingLayer::configure(ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info)
{
    // Check if we have Global Pooling Layer
    _is_global_pooling_layer = (input->info()->dimension(0) == pool_info.pool_size()) && (input->info()->dimension(1) == pool_info.pool_size());

    // Configure pooling kernel
    _pooling_layer_kernel.configure(input, output, pool_info);

    // Configure border depending on operation required (quantize border in case of asymmetric data_type)
    BorderMode border_mode = (pool_info.pool_type() == PoolingType::MAX) ? BorderMode::REPLICATE : BorderMode::CONSTANT;
    PixelValue zero_value(0.f);
    if(is_data_type_quantized_asymmetric(input->info()->data_type()) && !pool_info.exclude_padding())
    {
        zero_value = PixelValue(static_cast<uint32_t>(input->info()->quantization_info().offset));
    }
    _border_handler.configure(input, _pooling_layer_kernel.border_size(), border_mode, zero_value);
}

Status NEPoolingLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    return NEPoolingLayerKernel::validate(input, output, pool_info);
}

void NEPoolingLayer::run()
{
    // Fill border
    NEScheduler::get().schedule(&_border_handler, Window::DimY);

    // Run pooling layer
    NEScheduler::get().schedule(&_pooling_layer_kernel, _is_global_pooling_layer ? Window::DimZ : Window::DimY);
}