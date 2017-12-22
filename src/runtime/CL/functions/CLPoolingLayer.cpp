/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLPoolingLayerKernel.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

void CLPoolingLayer::configure(ICLTensor *input, ICLTensor *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    // Configure pooling kernel
    auto k = arm_compute::support::cpp14::make_unique<CLPoolingLayerKernel>();
    k->set_target(CLScheduler::get().target());
    k->configure(input, output, pool_info);
    _kernel = std::move(k);

    // Configure border depending on operation required
    BorderMode border_mode = (PoolingType::MAX == pool_info.pool_type()) ? BorderMode::REPLICATE : BorderMode::CONSTANT;
    // Quantize border in case data type is quantized asymmetric data type
    uint32_t border_value = 0;
    if(is_data_type_quantized_asymmetric(input->info()->data_type()) && !pool_info.exclude_padding())
    {
        border_value = static_cast<uint32_t>(input->info()->quantization_info().quantize(0.f, RoundingPolicy::TO_NEAREST_UP));
    }

    _border_handler.configure(input, _kernel->border_size(), border_mode, PixelValue(border_value));
}

Status CLPoolingLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    return CLPoolingLayerKernel::validate(input, output, pool_info);
}