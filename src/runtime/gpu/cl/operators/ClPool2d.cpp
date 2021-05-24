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
#include "src/runtime/gpu/cl/operators/ClPool2d.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/gpu/cl/ClCompileContext.h"
#include "src/core/gpu/cl/kernels/ClPool2dKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClPool2d::configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &info, ITensorInfo *indices)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    // Configure pooling kernel
    auto k = std::make_unique<kernels::ClPool2dKernel>();
    k->set_target(CLScheduler::get().target());
    k->configure(compile_context, src, dst, info, indices);
    _pooling = std::move(k);

    const DataType data_type = src->data_type();

    // Configure border depending on operation required (quantize border in case of asymmetric data_type)
    BorderMode border_mode{};
    PixelValue pixel_value(0.f);
    if(is_data_type_quantized_asymmetric(data_type) && !info.exclude_padding)
    {
        pixel_value = PixelValue(0, data_type, src->quantization_info());
    }

    // Data layout
    const auto data_layout = info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : info.data_layout;

    switch(data_layout)
    {
        case DataLayout::NCHW:
            border_mode = (PoolingType::MAX == info.pool_type) ? BorderMode::REPLICATE : BorderMode::CONSTANT;
            break;
        case DataLayout::NHWC:
            border_mode = BorderMode::CONSTANT;
            if(PoolingType::MAX == info.pool_type)
            {
                if(is_data_type_quantized(data_type))
                {
                    std::tie(pixel_value, std::ignore) = get_min_max(data_type);
                }
                else
                {
                    pixel_value = PixelValue(std::numeric_limits<float>::lowest());
                }
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Data layout not supported");
    }
    auto b = std::make_unique<CLFillBorderKernel>();
    b->configure(compile_context, src, _pooling->border_size(), border_mode, pixel_value);
    _border_handler = std::move(b);

    // Tune kernels
    CLScheduler::get().tune_kernel_static(*_pooling);
}

Status ClPool2d::validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &info, const ITensorInfo *indices)
{
    return kernels::ClPool2dKernel::validate(src, dst, info, indices);
}

void ClPool2d::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");

    CLScheduler::get().enqueue_op(*_border_handler.get(), tensors, false);
    CLScheduler::get().enqueue_op(*_pooling.get(), tensors, false);
}
} // namespace opencl
} // namespace arm_compute
