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
#include "src/gpu/cl/operators/ClDirectConv2d.h"

#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/kernels/ClActivationKernel.h"
#include "src/gpu/cl/kernels/ClDirectConv2dKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace
{
ITensorPack select_activation_src_dst(ITensorPack &tensors)
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, tensors.get_tensor(TensorType::ACL_DST));
    pack.add_tensor(TensorType::ACL_DST, tensors.get_tensor(TensorType::ACL_DST));
    return pack;
}
} // namespace

void ClDirectConv2d::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                               const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);

    // Configure direct convolution kernel
    const ActivationLayerInfo conv2d_act_info = (src->data_layout() == DataLayout::NHWC && is_data_type_float(src->data_type())) ? act_info : ActivationLayerInfo();
    auto                      k               = std::make_unique<kernels::ClDirectConv2dKernel>();
    k->set_target(CLScheduler::get().target());
    k->configure(compile_context, src, weights, biases, dst, conv_info, conv2d_act_info);
    _direct_conv_kernel = std::move(k);

    // Configure border handler
    PixelValue zero_value(0.f);
    if(is_data_type_quantized_asymmetric(src->data_type()))
    {
        zero_value = PixelValue(0, src->data_type(), src->quantization_info());
    }
    auto b = std::make_unique<CLFillBorderKernel>();
    b->configure(compile_context, src, _direct_conv_kernel->border_size(), BorderMode::CONSTANT, zero_value);
    _src_border_handler = std::move(b);

    // Fused activation is currently supported for NHWC and floating point types
    if(act_info.enabled() && !conv2d_act_info.enabled())
    {
        auto a = std::make_unique<kernels::ClActivationKernel>();
        a->configure(compile_context, dst, dst, act_info);
        _activation_kernel = std::move(a);
    }

    // Tune kernels
    CLScheduler::get().tune_kernel_static(*_direct_conv_kernel);
}

Status ClDirectConv2d::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClDirectConv2dKernel::validate(src, weights, biases, dst, conv_info, ActivationLayerInfo(), CLScheduler::get().target()));
    if(act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClActivationKernel::validate(dst, dst, act_info));
    }
    return Status{};
}

void ClDirectConv2d::run(ITensorPack &tensors)
{
    // Run border handler
    CLScheduler::get().enqueue_op(*_src_border_handler.get(), tensors, false);
    // Run direct convolution
    CLScheduler::get().enqueue_op(*_direct_conv_kernel.get(), tensors, false);
    // Run activation kernel
    if(_activation_kernel)
    {
        auto act_pack = select_activation_src_dst(tensors);
        CLScheduler::get().enqueue_op(*_activation_kernel.get(), act_pack, false);
    }
}
} // namespace opencl
} // namespace arm_compute
