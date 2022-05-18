/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#include "src/cpu/kernels/CpuDepthwiseConv2dNativeKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/wrapper/traits.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/depthwiseconv2d/list.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
static const std::vector<CpuDepthwiseConv2dNativeKernel::DepthwiseConv2dNativeKernel> available_kernels =
{
    {
        "neon_qu8_deptwiseconv2dnative",
        [](const DepthwiseConv2dNativeDataTypeISASelectorData & data)
        {
            return (data.weights_dt == DataType::QASYMM8);
        },
        REGISTER_QASYMM8_NEON(neon_qu8_deptwiseconv2dnative)
    },
    {
        "neon_qs8_deptwiseconv2dnative",
        [](const DepthwiseConv2dNativeDataTypeISASelectorData & data)
        {
            return (data.weights_dt == DataType::QASYMM8_SIGNED);
        },
        REGISTER_QASYMM8_SIGNED_NEON(neon_qs8_deptwiseconv2dnative)
    },
    {
        "neon_fp16_deptwiseconv2dnative",
        [](const DepthwiseConv2dNativeDataTypeISASelectorData & data)
        {
            return (data.weights_dt == DataType::F16 && data.isa.fp16);
        },
        REGISTER_FP16_NEON(neon_fp16_deptwiseconv2dnative)
    },
    {
        "neon_fp32_deptwiseconv2dnative",
        [](const DepthwiseConv2dNativeDataTypeISASelectorData & data)
        {
            return (data.weights_dt == DataType::F32);
        },
        REGISTER_FP32_NEON(neon_fp32_deptwiseconv2dnative)
    },
    {
        "neon_qp8_qu8_deptwiseconv2dnative",
        [](const DepthwiseConv2dNativeDataTypeISASelectorData & data)
        {
            return (data.weights_dt == DataType::QSYMM8_PER_CHANNEL && data.source_dt == DataType::QASYMM8);
        },
        REGISTER_QASYMM8_NEON(neon_qp8_qu8_deptwiseconv2dnative)
    },
    {
        "neon_qp8_qs8_deptwiseconv2dnative",
        [](const DepthwiseConv2dNativeDataTypeISASelectorData & data)
        {
            return (data.weights_dt == DataType::QSYMM8_PER_CHANNEL && data.source_dt != DataType::QASYMM8);
        },
        REGISTER_QASYMM8_SIGNED_NEON(neon_qp8_qs8_deptwiseconv2dnative)
    },
};

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ConvolutionInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(info.depth_multiplier == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(1) + (weights->dimension(1) - 1) * (info.dilation.x() - 1) > src->dimension(1) + info.pad_stride_info.pad_left() + info.pad_stride_info.pad_right());
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(2) + (weights->dimension(2) - 1) * (info.dilation.y() - 1) > src->dimension(2) + info.pad_stride_info.pad_top() + info.pad_stride_info.pad_bottom());
    ARM_COMPUTE_RETURN_ERROR_ON((src->dimension(0) * info.depth_multiplier) != weights->dimension(0));
    ARM_COMPUTE_RETURN_ERROR_ON((info.dilation.x() < 1) || (info.dilation.y() < 1));
    ARM_COMPUTE_RETURN_ERROR_ON((info.pad_stride_info.stride().first < 1) || (info.pad_stride_info.stride().second < 1));

    if(is_data_type_quantized_per_channel(weights->data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QSYMM8_PER_CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(0) != weights->quantization_info().scale().size());
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);
    }

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(0));

        if(is_data_type_quantized_asymmetric(src->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        }
    }

    if(dst->total_size() != 0)
    {
        const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*src, *weights, info);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }

    return Status{};
}
} // namespace

void CpuDepthwiseConv2dNativeKernel::configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ConvolutionInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, (biases != nullptr) ? biases : nullptr, dst, info));

    _has_biases = (biases != nullptr);
    _conv_info  = info;

    const auto uk = CpuDepthwiseConv2dNativeKernel::get_implementation(
                        DepthwiseConv2dNativeDataTypeISASelectorData{ weights->data_type(), src->data_type(), CPUInfo::get().get_isa() });
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);
    _func = uk->ukernel;

    const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*src, *weights, info);
    auto_init_if_empty(*dst, src->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape).set_quantization_info(dst->quantization_info()));

    Window win = calculate_max_window(*dst, Steps());
    ICpuKernel::configure(win);
}

Status CpuDepthwiseConv2dNativeKernel::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ConvolutionInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, biases, dst, info));
    return Status{};
}

void CpuDepthwiseConv2dNativeKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    const auto src     = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto weights = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto biases  = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto       dst     = tensors.get_tensor(TensorType::ACL_DST);
    _func(src, weights, biases, dst, window, _has_biases, _conv_info);
}

const char *CpuDepthwiseConv2dNativeKernel::name() const
{
    return "CpuDepthwiseConv2dNativeKernel";
}

const std::vector<CpuDepthwiseConv2dNativeKernel::DepthwiseConv2dNativeKernel> &CpuDepthwiseConv2dNativeKernel::get_available_kernels()
{
    return available_kernels;
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
