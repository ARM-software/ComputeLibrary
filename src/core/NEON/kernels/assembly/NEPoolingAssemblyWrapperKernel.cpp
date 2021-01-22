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
#include "src/core/NEON/kernels/assembly/NEPoolingAssemblyWrapperKernel.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;

void NEPoolingAssemblyWrapperKernel::configure(const ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &info, const CPUInfo &cpu_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output initialization if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_pool_shape(*input, info)));

    const bool requantize = input->quantization_info() != output->quantization_info();

    switch(input->data_type())
    {
        case DataType::QASYMM8:
            if(requantize)
            {
                create_arm_pooling_requant<uint8_t, uint8_t>(input, output, info, cpu_info);
            }
            else
            {
                create_arm_pooling<uint8_t, uint8_t>(input, output, info, cpu_info);
            }
            break;
        case DataType::QASYMM8_SIGNED:
            if(requantize)
            {
                create_arm_pooling_requant<int8_t, int8_t>(input, output, info, cpu_info);
            }
            else
            {
                create_arm_pooling<int8_t, int8_t>(input, output, info, cpu_info);
            }
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            create_arm_pooling<float16_t, float16_t>(input, output, info, cpu_info);
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
            create_arm_pooling<float, float>(input, output, info, cpu_info);
            break;
        default:
            break;
    }

    Window win = calculate_max_window(*output, Steps());
    INEKernel::configure(win);
}

Status NEPoolingAssemblyWrapperKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

#ifndef __aarch64__
    ARM_COMPUTE_RETURN_ERROR_MSG("32-bit is not supported by assembly kernels");
#endif /* __aarch64__ */
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((input->data_layout() != DataLayout::NHWC) || (info.data_layout != DataLayout::NHWC), "Only NHWC is supported by assembly kernels");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((info.pool_type != PoolingType::AVG) && (info.pool_type != PoolingType::MAX),
                                    "Only AVG and MAX pooling are supported by assembly kernels");

    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

        const auto input_qinfo  = input->quantization_info().uniform();
        const auto output_qinfo = output->quantization_info().uniform();

        if(input_qinfo != output_qinfo)
        {
            const float multiplier = input_qinfo.scale / output_qinfo.scale;
            int32_t     output_multiplier{};
            int32_t     output_shift{};
            ARM_COMPUTE_RETURN_ERROR_ON(quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift));
        }
        else
        {
            if(input->data_type() == DataType::QASYMM8)
            {
                const bool has_padding = info.pad_stride_info.has_padding();
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(!info.exclude_padding && has_padding, "Assembly kernels do not support padding for QASYMM8 with same input/output quantization info");
            }
        }
    }
    else
    {
        if(input->data_type() == DataType::QASYMM8)
        {
            // If output is not configured, the quantization info are the same
            const bool has_padding = info.pad_stride_info.has_padding();
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(!info.exclude_padding && has_padding, "Assembly kernels do not support padding for QASYMM8 with same input/output quantization info");
        }
    }
    return Status{};
}

void NEPoolingAssemblyWrapperKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_kernel_asm.get());
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(info);

    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *input     = tensors.get_const_tensor(TensorType::ACL_SRC);
    ITensor       *output    = tensors.get_tensor(TensorType::ACL_DST_0);
    ITensor       *workspace = tensors.get_tensor(TensorType::ACL_DST_1);

    const auto in_ptr        = input->buffer() + input->info()->offset_first_element_in_bytes();
    auto       out_ptr       = output->buffer() + output->info()->offset_first_element_in_bytes();
    auto       working_space = workspace->buffer() + workspace->info()->offset_first_element_in_bytes();

    _kernel_asm->execute(in_ptr, out_ptr, working_space, info.thread_id, info.num_threads);
}

size_t NEPoolingAssemblyWrapperKernel::get_working_size(unsigned int num_threads) const
{
    return _kernel_asm->get_working_size(num_threads);
}

bool NEPoolingAssemblyWrapperKernel::is_configured() const
{
    return _kernel_asm != nullptr;
}

template <typename TypeInput, typename TypeOutput>
void NEPoolingAssemblyWrapperKernel::create_arm_pooling(const ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &info, const CPUInfo &cpu_info)
{
    const arm_conv::pooling::PoolingType pool_type = (info.pool_type == PoolingType::AVG) ? arm_conv::pooling::PoolingType::AVERAGE : arm_conv::pooling::PoolingType::MAX;

    arm_conv::pooling::PoolingWindow window{};
    window.cols = static_cast<unsigned int>(info.pool_size.x());
    window.rows = static_cast<unsigned int>(info.pool_size.y());

    arm_conv::pooling::PoolingStride stride{};
    std::tie(stride.cols, stride.rows) = info.pad_stride_info.stride();

    const arm_conv::pooling::PaddingValues padding{ info.pad_stride_info.pad_left(), info.pad_stride_info.pad_top(), info.pad_stride_info.pad_right(), info.pad_stride_info.pad_bottom() };

    constexpr unsigned int idx_width    = 1;
    constexpr unsigned int idx_height   = 2;
    constexpr unsigned int idx_channels = 0;
    constexpr unsigned int idx_batches  = 3;

    const unsigned int n_batches   = input->dimension(idx_batches);
    const unsigned int input_rows  = input->dimension(idx_height);
    const unsigned int input_cols  = input->dimension(idx_width);
    const unsigned int n_channels  = input->dimension(idx_channels);
    const unsigned int output_rows = output->dimension(idx_height);
    const unsigned int output_cols = output->dimension(idx_width);

    arm_conv::pooling::PoolingArgs args(&cpu_info, pool_type, window, stride, info.exclude_padding, n_batches, input_rows, input_cols, n_channels, output_rows, output_cols, padding, nullptr);

    // Configure assembly pooling kernel
    auto pooling_kernel_asm = arm_conv::pooling::pooling<TypeInput, TypeOutput>(args);
    if(pooling_kernel_asm == nullptr)
    {
        // Configuration not supported: Leave function unconfigured:
        return;
    }

    _kernel_asm = std::move(pooling_kernel_asm);
}

template <typename TypeInput, typename TypeOutput>
void NEPoolingAssemblyWrapperKernel::create_arm_pooling_requant(const ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &info, const CPUInfo &cpu_info)
{
    const arm_conv::pooling::PoolingType pool_type = (info.pool_type == PoolingType::AVG) ? arm_conv::pooling::PoolingType::AVERAGE : arm_conv::pooling::PoolingType::MAX;

    arm_conv::pooling::PoolingWindow window{};
    window.cols = static_cast<unsigned int>(info.pool_size.x());
    window.rows = static_cast<unsigned int>(info.pool_size.y());

    arm_conv::pooling::PoolingStride stride{};
    std::tie(stride.cols, stride.rows) = info.pad_stride_info.stride();

    const arm_conv::pooling::PaddingValues padding{ info.pad_stride_info.pad_left(), info.pad_stride_info.pad_top(), info.pad_stride_info.pad_right(), info.pad_stride_info.pad_bottom() };

    constexpr unsigned int idx_width    = 1;
    constexpr unsigned int idx_height   = 2;
    constexpr unsigned int idx_channels = 0;
    constexpr unsigned int idx_batches  = 3;

    const unsigned int n_batches   = input->dimension(idx_batches);
    const unsigned int input_rows  = input->dimension(idx_height);
    const unsigned int input_cols  = input->dimension(idx_width);
    const unsigned int n_channels  = input->dimension(idx_channels);
    const unsigned int output_rows = output->dimension(idx_height);
    const unsigned int output_cols = output->dimension(idx_width);

    arm_conv::pooling::PoolingArgs args(&cpu_info, pool_type, window, stride, info.exclude_padding, n_batches, input_rows, input_cols, n_channels, output_rows, output_cols, padding, nullptr);

    const auto input_qinfo  = input->quantization_info().uniform();
    const auto output_qinfo = output->quantization_info().uniform();

    const float multiplier = input_qinfo.scale / output_qinfo.scale;
    int32_t     output_multiplier{};
    int32_t     output_shift{};
    quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);

    const arm_conv::pooling::Requantize32 requant_args(input_qinfo.offset,
                                                       output_qinfo.offset,
                                                       output_shift, // left shift
                                                       0,            // right shift
                                                       output_multiplier);

    // Configure assembly pooling kernel with requantization
    auto pooling_kernel_asm = arm_conv::pooling::pooling<TypeInput, TypeOutput, arm_conv::pooling::Requantize32>(args, requant_args);
    if(pooling_kernel_asm == nullptr)
    {
        // Configuration not supported: Leave function unconfigured:
        return;
    }

    _kernel_asm = std::move(pooling_kernel_asm);
}
} // namespace arm_compute
