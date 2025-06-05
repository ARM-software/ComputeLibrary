/*
 * Copyright (c) 2017-2022, 2024-2025 Arm Limited.
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
#include "src/cpu/kernels/CpuQuantizeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/common/utils/profile/acl_profile.h"
#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/quantize/generic/neon/list.h"

#include <arm_neon.h>
#include <map>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);
    if (src->data_type() == DataType::F32)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::QSYMM8, DataType::QASYMM8,
                                                             DataType::QASYMM8_SIGNED, DataType::QASYMM16,
                                                             DataType::QSYMM8_PER_CHANNEL);

        if (dst->data_type() == DataType::QSYMM8_PER_CHANNEL)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(
                dst->quantization_info().scale().size() !=
                dst->tensor_shape()[get_data_layout_dimension_index(dst->data_layout(), DataLayoutDimension::CHANNEL)]);
        }
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::QSYMM8, DataType::QASYMM8,
                                                             DataType::QASYMM8_SIGNED, DataType::QASYMM16);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);

    return Status{};
}

} // namespace

void CpuQuantizeKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuQuantizeKernel::configure");
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

    static const std::map<std::string, QuantizeFunctionExecutorPtr> quant_map = {
        {"op_QASYMM8_QASYMM8", REGISTER_INTEGER_NEON(u8_u8_run_quantize_qasymm8)},
        {"op_QASYMM8_QASYMM8_SIGNED", REGISTER_INTEGER_NEON(u8_i8_run_quantize_qasymm8)},
        {"op_QASYMM8_QASYMM16", REGISTER_INTEGER_NEON(u8_run_quantize_qasymm16)},

        {"op_QASYMM8_SIGNED_QASYMM8", REGISTER_INTEGER_NEON(i8_u8_run_quantize_qasymm8)},
        {"op_QASYMM8_SIGNED_QASYMM8_SIGNED", REGISTER_INTEGER_NEON(i8_i8_run_quantize_qasymm8)},
        {"op_QASYMM8_SIGNED_QASYMM16", REGISTER_INTEGER_NEON(i8_run_quantize_qasymm16)},

        // Functions for offset only requantization
        {"op_OFFSET_ONLY_QASYMM8_QASYMM8", REGISTER_INTEGER_NEON(u8_u8_run_requantize_offset_only)},
        {"op_OFFSET_ONLY_QASYMM8_QASYMM8_SIGNED", REGISTER_INTEGER_NEON(u8_i8_run_requantize_offset_only)},
        {"op_OFFSET_ONLY_QASYMM8_SIGNED_QASYMM8", REGISTER_INTEGER_NEON(i8_u8_run_requantize_offset_only)},
        {"op_OFFSET_ONLY_QASYMM8_SIGNED_QASYMM8_SIGNED", REGISTER_INTEGER_NEON(i8_i8_run_requantize_offset_only)},

        // Functions for offset uint8 to int8 and vice versa quantization (no scale changes)
        {"op_OFFSET_ONLY_CONVERT_QASYMM8_SIGNED_QASYMM8",
         REGISTER_INTEGER_NEON(i8_u8_run_requantize_offset_only_convert)},
        {"op_OFFSET_ONLY_CONVERT_QASYMM8_QASYMM8_SIGNED",
         REGISTER_INTEGER_NEON(u8_i8_run_requantize_offset_only_convert)},

        {"op_F32_QSYMM8", REGISTER_FP32_NEON(fp32_i8_run_quantize_qsymm8)},
        {"op_F32_QASYMM8", REGISTER_FP32_NEON(fp32_u8_run_quantize_qasymm8)},
        {"op_F32_QASYMM8_SIGNED", REGISTER_FP32_NEON(fp32_i8_run_quantize_qasymm8)},
        {"op_F32_QASYMM16", REGISTER_FP32_NEON(fp32_run_quantize_qasymm16)},
        {"op_F32_QSYMM8_PER_CHANNEL", REGISTER_FP32_NEON(fp32_i8_run_quantize_qsymm8_per_channel)},
#ifdef ARM_COMPUTE_ENABLE_FP16
        {"op_F16_QASYMM8", REGISTER_FP16_NEON(fp16_u8_run_quantize_qasymm8)},
        {"op_F16_QASYMM8_SIGNED", REGISTER_FP16_NEON(fp16_i8_run_quantize_qasymm8)},
        {"op_F16_QASYMM16", REGISTER_FP16_NEON(fp16_run_quantize_qasymm16)},
#endif /* ARM_COMPUTE_ENABLE_FP16 */
    };

    std::string function_to_call("op_");

    // For offset only functions - must be 8-bit and have identical scale values.
    if (src->quantization_info().scale() == dst->quantization_info().scale() &&
        (is_data_type_quantized_asymmetric_char(src->data_type()) &&
         is_data_type_quantized_asymmetric_char(dst->data_type())))
    {
        function_to_call += "OFFSET_ONLY_";
        // For optimized datatype conversion 8-bit re-quantization offset only functions.
        // These must have an offset of exactly 128 to match requirements - has specific circumstances to match use case.
        auto uqinfo =
            compute_requantization_scale_offset(src->quantization_info().uniform(), dst->quantization_info().uniform());
        const auto src_dt = src->data_type();
        if (src->data_type() != dst->data_type() && ((src_dt == DataType::QASYMM8_SIGNED && uqinfo.offset == 128) ||
                                                     (src_dt == DataType::QASYMM8 && uqinfo.offset == -128)))
        {
            function_to_call += "CONVERT_";
        }
    }

    // Specify datatype for function
    function_to_call += string_from_data_type(src->data_type()) + "_";
    function_to_call += string_from_data_type(dst->data_type());
    auto it = quant_map.find(function_to_call);

    if (it == quant_map.end())
    {
        ARM_COMPUTE_ERROR("Unsupported combination of input and output data types");
    }
    _func = it->second;

    // Calculate window. Squash if possible.
    Window win;
    if (dst->data_type() == DataType::QSYMM8_PER_CHANNEL)
    {
        // Bring back a full N-dimensional iteration (so channel coord actually goes 0…C-1):
        win              = calculate_max_window(*src);
        _split_dimension = Window::DimY;
    }
    else
    {
        std::tie(win, _split_dimension) = calculate_squashed_or_max_window(*src);
    }

    ICpuKernel::configure(win);
}

Status CpuQuantizeKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuQuantizeKernel::validate");
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
    return Status{};
}

void CpuQuantizeKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuQuantizeKernel::run_op");
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);
    (*_func)(src, dst, window);
}

const char *CpuQuantizeKernel::name() const
{
    return "CpuQuantizeKernel";
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
