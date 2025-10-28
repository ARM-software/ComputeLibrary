/*
 * Copyright (c) 2017-2021, 2024-2025 Arm Limited.
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
#include "src/cpu/kernels/CpuDirectConv2dOutputStageKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/common/utils/profile/acl_profile.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/directconv2d_output_stage/list.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo                                 *src,
                          const ITensorInfo                                 *bias,
                          const ITensorInfo                                 *dst,
                          const DirectConvolutionLayerOutputStageKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_SIZE_UNSUPPORTED(src, bias);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::S32, DataType::F32);

    if (bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->dimension(0) != src->dimension(get_data_layout_dimension_index(
                                                              src->data_layout(), DataLayoutDimension::CHANNEL)));
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
    }

    if (src->data_type() == DataType::S32)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst == nullptr, "In-place computation not allowed for quantized output");
    }

    // Checks performed when output is configured
    if ((dst != nullptr) && (dst->total_size() != 0))
    {
        if (is_data_type_float(src->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_SIZE_UNSUPPORTED(dst);
    }
    else if (src->data_type() == DataType::S32)
    {
        // In case of quantized computation and unconfigured output, the output data type must be provided through DirectConvolutionLayerOutputStageKernelInfo
        ARM_COMPUTE_RETURN_ERROR_ON((info.output_data_type != DataType::QASYMM8) &&
                                    (info.output_data_type != DataType::QASYMM8_SIGNED));
    }

    return Status{};
}
} // namespace

void CpuDirectConv2dOutputStageKernel::configure(ITensorInfo                                       *src,
                                                 const ITensorInfo                                 *bias,
                                                 ITensorInfo                                       *dst,
                                                 const DirectConvolutionLayerOutputStageKernelInfo &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU,
                            "CpuDirectConv2dOutputStageKernel::configure");
    ARM_COMPUTE_UNUSED(bias);
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, bias, dst, info));

    _func                         = nullptr;
    _result_fixedpoint_multiplier = info.result_fixedpoint_multiplier;
    _result_shift                 = info.result_shift;
    _result_offset_after_shift    = info.result_offset_after_shift;

    // Auto-initialize output output if required
    if (dst != nullptr)
    {
        // Work out expected output data type
        const DataType output_dt = (src->data_type() == DataType::S32) ? info.output_data_type : DataType::S32;
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*dst, src->clone()->set_data_type(output_dt));
    }

    Window win = calculate_max_window(*src, Steps());

    ICpuKernel::configure(win);

    const bool is_qasymm8_signed =
        (dst != nullptr) ? is_data_type_quantized_asymmetric_signed(dst->data_type()) : false;

    // Set appropriate function
    if (src->data_layout() == DataLayout::NCHW)
    {
        switch (src->data_type())
        {
            case DataType::S32:
            {
                if (is_qasymm8_signed)
                {
#ifdef ENABLE_QASYMM8_SIGNED_KERNELS
                    _func = &output_stage_nchw_qs8;
#endif // ENABLE_QASYMM8_SIGNED_KERNELS
                }
                else
                {
#ifdef ENABLE_QASYMM8_KERNELS
                    _func = &output_stage_nchw_qu8;
#endif // ENABLE_QASYMM8_KERNELS
                }
                break;
            }
#ifdef ENABLE_FP16_KERNELS
            case DataType::F16:
            {
                _func = &output_stage_nchw_fp16;
                break;
            }
#endif // ENABLE_FP16_KERNELS
            case DataType::F32:
            {
#ifdef ENABLE_FP32_KERNELS
                _func = &output_stage_nchw_fp32;
#endif // ENABLE_FP32_KERNELS
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported combination of types among the inputs.");
            }
        }
    }
    else
    {
        switch (src->data_type())
        {
            case DataType::S32:
            {
                if (is_qasymm8_signed)
                {
#ifdef ENABLE_QASYMM8_SIGNED_KERNELS
                    _func = &output_stage_nhwc_qs8;
#endif // ENABLE_QASYMM8_SIGNED_KERNELS
                }
                else
                {
#ifdef ENABLE_QASYMM8_KERNELS
                    _func = &output_stage_nhwc_qu8;
#endif // QASYMM8_SIGNED_KERNELS
                }
                break;
            }
#ifdef ENABLE_FP16_KERNELS
            case DataType::F16:
            {
                _func = &output_stage_nhwc_fp16;
                break;
            }
#endif // ENABLE_FP16_KERNELS
            case DataType::F32:
            {
#ifdef ENABLE_FP32_KERNELS
                _func = &output_stage_nhwc_fp32;
#endif // ENABLE_FP32_KERNELS
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported combination of types among the inputs.");
            }
        }
    }
}

Status CpuDirectConv2dOutputStageKernel::validate(const ITensorInfo                                 *src,
                                                  const ITensorInfo                                 *bias,
                                                  const ITensorInfo                                 *dst,
                                                  const DirectConvolutionLayerOutputStageKernelInfo &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU,
                            "CpuDirectConv2dOutputStageKernel::validate");
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, bias, dst, info));
    return Status{};
}

void CpuDirectConv2dOutputStageKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU,
                            "CpuDirectConv2dOutputStageKernel::run_op");
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    auto src  = tensors.get_tensor(TensorType::ACL_SRC_0);
    auto bias = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto dst  = tensors.get_tensor(TensorType::ACL_DST);

    (*_func)(src, bias, window, dst, _result_fixedpoint_multiplier, _result_shift, _result_offset_after_shift);
}

const char *CpuDirectConv2dOutputStageKernel::name() const
{
    return "CpuDirectConv2dOutputStageKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
