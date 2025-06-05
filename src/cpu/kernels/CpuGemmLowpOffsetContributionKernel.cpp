/*
 * Copyright (c) 2017-2022,2024-2025 Arm Limited.
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
#include "src/cpu/kernels/CpuGemmLowpOffsetContributionKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/common/utils/profile/acl_profile.h"
#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/gemmlowp/generic/neon/impl.h"
#include "src/cpu/kernels/gemmlowp/generic/neon/list.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *mm_result,
                          const ITensorInfo *vector_sum_col,
                          const ITensorInfo *vector_sum_row,
                          int32_t            a_offset,
                          int32_t            b_offset)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mm_result, 1, DataType::S32, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(mm_result);

    // We run if the offset is nonzero or a sum col has been provided, we need
    // the second option in case the QuantizationInfo is dynamic
    if (a_offset != 0 || vector_sum_col != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_col, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(vector_sum_col->dimension(0) != mm_result->dimension(0));
    }

    // We run if the offset is nonzero or a sum row has been provided, we need
    // the second option in case the QuantizationInfo is dynamic
    if (b_offset != 0 || vector_sum_row != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_row, 1, DataType::S32);

        // Check if input is a 3D reinterpretation
        const bool reinterpret_as_3d =
            mm_result->num_dimensions() > 1 && mm_result->tensor_shape().y() != vector_sum_row->tensor_shape().x();

        // Validate input
        ARM_COMPUTE_RETURN_ERROR_ON(reinterpret_as_3d && vector_sum_row->dimension(0) !=
                                                             (mm_result->dimension(1) * mm_result->dimension(2)));
        ARM_COMPUTE_RETURN_ERROR_ON(!reinterpret_as_3d && vector_sum_row->dimension(0) != mm_result->dimension(1));

        TensorShape output_shape = mm_result->tensor_shape();
        if (output_shape.num_dimensions() > 1)
        {
            const unsigned int output_batch_idx = reinterpret_as_3d ? 3 : 2;

            TensorShape vector_sum_row_shape = vector_sum_row->tensor_shape();
            vector_sum_row_shape.collapse_from(1);
            output_shape.collapse_from(output_batch_idx);

            ARM_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_row_shape[1] != output_shape[output_batch_idx],
                                            "mm_result tensor must have the same number of batches of output tensor");

            if (vector_sum_col != nullptr)
            {
                TensorShape vector_sum_col_shape = vector_sum_col->tensor_shape();
                vector_sum_col_shape.collapse_from(1);

                ARM_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_col_shape[1] != 1 &&
                                                    vector_sum_col_shape[1] != vector_sum_row_shape[1],
                                                "vector_sum_col tensor must have the same number of batches of "
                                                "vector_sum_row_shape or the number of batches must be set to 1");
            }
        }
    }

    return Status{};
}

} // namespace

void CpuGemmLowpOffsetContributionKernel::configure(ITensorInfo *mm_result,
                                                    ITensorInfo *vector_sum_col,
                                                    ITensorInfo *vector_sum_row,
                                                    int32_t      k,
                                                    int32_t      a_offset,
                                                    int32_t      b_offset,
                                                    float        scale)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU,
                            "CpuGemmLowpOffsetContributionKernel::configure");
    // Perform validate step
    ARM_COMPUTE_UNUSED(vector_sum_row);
    ARM_COMPUTE_ERROR_ON_NULLPTR(mm_result);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(mm_result, vector_sum_col, vector_sum_row, a_offset, b_offset));

    switch (mm_result->data_type())
    {
        case DataType::F32:
        {
            _func = REGISTER_FP32_NEON(cpu::neon_run_offset_contribution_fp32);
            break;
        }
        case DataType::F16:
        {
            _func = REGISTER_FP16_NEON(cpu::neon_run_offset_contribution_fp16);
            break;
        }
        case DataType::S32:
        {
            _func = &cpu::neon_run_offset_contribution_int32;
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported");
            break;
        }
    }

    _a_offset = a_offset;
    _b_offset = b_offset;
    _k        = k;

    _scale = scale;

    if (vector_sum_col != nullptr)
    {
        // Check if vector_sum_col_shape should be slidden or not
        // Don't slide vector_sum_col_shape along the y dimension if vector_sum_col_shape has just 1 dimension and vector_sum_row_shape more than 1
        // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
        _slide_vector_sum_col = vector_sum_col->tensor_shape().num_dimensions() > 1;
    }

    // Configure kernel window
    Window win = calculate_max_window(*mm_result, Steps());
    ICpuKernel::configure(win);
}

void CpuGemmLowpOffsetContributionKernel::set_a_offset(int32_t a_offset)
{
    _a_offset = a_offset;
}

void CpuGemmLowpOffsetContributionKernel::set_b_offset(int32_t b_offset)
{
    _b_offset = b_offset;
}

void CpuGemmLowpOffsetContributionKernel::set_scale(float scale)
{
    _scale = scale;
}

Status CpuGemmLowpOffsetContributionKernel::validate(const ITensorInfo *mm_result,
                                                     const ITensorInfo *vector_sum_col,
                                                     const ITensorInfo *vector_sum_row,
                                                     int32_t            a_offset,
                                                     int32_t            b_offset)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU,
                            "CpuGemmLowpOffsetContributionKernel::validate");
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(mm_result, vector_sum_col, vector_sum_row, a_offset, b_offset));
    return Status{};
}

void CpuGemmLowpOffsetContributionKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU,
                            "CpuGemmLowpOffsetContributionKernel::run_op");
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto vector_sum_col = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto vector_sum_row = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto mm_result      = tensors.get_tensor(TensorType::ACL_DST);

    // Check if input is a 3D reinterpretation
    const bool reinterpret_as_3d = vector_sum_row != nullptr && mm_result->info()->num_dimensions() > 1 &&
                                   mm_result->info()->tensor_shape().y() != vector_sum_row->info()->tensor_shape().x();

    // check to see what is the output type of result
    auto k_offset = _a_offset * _b_offset * _k;
    _func(window, mm_result, vector_sum_col, vector_sum_row, _a_offset, _b_offset, k_offset, _scale,
          _slide_vector_sum_col, reinterpret_as_3d);
}

const char *CpuGemmLowpOffsetContributionKernel::name() const
{
    return "CpuGemmLowpOffsetContributionKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
