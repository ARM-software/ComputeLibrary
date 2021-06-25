/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/core/gpu/cl/kernels/ClGemmLowpOffsetContributionOutputStageKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias, const ITensorInfo *dst,
                          int32_t a_offset, int32_t b_offset, const GEMMLowpOutputStageInfo &output_stage, const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mm_result, 1, DataType::S32);

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(mm_result->dimension(0) != bias->dimension(0));
    }

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_multipliers, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON(output_multipliers->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_shifts, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON(output_shifts->num_dimensions() > 1);
    if(output_stage.is_quantized_per_channel)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(mm_result->dimension(0) != output_shifts->dimension(0));
        ARM_COMPUTE_RETURN_ERROR_ON(mm_result->dimension(0) != output_multipliers->dimension(0));
    }

    // If a_offset == 0, vector_sum_col can be a nullptr
    if(a_offset != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_col, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(vector_sum_col->dimension(0) != mm_result->dimension(0));
    }

    // If b_offset == 0, vector_sum_row can be a nullptr
    if(b_offset != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_row, 1, DataType::S32);

        // Check if input is a 3D reinterpretation
        const bool reinterpret_as_3d = mm_result->num_dimensions() > 1 && mm_result->tensor_shape().y() != vector_sum_row->tensor_shape().x();

        // Validate input
        ARM_COMPUTE_RETURN_ERROR_ON(reinterpret_as_3d && vector_sum_row->dimension(0) != (mm_result->dimension(1) * mm_result->dimension(2)));
        ARM_COMPUTE_RETURN_ERROR_ON(!reinterpret_as_3d && vector_sum_row->dimension(0) != mm_result->dimension(1));

        TensorShape output_shape = mm_result->tensor_shape();
        if(output_shape.num_dimensions() > 1)
        {
            const unsigned int output_batch_idx = reinterpret_as_3d ? 3 : 2;

            TensorShape vector_sum_row_shape = vector_sum_row->tensor_shape();
            vector_sum_row_shape.collapse_from(1);
            output_shape.collapse_from(output_batch_idx);

            ARM_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_row_shape[1] != output_shape[output_batch_idx],
                                            "mm_result tensor must have the same number of batches of output tensor");

            if(a_offset != 0)
            {
                TensorShape vector_sum_col_shape = vector_sum_col->tensor_shape();
                vector_sum_col_shape.collapse_from(1);

                ARM_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_col_shape[1] != 1 && vector_sum_col_shape[1] != vector_sum_row_shape[1],
                                                "vector_sum_col tensor must have the same number of batches of vector_sum_row_shape or the number of batches must be set to 1");
            }
        }
    }

    ARM_COMPUTE_RETURN_ERROR_ON(output_stage.type == GEMMLowpOutputStageType::NONE);
    // Checks performed when output is configured
    if((dst != nullptr) && (dst->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON(output_stage.output_data_type != dst->data_type());
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mm_result, dst);
    }

    ARM_COMPUTE_RETURN_ERROR_ON(output_stage.gemmlowp_min_bound > output_stage.gemmlowp_max_bound);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output_stage.gemmlowp_multipliers.size() != output_stage.gemmlowp_shifts.size(), "per channel quantization info is incorrect");

    return Status{};
}
} // namespace

ClGemmLowpOffsetContributionOutputStageKernel::ClGemmLowpOffsetContributionOutputStageKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClGemmLowpOffsetContributionOutputStageKernel::configure(const CLCompileContext &compile_context,
                                                              const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias, ITensorInfo *dst,
                                                              int32_t k, int32_t a_offset, int32_t b_offset, const GEMMLowpOutputStageInfo &output_stage,
                                                              const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(mm_result, dst, output_multipliers, output_shifts);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(mm_result, vector_sum_col, vector_sum_row, bias, dst, a_offset, b_offset, output_stage, output_multipliers, output_shifts));

    auto padding_info = get_padding_info({ mm_result, vector_sum_col, vector_sum_row, bias, dst, output_multipliers, output_shifts });

    const int min = output_stage.gemmlowp_min_bound;
    const int max = output_stage.gemmlowp_max_bound;

    _is_quantized_per_channel = output_stage.is_quantized_per_channel;

    // Check if input is a 3D reinterpretation
    const bool reinterpret_as_3d = vector_sum_row != nullptr
                                   && mm_result->num_dimensions() > 1
                                   && mm_result->tensor_shape().y() != vector_sum_row->tensor_shape().x();

    // Auto initialize the output
    auto_init_if_empty(*dst, mm_result->clone()->set_data_type(output_stage.output_data_type));

    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(4, mm_result->dimension(0));

    // Set the arguments to pass at compile time
    CLBuildOptions build_opts;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(mm_result->dimension(0) % num_elems_processed_per_iteration));

    // If a_offset == 0, vector_sum_col can be a nullptr
    if(a_offset != 0)
    {
        build_opts.add_option("-DA_OFFSET=" + support::cpp11::to_string(a_offset));
        build_opts.add_option_if(vector_sum_col->tensor_shape().num_dimensions() > 1, "-DSUM_COL_HAS_BATCHES");
    }
    // If b_offset == 0, vector_sum_row can be a nullptr
    build_opts.add_option_if(b_offset != 0, "-DB_OFFSET=" + support::cpp11::to_string(b_offset));
    build_opts.add_option("-DK_OFFSET=" + support::cpp11::to_string(a_offset * b_offset * k));
    build_opts.add_option_if(reinterpret_as_3d, "-DHEIGHT_INPUT3D=" + support::cpp11::to_string(mm_result->dimension(1)));
    build_opts.add_option_if(reinterpret_as_3d, "-DDEPTH_INPUT3D=" + support::cpp11::to_string(mm_result->dimension(2)));
    build_opts.add_option_if(bias != nullptr, "-DADD_BIAS");
    build_opts.add_option("-DRESULT_OFFSET=" + support::cpp11::to_string(output_stage.gemmlowp_offset));
    build_opts.add_option("-DRESULT_MULTIPLIER=" + support::cpp11::to_string(output_stage.gemmlowp_multipliers[0]));
    build_opts.add_option("-DRESULT_SHIFT=" + support::cpp11::to_string(output_stage.gemmlowp_shifts[0]));
    build_opts.add_option_if(_is_quantized_per_channel, "-DPER_CHANNEL_QUANTIZATION");
    build_opts.add_option("-DOUTPUT_DATA_TYPE=" + get_cl_type_from_data_type(dst->data_type()));

    PixelValue min_val{};
    PixelValue max_val{};
    std::tie(min_val, max_val) = get_min_max(dst->data_type());
    build_opts.add_option_if((min > min_val.get<int32_t>()), "-DMIN_BOUND=" + support::cpp11::to_string(min));
    build_opts.add_option_if((max < max_val.get<int32_t>()), "-DMAX_BOUND=" + support::cpp11::to_string(max));

    std::string kernel_name("gemmlowp_offset_contribution");
    kernel_name += "_" + string_from_gemmlowp_output_stage(output_stage.type);

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*mm_result, Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name + "_";
    _config_id += support::cpp11::to_string(mm_result->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(mm_result->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(mm_result->dimension(2));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClGemmLowpOffsetContributionOutputStageKernel::validate(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias,
                                                               const ITensorInfo *dst, int32_t a_offset, int32_t b_offset, const GEMMLowpOutputStageInfo &output_stage,
                                                               const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(mm_result, vector_sum_col, vector_sum_row, bias, dst, a_offset, b_offset, output_stage, output_multipliers, output_shifts));
    return Status{};
}

void ClGemmLowpOffsetContributionOutputStageKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto mm_result          = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    const auto bias               = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_BIAS));
    const auto vector_sum_col     = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_VEC_COL_SUM));
    const auto vector_sum_row     = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_VEC_ROW_SUM));
    const auto output_shifts      = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SHIFTS));
    const auto output_multipliers = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_MULTIPLIERS));
    auto       dst                = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    // Set window for vector_sum_col
    Window win_vector_sum_col = slice;
    win_vector_sum_col.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_vector_sum_col.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Set window for vector_sum_row
    Window win_vector_sum_row = slice;
    win_vector_sum_row.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_vector_sum_row.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_vector_sum_col.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Window biases_slice = slice;
    biases_slice.set(Window::DimY, Window::Dimension(0, 1, 1));
    biases_slice.set(Window::DimZ, Window::Dimension(0, 1, 1));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, mm_result, slice);
        add_2D_tensor_argument_if((vector_sum_col != nullptr), idx, vector_sum_col, win_vector_sum_col);
        add_2D_tensor_argument_if((vector_sum_row != nullptr), idx, vector_sum_row, win_vector_sum_row);
        add_1D_tensor_argument_if((bias != nullptr), idx, bias, biases_slice);
        add_3D_tensor_argument(idx, dst, slice);
        add_1D_tensor_argument_if(_is_quantized_per_channel, idx, output_multipliers, biases_slice);
        add_1D_tensor_argument_if(_is_quantized_per_channel, idx, output_shifts, biases_slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
