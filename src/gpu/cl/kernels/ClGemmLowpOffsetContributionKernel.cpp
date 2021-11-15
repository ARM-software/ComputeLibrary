/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/ClGemmLowpOffsetContributionKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

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
Status validate_arguments(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias,
                          int32_t a_offset, int32_t b_offset)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mm_result, 1, DataType::S32);

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(mm_result->dimension(0) != bias->dimension(0));
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

    return Status{};
}
} // namespace

ClGemmLowpOffsetContributionKernel::ClGemmLowpOffsetContributionKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClGemmLowpOffsetContributionKernel::configure(const CLCompileContext &compile_context,
                                                   const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias,
                                                   int32_t k, int32_t a_offset, int32_t b_offset)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(mm_result);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(mm_result, vector_sum_col, vector_sum_row, bias, a_offset, b_offset));

    auto padding_info = get_padding_info({ mm_result, vector_sum_col, vector_sum_row, bias });

    // Check if input is a 3D reinterpretation
    const bool reinterpret_as_3d = vector_sum_row != nullptr
                                   && mm_result->num_dimensions() > 1
                                   && mm_result->tensor_shape().y() != vector_sum_row->tensor_shape().x();

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

    std::string kernel_name("gemmlowp_offset_contribution");

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*mm_result, Steps(num_elems_processed_per_iteration));
    IClKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name + "_";
    _config_id += support::cpp11::to_string(mm_result->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(mm_result->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(mm_result->dimension(2));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClGemmLowpOffsetContributionKernel::validate(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias,
                                                    int32_t a_offset, int32_t b_offset)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(mm_result, vector_sum_col, vector_sum_row, bias, a_offset, b_offset));
    return Status{};
}

void ClGemmLowpOffsetContributionKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IClKernel::window(), window);

    const auto vector_sum_col = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_VEC_COL_SUM));
    const auto vector_sum_row = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_VEC_ROW_SUM));
    const auto bias           = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_BIAS));
    const auto mm_result      = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_SRC_DST));

    Window collapsed = window.collapse_if_possible(IClKernel::window(), Window::DimZ);
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

        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
