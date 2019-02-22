/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLGEMMLowpOffsetContributionOutputStageKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/ToolchainSupport.h"

#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
Status validate_arguments(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias, const ITensorInfo *output,
                          int32_t a_offset, int32_t b_offset, const GEMMLowpOutputStageInfo &output_stage)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mm_result, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON(output_stage.type == GEMMLowpOutputStageType::NONE);
    ARM_COMPUTE_RETURN_ERROR_ON(output_stage.gemmlowp_max_bound > 255);
    ARM_COMPUTE_RETURN_ERROR_ON(output_stage.gemmlowp_min_bound < 0 || output_stage.gemmlowp_min_bound > output_stage.gemmlowp_max_bound);

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

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mm_result, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *mm_result, ITensorInfo *vector_sum_col, ITensorInfo *vector_sum_row, ITensorInfo *bias, ITensorInfo *output,
                                                        int32_t a_offset, int32_t b_offset)
{
    constexpr unsigned int num_elems_processed_per_iteration = 4;
    bool                   window_changed                    = false;

    // Auto initialize the output
    auto_init_if_empty(*output, mm_result->clone()->set_data_type(DataType::QASYMM8));

    // Configure kernel window
    Window win = calculate_max_window(*mm_result, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal mm_result_access(mm_result, 0, num_elems_processed_per_iteration);
    window_changed = window_changed || update_window_and_padding(win, mm_result_access);

    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    window_changed = window_changed || update_window_and_padding(win, output_access);

    if(a_offset != 0)
    {
        AccessWindowHorizontal vector_sum_col_access(vector_sum_col, 0, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, vector_sum_col_access);
    }
    if(b_offset != 0)
    {
        AccessWindowStatic vector_sum_row_access(vector_sum_row, 0, 0, vector_sum_row->dimension(0), 0); // NOLINT
        window_changed = window_changed || update_window_and_padding(win, vector_sum_row_access);
    }

    if(bias != nullptr)
    {
        AccessWindowStatic bias_access(bias, 0, 0, ceil_to_multiple(bias->dimension(0), num_elems_processed_per_iteration), bias->tensor_shape()[1]);
        window_changed = window_changed || update_window_and_padding(win, bias_access);
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLGEMMLowpOffsetContributionOutputStageKernel::CLGEMMLowpOffsetContributionOutputStageKernel()
    : _mm_result(nullptr), _vector_sum_col(nullptr), _vector_sum_row(nullptr), _bias(nullptr), _output(nullptr)
{
}

void CLGEMMLowpOffsetContributionOutputStageKernel::configure(const ICLTensor *mm_result, const ICLTensor *vector_sum_col, const ICLTensor *vector_sum_row, const ICLTensor *bias, ICLTensor *output,
                                                              int32_t k, int32_t a_offset, int32_t b_offset, const GEMMLowpOutputStageInfo &output_stage)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(mm_result, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(mm_result->info(),
                                                  vector_sum_col != nullptr ? vector_sum_col->info() : nullptr,
                                                  vector_sum_row != nullptr ? vector_sum_row->info() : nullptr,
                                                  bias != nullptr ? bias->info() : nullptr,
                                                  output->info(),
                                                  a_offset, b_offset, output_stage)); // NOLINT

    const int min = output_stage.gemmlowp_min_bound;
    const int max = output_stage.gemmlowp_max_bound;

    _vector_sum_col = vector_sum_col;
    _vector_sum_row = vector_sum_row;
    _mm_result      = mm_result;
    _bias           = bias;
    _output         = output;

    // Check if input is a 3D reinterpretation
    const bool reinterpret_as_3d = vector_sum_row != nullptr
                                   && mm_result->info()->num_dimensions() > 1
                                   && mm_result->info()->tensor_shape().y() != vector_sum_row->info()->tensor_shape().x();

    // Set the arguments to pass at compile time
    CLBuildOptions build_opts;

    // If a_offset == 0, vector_sum_col can be a nullptr
    if(a_offset != 0)
    {
        build_opts.add_option("-DA_OFFSET=" + support::cpp11::to_string(a_offset));
        build_opts.add_option_if(vector_sum_col->info()->tensor_shape().num_dimensions() > 1, "-DSUM_COL_HAS_BATCHES");
    }
    // If b_offset == 0, vector_sum_row can be a nullptr
    build_opts.add_option_if(b_offset != 0, "-DB_OFFSET=" + support::cpp11::to_string(b_offset));
    build_opts.add_option("-DK_OFFSET=" + support::cpp11::to_string(a_offset * b_offset * k));
    build_opts.add_option_if(reinterpret_as_3d, "-DHEIGHT_INPUT3D=" + support::cpp11::to_string(mm_result->info()->dimension(1)));
    build_opts.add_option_if(reinterpret_as_3d, "-DDEPTH_INPUT3D=" + support::cpp11::to_string(mm_result->info()->dimension(2)));
    build_opts.add_option_if(bias != nullptr, "-DADD_BIAS");
    build_opts.add_option("-DRESULT_OFFSET=" + support::cpp11::to_string(output_stage.gemmlowp_offset));
    build_opts.add_option("-DRESULT_MULTIPLIER=" + support::cpp11::to_string(output_stage.gemmlowp_multiplier));
    build_opts.add_option("-DRESULT_SHIFT=" + support::cpp11::to_string(output_stage.gemmlowp_shift));
    build_opts.add_option_if((min != 0) && (min != max), "-DMIN_BOUND=" + support::cpp11::to_string(min));
    build_opts.add_option_if((max != 255) && (min != max), "-DMAX_BOUND=" + support::cpp11::to_string(max));

    std::string kernel_name("gemmlowp_offset_contribution");

    // Fuse output stage
    if(output_stage.type != GEMMLowpOutputStageType::NONE)
    {
        kernel_name += "_" + string_from_gemmlowp_output_stage(output_stage.type);
    }
    else
    {
        ARM_COMPUTE_ERROR("GEMMLowpOutputStage can not be NONE!");
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(mm_result->info(),
                                                    vector_sum_col != nullptr ? vector_sum_col->info() : nullptr,
                                                    vector_sum_row != nullptr ? vector_sum_row->info() : nullptr,
                                                    bias != nullptr ? bias->info() : nullptr,
                                                    output->info(),
                                                    a_offset, b_offset); // NOLINT
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name + "_";
    _config_id += support::cpp11::to_string(mm_result->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(mm_result->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(mm_result->info()->dimension(2));
}

Status CLGEMMLowpOffsetContributionOutputStageKernel::validate(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias,
                                                               const ITensorInfo *output,
                                                               int32_t a_offset, int32_t b_offset, const GEMMLowpOutputStageInfo &output_stage)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(mm_result, vector_sum_col, vector_sum_row, bias, output, a_offset, b_offset, output_stage));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(mm_result->clone().get(),
                                                              vector_sum_col != nullptr ? vector_sum_col->clone().get() : nullptr,
                                                              vector_sum_row != nullptr ? vector_sum_row->clone().get() : nullptr,
                                                              bias != nullptr ? bias->clone().get() : nullptr,
                                                              output->clone().get(),
                                                              a_offset, b_offset)
                                .first); // NOLINT

    return Status{};
}

void CLGEMMLowpOffsetContributionOutputStageKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

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
        add_3D_tensor_argument(idx, _mm_result, slice);
        if(_vector_sum_col != nullptr)
        {
            add_2D_tensor_argument(idx, _vector_sum_col, win_vector_sum_col);
        }
        if(_vector_sum_row != nullptr)
        {
            add_2D_tensor_argument(idx, _vector_sum_row, win_vector_sum_row);
        }
        if(_bias != nullptr)
        {
            add_1D_tensor_argument(idx, _bias, biases_slice);
        }
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
