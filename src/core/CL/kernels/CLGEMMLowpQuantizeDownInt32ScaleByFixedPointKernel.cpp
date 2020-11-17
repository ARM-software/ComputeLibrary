/*
 * Copyright (c) 2020 Arm Limited.
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
#include "src/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const GEMMLowpOutputStageInfo *info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S32);

    // Check biases if exist
    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != bias->dimension(0));
    }

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->data_type() != info->output_data_type, "Mismatching output data type");
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}
} // namespace

CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel::CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel()
    : _input(nullptr), _bias(nullptr), _output(nullptr)
{
}

Status CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                                                                    const GEMMLowpOutputStageInfo *info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, bias, output, info));

    return Status{};
}

void CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                                   const GEMMLowpOutputStageInfo *info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (bias != nullptr) ? bias->info() : nullptr, output->info(), info));

    auto padding_info = get_padding_info({ input, bias, output });

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_data_type(info->output_data_type));

    _input  = input;
    _bias   = bias;
    _output = output;

    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(4, input->info()->dimension(0));

    // Set the arguments to pass at compile time
    auto           min = info->gemmlowp_min_bound;
    auto           max = info->gemmlowp_max_bound;
    CLBuildOptions build_opts;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(input->info()->dimension(0) % num_elems_processed_per_iteration));
    build_opts.add_option("-DRESULT_OFFSET_AFTER_SHIFT=" + support::cpp11::to_string(info->gemmlowp_offset));
    build_opts.add_option("-DRESULT_FIXEDPOINT_MULTIPLIER=" + support::cpp11::to_string(info->gemmlowp_multiplier));
    build_opts.add_option("-DRESULT_SHIFT=" + support::cpp11::to_string(info->gemmlowp_shift));
    build_opts.add_option("-DOUTPUT_DATA_TYPE=" + get_cl_type_from_data_type(output->info()->data_type()));
    build_opts.add_option_if((min > std::get<0>(quantization::get_min_max_values_from_quantized_data_type(info->output_data_type))) && (min != max),
                             "-DMIN_BOUND=" + support::cpp11::to_string(min));
    build_opts.add_option_if((max < std::get<1>(quantization::get_min_max_values_from_quantized_data_type(info->output_data_type))) && (min != max),
                             "-DMAX_BOUND=" + support::cpp11::to_string(max));
    build_opts.add_option_if(bias != nullptr, "-DADD_BIAS");

    // Create kernel
    const std::string kernel_name = (info->output_data_type == DataType::QSYMM16) ? "gemmlowp_output_stage_quantize_down_fixedpoint_qsymm16" : "gemmlowp_output_stage_quantize_down_fixedpoint";
    _kernel                       = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    auto win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

void CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    // Create input window
    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    // Setup bias slice
    unsigned int idx1 = num_arguments_per_3D_tensor();
    if(_bias != nullptr)
    {
        Window biases_slice(slice);
        biases_slice.set(Window::DimY, Window::Dimension(0, 1, 1));
        biases_slice.set(Window::DimZ, Window::Dimension(0, 1, 1));
        add_1D_tensor_argument(idx1, _bias, biases_slice);
    }

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx1, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
} // namespace arm_compute
