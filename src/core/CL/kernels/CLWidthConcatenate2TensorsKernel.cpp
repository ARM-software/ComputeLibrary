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
#include "arm_compute/core/CL/kernels/CLWidthConcatenate2TensorsKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 8;

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output)
{
    // The window needs to be based on the output
    Window             win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));
    AccessWindowStatic input1_access(input1, 0, 0, ceil_to_multiple(input1->dimension(0), num_elems_processed_per_iteration), input1->dimension(1));
    const unsigned int input2_right_padding = (output->dimension(0) / num_elems_processed_per_iteration) * num_elems_processed_per_iteration - input1->dimension(
                                                  0) + num_elems_processed_per_iteration - input2->dimension(0);
    AccessWindowStatic input2_access(input2, -(input1->dimension(0) % num_elems_processed_per_iteration),
                                     0, input2->dimension(0) + input2_right_padding, input2->dimension(1));
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input1_access, input2_access, output_access);

    Window win_collapsed = win.collapse(win, Window::DimZ);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win_collapsed);
}
Status validate_arguments(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::U16, DataType::S16, DataType::F16, DataType::U32,
                                                         DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1, input2, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input1->dimension(0) + input2->dimension(0) > output->dimension(0));

    for(size_t i = 1; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input1->dimension(i) != output->dimension(i));
        ARM_COMPUTE_RETURN_ERROR_ON(input2->dimension(i) != output->dimension(i));
    }
    ARM_COMPUTE_RETURN_ERROR_ON(input1->num_dimensions() > 4);

    return Status{};
}
} // namespace

CLWidthConcatenate2TensorsKernel::CLWidthConcatenate2TensorsKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

Status CLWidthConcatenate2TensorsKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input1, input2, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input1->clone().get(), input2->clone().get(), output->clone().get()).first);
    return Status{};
}

void CLWidthConcatenate2TensorsKernel::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input1->info(), input2->info(), output->info()));

    _input1 = input1;
    _input2 = input2;
    _output = output;

    // Add build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_underlying_cl_type_from_data_type(input1->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DDEPTH=" + support::cpp11::to_string(input1->info()->dimension(2)));
    build_opts.add_option("-DINPUT1_WIDTH=" + support::cpp11::to_string(input1->info()->dimension(0)));
    build_opts.add_option("-DELEMENT_SIZE=" + support::cpp11::to_string(input1->info()->element_size()));

    if(is_data_type_quantized_asymmetric(input1->info()->data_type()) && input1->info()->quantization_info() != output->info()->quantization_info())
    {
        build_opts.add_option("-DOFFSET_IN1=" + float_to_string_with_full_precision(input1->info()->quantization_info().offset));
        build_opts.add_option("-DOFFSET_OUT=" + float_to_string_with_full_precision(output->info()->quantization_info().offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(input1->info()->quantization_info().scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(output->info()->quantization_info().scale));
        build_opts.add_option("-DOFFSET_IN2=" + float_to_string_with_full_precision(input2->info()->quantization_info().offset));
        build_opts.add_option("-DSCALE_IN2=" + float_to_string_with_full_precision(input2->info()->quantization_info().scale));
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("concatenate_width_x2", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input1->info(), input2->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    ICLKernel::configure_internal(std::get<1>(win_config));

    // Pass paddings as arguments to the kernel
    const unsigned int input1_width         = input1->info()->dimension(0);
    const unsigned int input1_right_padding = ceil_to_multiple(input1_width, num_elems_processed_per_iteration) - input1_width;
    const unsigned int input2_left_padding  = input1_width % num_elems_processed_per_iteration;
    unsigned int       idx0                 = 3 * num_arguments_per_4D_tensor();
    _kernel.setArg<cl_uint>(idx0++, input1_right_padding);
    _kernel.setArg<cl_uint>(idx0++, input2_left_padding);

    // Set config_id for enabling LWS tuning
    _config_id = "concatenate_width_x2_";
    _config_id += lower_string(string_from_data_type(input1->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input1->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input1->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input2->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input2->info()->dimension(1));
}

void CLWidthConcatenate2TensorsKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_4D();

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, _input1, slice);
        add_4D_tensor_argument(idx, _input2, slice);
        add_4D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, window, lws_hint());
    }
    while(window.slide_window_slice_4D(slice));
}
} // namespace arm_compute
