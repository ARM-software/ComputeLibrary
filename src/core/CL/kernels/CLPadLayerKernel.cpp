/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLPadLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    ARM_COMPUTE_UNUSED(input, output, constant_value);
    ARM_COMPUTE_RETURN_ERROR_ON(padding.empty());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(mode != PaddingMode::CONSTANT, "Only CONSTANT mode supported.");

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    ARM_COMPUTE_UNUSED(constant_value, mode);
    // Output auto initialization if not yet initialized
    const TensorShape expected_output_shape = arm_compute::misc::shape_calculator::compute_padded_shape(input->tensor_shape(), padding);
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(expected_output_shape));

    const unsigned int num_elems_processed_per_iteration = std::min(16U, 32U / static_cast<unsigned int>(element_size_from_data_type(input->data_type())));

    // Configure kernel window
    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    const int input_start_x = -(padding.at(0).first % num_elems_processed_per_iteration);
    const int input_start_y = padding.size() > 1 ? -padding.at(1).first : 0;

    AccessWindowRectangle  input_access(input, input_start_x, input_start_y, num_elems_processed_per_iteration, 1);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    const bool window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLPadLayerKernel::CLPadLayerKernel()
    : _input(nullptr), _output(nullptr), _input_start_x(0), _input_start_y(0)
{
}

void CLPadLayerKernel::configure(const ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), padding, constant_value, mode));

    _input  = input;
    _output = output;

    // Set build options
    const unsigned int num_elems_processed_per_iteration = std::min(16U, 32U / static_cast<unsigned int>(element_size_from_data_type(input->info()->data_type())));
    _input_start_x                                       = -(padding.at(0).first % num_elems_processed_per_iteration);
    _input_start_y                                       = padding.size() > 1 ? -padding.at(1).first : 0;

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DCONST_VAL=" + string_from_pixel_value(constant_value, input->info()->data_type()));
    build_opts.add_option("-DPAD_LEFT=" + support::cpp11::to_string(padding.at(0).first));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));
    build_opts.add_option("-DSELECT_DT=" + get_cl_select_type_from_data_type(input->info()->data_type()));
    build_opts.add_option_if(padding.at(0).first > num_elems_processed_per_iteration, "-DTHREADS_TO_SKIP_X=" + support::cpp11::to_string(padding.at(0).first / num_elems_processed_per_iteration));

    if(padding.size() > 1)
    {
        build_opts.add_option("-DPAD_TOP=" + support::cpp11::to_string(padding.at(1).first));
        build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(1)));

        if(padding.size() > 2)
        {
            build_opts.add_option("-DPAD_NEAR=" + support::cpp11::to_string(padding.at(2).first));
            build_opts.add_option("-DSRC_DEPTH=" + support::cpp11::to_string(input->info()->dimension(2)));

            if(padding.size() > 3)
            {
                build_opts.add_option("-DPAD_BTOP=" + support::cpp11::to_string(padding.at(3).first));
                build_opts.add_option("-DSRC_BATCH=" + support::cpp11::to_string(input->info()->dimension(3)));
            }
        }
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("pad_layer", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), padding, constant_value, mode);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);
}

Status CLPadLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, padding, constant_value, mode));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), padding, constant_value, mode).first);

    return Status{};
}

void CLPadLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window win_in = window;
    win_in.adjust(Window::DimX, _input_start_x, true);
    win_in.adjust(Window::DimY, _input_start_y, true);

    Window       slice_out = window.first_slice_window_3D();
    Window       slice_in  = win_in.first_slice_window_3D();
    unsigned int batch     = 0;
    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_3D_tensor_argument(idx, _output, slice_out);
        add_argument<unsigned int>(idx, batch++);

        enqueue(queue, *this, slice_out, lws_hint());
    }
    while(window.slide_window_slice_3D(slice_out) && win_in.slide_window_slice_3D(slice_in));
}
} // namespace arm_compute
