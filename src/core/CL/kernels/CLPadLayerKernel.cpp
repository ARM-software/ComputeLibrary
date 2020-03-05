/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_UNUSED(constant_value);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(padding.size() > input->num_dimensions());
    if(mode == PaddingMode::REFLECT || mode == PaddingMode::SYMMETRIC)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(padding.size() > 3);

        const auto is_reflect = static_cast<unsigned int>(mode == PaddingMode::REFLECT);
        for(size_t i = 0; i < padding.size(); ++i)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(padding.at(i).first > (input->dimension(i) - is_reflect));
            ARM_COMPUTE_RETURN_ERROR_ON(padding.at(i).second > (input->dimension(i) - is_reflect));
        }
    }

    if(output->total_size() > 0)
    {
        TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(input->tensor_shape(), padding);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(output, input);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), padded_shape);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode,
                                                        unsigned int &num_elems_processed_per_iteration)
{
    ARM_COMPUTE_UNUSED(constant_value, mode);

    const TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(input->tensor_shape(), padding);
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(padded_shape));

    num_elems_processed_per_iteration = std::min(16U, 32U / static_cast<unsigned int>(element_size_from_data_type(input->data_type())));
    if(input->dimension(0) < num_elems_processed_per_iteration)
    {
        num_elems_processed_per_iteration = 1 << static_cast<unsigned int>(std::log2(input->dimension(0)));
    }

    // Configure kernel window
    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    const int input_start_x = mode == PaddingMode::CONSTANT ? -(padding.at(0).first % num_elems_processed_per_iteration) : 0;
    const int input_start_y = (mode == PaddingMode::CONSTANT && padding.size() > 1) ? -padding.at(1).first : 0;

    AccessWindowRectangle  input_access(input, input_start_x, input_start_y, num_elems_processed_per_iteration, 1);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    const bool window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLPadLayerKernel::CLPadLayerKernel()
    : _input(nullptr), _output(nullptr), _input_start_x(0), _input_start_y(0), _4d_enabled(false)
{
}

void CLPadLayerKernel::configure(const ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), padding, constant_value, mode));

    _input      = input;
    _output     = output;
    _4d_enabled = (mode == PaddingMode::CONSTANT) && (padding.size() > 3);

    // Configure window
    unsigned int vec_size;
    auto         win_config = validate_and_configure_window(input->info(), output->info(), padding, constant_value, mode, vec_size);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set build options
    std::string kernel_name = "pad_layer_";

    const DataType    &data_type       = input->info()->data_type();
    const unsigned int input_width     = input->info()->dimension(0);
    const unsigned int input_height    = input->info()->dimension(1);
    const unsigned int input_depth     = input->info()->dimension(2);
    const unsigned int pad_x_before    = padding.at(0).first;
    const unsigned int pad_y_before    = padding.size() > 1 ? padding.at(1).first : 0;
    const unsigned int pad_z_before    = padding.size() > 2 ? padding.at(2).first : 0;
    const unsigned int pad_right_start = input_width + pad_x_before;

    _input_start_x = mode == PaddingMode::CONSTANT ? -(pad_x_before % vec_size) : 0;
    _input_start_y = (mode == PaddingMode::CONSTANT && padding.size() > 1) ? -padding.at(1).first : 0;

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DSELECT_DT=" + get_cl_select_type_from_data_type(data_type));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size));
    build_opts.add_option("-DPAD_X_BEFORE=" + support::cpp11::to_string(pad_x_before));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input_width));
    if(padding.size() > 1)
    {
        build_opts.add_option("-DPAD_Y_BEFORE=" + support::cpp11::to_string(pad_y_before));
        build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input_height));

        if(padding.size() > 2)
        {
            build_opts.add_option("-DPAD_Z_BEFORE=" + support::cpp11::to_string(pad_z_before));
            build_opts.add_option("-DSRC_DEPTH=" + support::cpp11::to_string(input_depth));
        }
    }

    switch(mode)
    {
        case PaddingMode::CONSTANT:
        {
            kernel_name += "constant";

            build_opts.add_option("-DCONST_VAL=" + string_from_pixel_value(constant_value, data_type));
            build_opts.add_option_if(pad_x_before >= vec_size, "-DNUM_THREADS_TO_SKIP_X=" + support::cpp11::to_string(pad_x_before / vec_size));

            if(_4d_enabled)
            {
                build_opts.add_option("-DPAD_W_BEFORE=" + support::cpp11::to_string(padding.at(3).first));
                build_opts.add_option("-DSRC_BATCH=" + support::cpp11::to_string(input->info()->dimension(3)));
            }

            break;
        }
        case PaddingMode::SYMMETRIC:
        case PaddingMode::REFLECT:
        {
            kernel_name += "symmetric_reflect";

            const auto is_reflect = static_cast<unsigned int>(mode == PaddingMode::REFLECT);

            const unsigned int pad_x_before_remainder = pad_x_before % vec_size;
            const unsigned int pad_x_after_remainder  = pad_right_start % vec_size;
            const unsigned int after_pad_fact_x       = (2 * input_width + pad_x_before) - is_reflect;
            const unsigned int output_last_x          = ceil_to_multiple(pad_right_start + padding.at(0).second, vec_size);

            build_opts.add_option("-DIS_REFLECT=" + support::cpp11::to_string(is_reflect));
            build_opts.add_option("-DPAD_X_BEFORE_REMAINDER=" + support::cpp11::to_string(pad_x_before_remainder));
            build_opts.add_option("-DPAD_X_AFTER_REMAINDER=" + support::cpp11::to_string(pad_x_after_remainder));
            build_opts.add_option("-DPAD_X_BEFORE_REMAINDER_REFL=" + support::cpp11::to_string((pad_x_before_remainder + is_reflect) % vec_size));
            build_opts.add_option("-DPAD_X_AFTER_REMAINDER_REFL=" + support::cpp11::to_string((pad_x_after_remainder - is_reflect) % vec_size));
            build_opts.add_option("-DAFTER_PAD_FACT_X=" + support::cpp11::to_string(after_pad_fact_x));
            build_opts.add_option_if(after_pad_fact_x < output_last_x, "-DAFTER_PAD_REM=" + support::cpp11::to_string(after_pad_fact_x % vec_size));

            break;
        }
        default:
            ARM_COMPUTE_ERROR("Padding mode not supported.");
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));
}

Status CLPadLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    unsigned int vec_size;
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, padding, constant_value, mode));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), padding, constant_value, mode, vec_size).first);

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
        if(_4d_enabled)
        {
            add_argument<unsigned int>(idx, batch++);
        }

        enqueue(queue, *this, slice_out, lws_hint());
    }
    while(window.slide_window_slice_3D(slice_out) && win_in.slide_window_slice_3D(slice_in));
}
} // namespace arm_compute
