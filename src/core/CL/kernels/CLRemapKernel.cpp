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
#include "src/core/CL/kernels/CLRemapKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
CLRemapKernel::CLRemapKernel()
    : _input(nullptr), _output(nullptr), _map_x(nullptr), _map_y(nullptr), _data_layout(DataLayout::NCHW)
{
    _type = CLKernelType::ELEMENTWISE;
}

BorderSize CLRemapKernel::border_size() const
{
    return _data_layout == DataLayout::NCHW ? BorderSize(1) : BorderSize(0);
}

template <class T>
void CLRemapKernel::set_constant_border(unsigned int idx, const PixelValue &constant_border_value)
{
    T value;
    constant_border_value.get(value);
    ICLKernel::add_argument<T>(idx, static_cast<T>(value));
}

Status CLRemapKernel::validate(const ITensorInfo *input, const ITensorInfo *map_x, const ITensorInfo *map_y, const ITensorInfo *output, RemapInfo info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, map_x, map_y, output);
    if(input->data_layout() == DataLayout::NCHW)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::F16);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() != output->data_type(), "Input/output have different data types");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(map_x, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(map_y, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.policy == InterpolationPolicy::AREA, "Area interpolation is not supported!");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.border_mode != BorderMode::CONSTANT && info.border_mode != BorderMode::UNDEFINED, "Border mode not supported");
    return Status{};
}

void CLRemapKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output, RemapInfo info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, map_x, map_y, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLRemapKernel::validate(input->info(), map_x->info(), map_y->info(), output->info(), info));

    _input       = input;
    _output      = output;
    _map_x       = map_x;
    _map_y       = map_y;
    _data_layout = input->info()->data_layout();

    const bool is_nhwc            = _data_layout == DataLayout::NHWC;
    const bool is_constant_border = info.border_mode == BorderMode::CONSTANT;

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.add_option_if(is_nhwc, "-DDEPTH_OUT=" + support::cpp11::to_string(output->info()->dimension(2)));
    build_opts.add_option_if(is_constant_border, "-DCONSTANT_BORDER");

    const std::string interpolation_name = lower_string(string_from_interpolation_policy(info.policy));
    const std::string kernel_name        = "remap_" + interpolation_name + "_" + lower_string(string_from_data_layout(_data_layout));
    _kernel                              = create_kernel(compile_context, kernel_name, build_opts.options());

    const unsigned int num_elems_processed_per_iteration = is_nhwc ? 1 : 4;
    const int          idx_height                        = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const int          idx_width                         = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int          input_height                      = input->info()->dimension(idx_height);
    const int          input_width                       = input->info()->dimension(idx_width);

    // Configure window
    Window win = calculate_max_window(*_output->info(), Steps(num_elems_processed_per_iteration));

    // Update padding in NCHW case
    if(_data_layout == DataLayout::NCHW)
    {
        const int          total_right  = ceil_to_multiple(input_width, num_elems_processed_per_iteration);
        const int          access_right = total_right + (((total_right - input_width) == 0) ? border_size().right : 0);
        AccessWindowStatic input_access(input->info(), -border_size().left, -border_size().top, access_right, input_height + border_size().bottom);

        AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

        update_window_and_padding(win, input_access, output_access);
    }

    ICLKernel::configure_internal(win);

    // Set static arguments
    unsigned int idx = 4 * (is_nhwc ? num_arguments_per_4D_tensor() : num_arguments_per_2D_tensor());
    _kernel.setArg<cl_float>(idx++, input_width);
    _kernel.setArg<cl_float>(idx++, input_height);
    if(is_nhwc && is_constant_border)
    {
        switch(input->info()->data_type())
        {
            case DataType::U8:
                set_constant_border<uint8_t>(idx, info.constant_border_value);
                break;
            case DataType::F16:
                static_assert(sizeof(cl_half) == sizeof(half), "Half must be same size as cl_half");
                static_assert(sizeof(cl_half) == 2, "Half must be 16 bit");
                set_constant_border<half>(idx, info.constant_border_value);
                break;
            default:
                ARM_COMPUTE_ERROR("Data Type not handled");
        }
    }
}

void CLRemapKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);
    switch(_data_layout)
    {
        case DataLayout::NCHW:
        {
            Window slice = window.first_slice_window_2D();
            do
            {
                unsigned int idx = 0;
                add_2D_tensor_argument(idx, _input, slice);
                add_2D_tensor_argument(idx, _output, slice);
                add_2D_tensor_argument(idx, _map_x, slice);
                add_2D_tensor_argument(idx, _map_y, slice);
                enqueue(queue, *this, slice, lws_hint());

            }
            while(window.slide_window_slice_2D(slice));
            break;
        }
        case DataLayout::NHWC:
        {
            Window collapsed = window.collapse(ICLKernel::window(), Window::DimZ);
            Window slice     = collapsed.first_slice_window_4D();

            unsigned int idx = 0;
            add_4D_tensor_argument(idx, _input, slice);
            add_4D_tensor_argument(idx, _output, slice);
            add_4D_tensor_argument(idx, _map_x, slice);
            add_4D_tensor_argument(idx, _map_y, slice);
            enqueue(queue, *this, slice, lws_hint());
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Invalid Data layout");
    }
}
} // namespace arm_compute
