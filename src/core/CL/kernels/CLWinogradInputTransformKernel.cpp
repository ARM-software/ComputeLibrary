/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLWinogradInputTransformKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() != DataLayout::NCHW);

    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const Size2D        kernel_size      = winograd_info.kernel_size;
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.stride().first != 1 || conv_info.stride().second != 1, "Winograd input transform only supports unit strides");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(kernel_size != Size2D(3U, 3U), "Winograd input transform only supports 3x3 kernels");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output_tile_size != Size2D(2U, 2U), "Winograd input transform only supports 2x2 output tile size");
    ARM_COMPUTE_UNUSED(conv_info);
    ARM_COMPUTE_UNUSED(output_tile_size);
    ARM_COMPUTE_UNUSED(kernel_size);

    // Validate configured output
    if(output->total_size() != 0)
    {
        const TensorShape output_shape = misc::shape_calculator::compute_winograd_input_transform_shape(*input, winograd_info);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const Size2D        kernel_size      = winograd_info.kernel_size;

    const unsigned int num_elems_read_per_iteration_x = output_tile_size.width + kernel_size.width - 1;
    const unsigned int num_elems_read_per_iteration_y = output_tile_size.height + kernel_size.height - 1;

    Window win = calculate_max_window(*input, Steps(1, 1));

    AccessWindowRectangle input_access(input, -conv_info.pad_left(), -conv_info.pad_top(), num_elems_read_per_iteration_x, num_elems_read_per_iteration_y);

    bool window_changed = update_window_and_padding(win, input_access);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLWinogradInputTransformKernel::CLWinogradInputTransformKernel()
    : _border_size(0), _input(nullptr), _output(nullptr), _num_tiles_x(0), _num_tiles_y(0), _step_z(1)
{
}

BorderSize CLWinogradInputTransformKernel::border_size() const
{
    return _border_size;
}

void CLWinogradInputTransformKernel::configure(const ICLTensor *input, ICLTensor *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), winograd_info));

    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const Size2D        kernel_size      = winograd_info.kernel_size;

    // Compute number of elements to process in the X and Y direction
    const int num_elements_x = input->info()->dimension(0) - (kernel_size.width - 1) + conv_info.pad_left() + conv_info.pad_right();
    const int num_elements_y = input->info()->dimension(1) - (kernel_size.height - 1) + conv_info.pad_top() + conv_info.pad_bottom();

    // Check if we need to extend the right or bottom border
    // FIXME: This actually is not needed. Added just for validating the result;
    const unsigned int extra_border_right  = ((num_elements_x % output_tile_size.width) == 0) ? 0u : static_cast<unsigned int>(output_tile_size.width - 1);
    const unsigned int extra_border_bottom = ((num_elements_y % output_tile_size.height) == 0) ? 0u : static_cast<unsigned int>(output_tile_size.height - 1);

    _input       = input;
    _output      = output;
    _border_size = BorderSize(conv_info.pad_top(), conv_info.pad_right() + extra_border_right, conv_info.pad_bottom() + extra_border_bottom, conv_info.pad_left());
    _num_tiles_x = std::ceil(num_elements_x / static_cast<float>(output_tile_size.width));
    _num_tiles_y = std::ceil(num_elements_y / static_cast<float>(output_tile_size.height));

    const TensorShape output_shape = misc::shape_calculator::compute_winograd_input_transform_shape(*input->info(), winograd_info);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    ARM_COMPUTE_ERROR_ON(_num_tiles_x * _num_tiles_y != static_cast<int>(output->info()->dimension(1)));

    CLBuildOptions build_opts;
    build_opts.add_option("-DNUM_TILES_X=" + support::cpp11::to_string(_num_tiles_x));
    build_opts.add_option("-DPAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));
    build_opts.add_option("-DPAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));

    // Create kernel
    std::string kernel_name = "winograd_input_transform_" + output_tile_size.to_string() + "_" + kernel_size.to_string();

    // Check optimized kernel if output_dims == 2x2
    if(output_tile_size.width == 2 && output_tile_size.height == 2)
    {
        if((_input->info()->dimension(2) % 2) != 0)
        {
            _step_z = 1;
        }
        else
        {
            _step_z   = 2;
            _lws_hint = cl::NDRange(1, 1, 8);
        }
    }

    // Append stepz and data layout
    kernel_name += "_stepz";
    kernel_name += support::cpp11::to_string(_step_z);
    kernel_name += "_nchw";

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Create window and update padding
    auto win_config = validate_and_configure_window(input->info(), output->info(), winograd_info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);

    _config_id = kernel_name;
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(conv_info.pad_left());
    _config_id += "_";
    _config_id += support::cpp11::to_string(conv_info.pad_top());
}

Status CLWinogradInputTransformKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, winograd_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), winograd_info).first);

    return Status{};
}

void CLWinogradInputTransformKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice = window.first_slice_window_3D();
    slice.set(Window::DimX, Window::Dimension(0, _num_tiles_x, 1));
    slice.set(Window::DimY, Window::Dimension(0, _num_tiles_y, 1));

    ARM_COMPUTE_ERROR_ON(((slice.z().end() - slice.z().start()) % _step_z) != 0);
    slice.set(Window::DimZ, Window::Dimension(slice.z().start(), slice.z().end(), _step_z));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);

        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window.slide_window_slice_3D(slice));
}
