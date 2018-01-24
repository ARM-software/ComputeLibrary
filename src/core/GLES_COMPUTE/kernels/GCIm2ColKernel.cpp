/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#include "arm_compute/core/GLES_COMPUTE/kernels/GCIm2ColKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <tuple>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    }

    return Status{};
}
} // namespace

GCIm2ColKernel::GCIm2ColKernel()
    : _input(nullptr), _output(nullptr), _convolved_dims(), _kernel_dims(), _num_elems_processed_per_iteration(1), _run_func(nullptr)
{
}

void GCIm2ColKernel::configure(const IGCTensor *input, IGCTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    // Create kernel
    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    build_opts.insert("#define " + dt_name);

    if(has_bias)
    {
        build_opts.emplace("#define HAS_BIAS");
    }

    int stride_x = 0;
    int stride_y = 0;

    std::tie(stride_x, stride_y) = conv_info.stride();
    _kernel_dims = std::make_pair(kernel_dims.width, kernel_dims.height);

    const bool run_img2col_reduced = (output->info()->dimension(0) == (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))) && (TensorShape::num_max_dimensions >= 4)
                                     && (std::equal(input->info()->tensor_shape().cbegin() + 3,
                                                    input->info()->tensor_shape().cend(),
                                                    output->info()->tensor_shape().cbegin() + 1))
                                     && ((stride_x == 1) && (stride_y == 1) && !conv_info.has_padding());

    std::string kernel_name = "im2col_generic";
    if(!run_img2col_reduced)
    {
        if(input->info()->data_type() == DataType::F16 && _kernel_dims == std::pair<unsigned int, unsigned int>(1, 1))
        {
            build_opts.emplace("#define KERNEL_1x1");
        }

        build_opts.emplace("#define IM2COL_GENERIC");
        _convolved_dims = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1),
                                            kernel_dims.width, kernel_dims.height,
                                            conv_info);
        _num_elems_processed_per_iteration = 2;

        build_opts.emplace("#define KERNEL_WIDTH " + support::cpp11::to_string(kernel_dims.width));
        build_opts.emplace("#define KERNEL_HEIGHT " + support::cpp11::to_string(kernel_dims.height));
        build_opts.emplace("#define KERNEL_DEPTH " + support::cpp11::to_string(input->info()->dimension(2)));
        build_opts.emplace("#define CONVOLVED_WIDTH " + support::cpp11::to_string(_convolved_dims.first));
        build_opts.emplace("#define CONVOLVED_HEIGHT " + support::cpp11::to_string(_convolved_dims.second));
        build_opts.emplace("#define STRIDE_X " + support::cpp11::to_string(conv_info.stride().first));
        build_opts.emplace("#define STRIDE_Y " + support::cpp11::to_string(conv_info.stride().second));
        build_opts.emplace("#define PAD_LEFT " + support::cpp11::to_string(conv_info.pad_left()));
        build_opts.emplace("#define PAD_TOP " + support::cpp11::to_string(conv_info.pad_top()));
        build_opts.emplace("#define PAD_RIGHT " + support::cpp11::to_string(conv_info.pad_right()));
        build_opts.emplace("#define PAD_BOTTOM " + support::cpp11::to_string(conv_info.pad_bottom()));
        build_opts.emplace("#define SRC_WIDTH " + support::cpp11::to_string(input->info()->dimension(0)));
        build_opts.emplace("#define SRC_HEIGHT " + support::cpp11::to_string(input->info()->dimension(1)));

        _run_func = &GCIm2ColKernel::run_generic;
    }
    else
    {
        build_opts.emplace("#define IM2COL_REDUCED");
        kernel_name = "im2col_reduced";

        if(input->info()->data_type() == DataType::F32)
        {
            _num_elems_processed_per_iteration = 4 / input->info()->element_size();
        }
        else if(input->info()->data_type() == DataType::F16)
        {
            int input_width  = input->info()->dimension(0);
            int input_height = input->info()->dimension(1);

            build_opts.emplace("#define IMAGE_SIZE " + support::cpp11::to_string(input_width * input_height));
            if(input_width % 8 == 0)
            {
                _num_elems_processed_per_iteration = 8;
                build_opts.emplace("#define IM2COL_REDUCED_8X");
            }
            else if(input_width % 4 == 0)
            {
                _num_elems_processed_per_iteration = 4;
                build_opts.emplace("#define IM2COL_REDUCED_4X");
            }
            else if(input_width % 2 == 0)
            {
                _num_elems_processed_per_iteration = 2;
                build_opts.emplace("#define IM2COL_REDUCED_2X");
            }
            else
            {
                _num_elems_processed_per_iteration = 2;
                build_opts.emplace("#define IM2COL_REDUCED_GENERIC");
            }
        }

        _run_func = &GCIm2ColKernel::run_reduced;
    }

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel(kernel_name, build_opts));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(_num_elems_processed_per_iteration));

    if(input->info()->data_type() == DataType::F16)
    {
        // Calculate input right and bottom border
        const int input_width         = input->info()->dimension(0);
        const int input_height        = input->info()->dimension(1);
        int       input_total_width   = input->info()->padding().left + input_width + input->info()->padding().right;
        int       input_padding_right = ceil_to_multiple(input_total_width, _num_elems_processed_per_iteration) - input_total_width;
        input_total_width             = input_width + input_padding_right + input->info()->padding().right;
        AccessWindowStatic input_access(input->info(), 0, 0, input_total_width, input_height);

        // Calculate output right and bottom border
        const int          output_width         = output->info()->dimension(0);
        const int          output_height        = output->info()->dimension(1);
        const int          output_padding_right = ceil_to_multiple(output_width, _num_elems_processed_per_iteration) - output_width;
        AccessWindowStatic output_access(output->info(), 0, 0, output_width + output_padding_right, output_height);

        update_window_and_padding(win, input_access, output_access);
    }

    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    if(!run_img2col_reduced)
    {
        // set the Z dimension's step same size as the whole dimension so that one can't split across the Z dimension
        win.set_dimension_step(Window::DimZ, win[Window::DimZ].end() - win[Window::DimZ].start());
    }

    IGCKernel::configure(win);
}

Status GCIm2ColKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias)
{
    ARM_COMPUTE_UNUSED(kernel_dims);
    ARM_COMPUTE_UNUSED(conv_info);
    ARM_COMPUTE_UNUSED(has_bias);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    return Status{};
}

void GCIm2ColKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON(_run_func == nullptr);
    (this->*_run_func)(window);
}

void GCIm2ColKernel::run_generic(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IGCKernel::window(), window);

    // Get initial windows
    Window window_collapsed = window.collapse_if_possible(IGCKernel::window(), Window::DimZ);

    // Change the Z dimension's step back to 1
    window_collapsed.set_dimension_step(Window::DimZ, 1);

    Window slice     = window_collapsed.first_slice_window_3D();
    Window slice_in  = window_collapsed.first_slice_window_3D();
    Window slice_out = window_collapsed.first_slice_window_3D();

    // Setup slice
    slice.set(Window::DimX, Window::Dimension(0, static_cast<int>(_convolved_dims.first), 1));
    slice.set(Window::DimY, Window::Dimension(0, static_cast<int>(_convolved_dims.second), 1));

    // Setup output slice
    slice_out.set(Window::DimX, Window::Dimension(0, _output->info()->dimension(0), _num_elems_processed_per_iteration));
    slice_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));
    slice_out.set(Window::DimZ, Window::Dimension(0, 1, 1));

    // we need top/left pad to be included in valid region
    if(_input->info()->data_type() == DataType::F16)
    {
        (dynamic_cast<TensorInfo *>(_input->info()))->init(_input->info()->tensor_shape(), _input->info()->num_channels(), _input->info()->data_type(), _input->info()->strides_in_bytes(), 0,
                                                           _input->info()->total_size(), _input->info()->fixed_point_position());
    }

    _kernel.use();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, 1, slice_in);
        add_2D_tensor_argument(idx, _output, 2, slice_out);
        _kernel.set_argument(idx++, static_cast<unsigned int>(_input->info()->strides_in_bytes()[3]));
        _kernel.set_argument(idx++, static_cast<unsigned int>(_output->info()->strides_in_bytes()[3]));
        _kernel.update_shader_params();

        enqueue(*this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice) && window_collapsed.slide_window_slice_3D(slice_out) && window_collapsed.slide_window_slice_3D(slice_in));
}

void GCIm2ColKernel::run_reduced(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IGCKernel::window(), window);

    Window out_window;
    out_window.use_tensor_dimensions(_output->info()->tensor_shape());

    Window out_slice = out_window.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_3D();

    _kernel.use();

    // Run kernel
    do
    {
        // Set arguments
        unsigned int idx = 0;

        add_3D_tensor_argument(idx, _input, 1, in_slice);
        add_1D_tensor_argument(idx, _output, 2, out_slice);
        _kernel.set_argument(idx++, _input->info()->dimension(0));
        _kernel.set_argument(idx++, _input->info()->dimension(1));
        _kernel.update_shader_params();

        enqueue(*this, in_slice);
    }
    while(window.slide_window_slice_3D(in_slice) && out_window.slide_window_slice_1D(out_slice));
}
