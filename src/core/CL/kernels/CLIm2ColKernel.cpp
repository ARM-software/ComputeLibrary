/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLIm2ColKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <tuple>

using namespace arm_compute;

CLIm2ColKernel::CLIm2ColKernel()
    : _input(nullptr), _output(nullptr), _convolved_dims(), _num_elems_processed_per_iteration(1), _run_func(nullptr)
{
}

void CLIm2ColKernel::configure(const ICLTensor *input, ICLTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    _input  = input;
    _output = output;

    // Create kernel
    std::set<std::string> build_opts;
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.emplace((has_bias ? "-DHAS_BIAS" : ""));

    if(is_data_type_fixed_point(input->info()->data_type()))
    {
        build_opts.emplace("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));
    }

    int pad_x    = 0;
    int pad_y    = 0;
    int stride_x = 0;
    int stride_y = 0;
    std::tie(pad_x, pad_y)       = conv_info.pad();
    std::tie(stride_x, stride_y) = conv_info.stride();

    const bool run_img2col_reduced = (output->info()->dimension(0) == (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))) && (TensorShape::num_max_dimensions >= 4)
                                     && (std::equal(input->info()->tensor_shape().cbegin() + 3,
                                                    input->info()->tensor_shape().cend(),
                                                    output->info()->tensor_shape().cbegin() + 1))
                                     && ((stride_x == 1) && (stride_y == 1) && (pad_x == 0) && (pad_y == 0));

    if(!run_img2col_reduced)
    {
        _convolved_dims = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1),
                                            kernel_dims.width, kernel_dims.height,
                                            conv_info);
        _num_elems_processed_per_iteration = output->info()->dimension(0);

        build_opts.emplace("-DKERNEL_WIDTH=" + support::cpp11::to_string(kernel_dims.width));
        build_opts.emplace("-DKERNEL_HEIGHT=" + support::cpp11::to_string(kernel_dims.height));
        build_opts.emplace("-DKERNEL_DEPTH=" + support::cpp11::to_string(input->info()->dimension(2)));
        build_opts.emplace("-DCONVOLVED_WIDTH=" + support::cpp11::to_string(_convolved_dims.first));
        build_opts.emplace("-DCONVOLVED_HEIGHT=" + support::cpp11::to_string(_convolved_dims.second));
        build_opts.emplace("-DSTRIDE_X=" + support::cpp11::to_string(conv_info.stride().first));
        build_opts.emplace("-DSTRIDE_Y=" + support::cpp11::to_string(conv_info.stride().second));
        build_opts.emplace("-DPAD_X=" + support::cpp11::to_string(conv_info.pad().first));
        build_opts.emplace("-DPAD_Y=" + support::cpp11::to_string(conv_info.pad().second));
        build_opts.emplace("-DSRC_WIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));
        build_opts.emplace("-DSRC_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(1)));

        if(kernel_dims.width == 3 && kernel_dims.height == 3 && conv_info.pad().first == 0 && conv_info.pad().second == 0)
        {
            _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("im2col_kernel3x3_padx0_pady0", build_opts));
        }
        else
        {
            _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("im2col_generic", build_opts));
        }
        _run_func = &CLIm2ColKernel::run_generic;
    }
    else
    {
        _num_elems_processed_per_iteration = 1;
        _kernel                            = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("im2col_reduced", build_opts));
        _run_func                          = &CLIm2ColKernel::run_reduced;
    }

    // Configure  kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    // The CLIm2ColKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));
    if(!run_img2col_reduced)
    {
        // set the Z dimension's step same size as the whole dimension so that one can't split across the Z dimension
        win.set_dimension_step(Window::DimZ, win[Window::DimZ].end() - win[Window::DimZ].start());
    }

    ICLKernel::configure(win);
}

void CLIm2ColKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON(_run_func == nullptr);
    (this->*_run_func)(window, queue);
}

void CLIm2ColKernel::run_generic(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    // Get initial windows
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    // Change the Z dimension's step back to 1
    window_collapsed.set_dimension_step(Window::DimZ, 1);

    Window slice     = window_collapsed.first_slice_window_3D();
    Window slice_in  = window_collapsed.first_slice_window_3D();
    Window slice_out = window_collapsed.first_slice_window_3D();

    // Setup slice
    slice.set(Window::DimX, Window::Dimension(0, static_cast<int>(_convolved_dims.first), 1));
    slice.set(Window::DimY, Window::Dimension(0, static_cast<int>(_convolved_dims.second), 1));

    // Setup input slice
    // The first three dimensions of the input are increased by the inner loops
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Setup output slice
    slice_out.set(Window::DimX, Window::Dimension(0, _output->info()->dimension(0), _num_elems_processed_per_iteration));
    slice_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));
    slice_out.set(Window::DimZ, Window::Dimension(0, 1, 1));

    // Set the local-workgroup size
    _lws_hint = cl::NDRange(4, 4, 4);

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_input->info()->dimension(2)));
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_input->info()->strides_in_bytes()[3]));
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_output->info()->strides_in_bytes()[3]));
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window_collapsed.slide_window_slice_3D(slice) && window_collapsed.slide_window_slice_3D(slice_out) && window_collapsed.slide_window_slice_3D(slice_in));
}

void CLIm2ColKernel::run_reduced(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window out_window;
    out_window.use_tensor_dimensions(_output->info()->tensor_shape());

    Window out_slice = out_window.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_3D();

    // Run kernel
    do
    {
        // Set arguments
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, in_slice);
        add_1D_tensor_argument(idx, _output, out_slice);

        _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(0));
        _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(1));
        enqueue(queue, *this, in_slice);
    }
    while(window.slide_window_slice_3D(in_slice) && out_window.slide_window_slice_1D(out_slice));
}
