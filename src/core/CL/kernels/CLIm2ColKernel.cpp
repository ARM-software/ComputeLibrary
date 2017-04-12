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
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <cmath>
#include <tuple>

using namespace arm_compute;

CLIm2ColKernel::CLIm2ColKernel()
    : _input(nullptr), _output(nullptr), _convolved_dims(), _conv_info(), _kernel_size(0), _num_elems_processed_per_iteration(1), _run_func(nullptr)
{
}

void CLIm2ColKernel::configure(const ICLTensor *input, ICLTensor *output, std::pair<unsigned int, unsigned int> convolved_dims, const PadStrideInfo &conv_info, bool has_bias)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);

    _input  = input;
    _output = output;

    // Create kernel
    std::set<std::string> build_opts;
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.emplace((has_bias ? "-DHAS_BIAS" : ""));

    int pad_x, pad_y, stride_x, stride_y = 0;
    std::tie(pad_x, pad_y)       = conv_info.pad();
    std::tie(stride_x, stride_y) = conv_info.stride();

    // Run the fully connected path if convolved_dims, stride_x, stride_y are 1 and pad_x, pad_y are 0.
    bool is_fc = ((convolved_dims.first & convolved_dims.second & stride_x & stride_y) == 1) && ((pad_x | pad_y) == 0);

    if(!is_fc)
    {
        _convolved_dims                    = convolved_dims;
        _conv_info                         = conv_info;
        _kernel_size                       = std::sqrt((output->info()->dimension(0) - (has_bias ? 1 : 0)) / input->info()->dimension(2));
        _num_elems_processed_per_iteration = output->info()->dimension(0);

        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("im2col_generic", build_opts));

        // Create static kernel arguments
        const cl_int2 input_dims =
        {
            {
                static_cast<cl_int>(input->info()->dimension(0)),
                static_cast<cl_int>(input->info()->dimension(1)),
            }
        };
        const cl_int2 strides =
        {
            {
                stride_x,
                stride_y,
            }
        };
        const cl_int2 paddings =
        {
            {
                pad_x,
                pad_y,
            }
        };

        // Set static kernel arguments
        unsigned int idx = num_arguments_per_2D_tensor() + num_arguments_per_3D_tensor();
        _kernel.setArg<cl_int>(idx++, _kernel_size);
        _kernel.setArg<cl_int>(idx++, input->info()->dimension(2) /* depth */);
        _kernel.setArg<cl_int>(idx++, _convolved_dims.first /* output width */);
        _kernel.setArg<cl_int2>(idx++, input_dims);
        _kernel.setArg<cl_int2>(idx++, strides);
        _kernel.setArg<cl_int2>(idx++, paddings);

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

    int pad_x, pad_y, stride_x, stride_y = 0;
    std::tie(pad_x, pad_y)       = _conv_info.pad();
    std::tie(stride_x, stride_y) = _conv_info.stride();

    // Get initial windows
    Window slice_in  = window.first_slice_window_3D();
    Window slice_out = window.first_slice_window_3D();

    // Setup input slice
    const int start_x = -pad_x + (_kernel_size >> 1);
    const int start_y = -pad_y + (_kernel_size >> 1);
    slice_in.set(Window::DimX, Window::Dimension(start_x, (static_cast<int>(_convolved_dims.first) * stride_x) + start_x, stride_x));
    slice_in.set(Window::DimY, Window::Dimension(start_y, (static_cast<int>(_convolved_dims.second) * stride_y) + start_y, stride_y));
    slice_in.set(Window::DimZ, Window::Dimension(0, 1, 1));

    // Setup output slice
    slice_out.set(Window::DimX, Window::Dimension(0, _output->info()->dimension(0), _num_elems_processed_per_iteration));
    slice_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));
    slice_out.set(Window::DimZ, Window::Dimension(0, 1, 1));

    do
    {
        // Set inputs
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_in);
    }
    while(window.slide_window_slice_3D(slice_out) && window.slide_window_slice_3D(slice_in));
}

void CLIm2ColKernel::run_reduced(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window out_window;
    out_window.use_tensor_dimensions(_output->info());

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
