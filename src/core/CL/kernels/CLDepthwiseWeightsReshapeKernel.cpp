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
#include "arm_compute/core/CL/kernels/CLDepthwiseWeightsReshapeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLDepthwiseWeightsReshapeKernel::CLDepthwiseWeightsReshapeKernel()
    : _input(nullptr), _biases(nullptr), _output(nullptr)
{
}

void CLDepthwiseWeightsReshapeKernel::configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *biases)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) != output->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(0) != (input->info()->dimension(0) * input->info()->dimension(1) + ((biases != nullptr) ? 1 : 0)));

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != input->info()->dimension(2));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    _input  = input;
    _biases = biases;
    _output = output;

    // Create kernel
    std::set<std::string> build_opts;

    build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.emplace("-DSRC_WIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));
    if(_biases != nullptr)
    {
        build_opts.emplace("-DHAS_BIAS");
    }

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("depthwise_weights_reshape", build_opts));

    // Configure  kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    // The CLDepthwiseWeightsReshapeKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLDepthwiseWeightsReshapeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window slice     = window.first_slice_window_3D();
    Window slice_out = window.first_slice_window_2D();

    // Setup slice
    slice.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _input->info()->dimension(0)));
    slice.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(1), 1));
    slice.set(Window::DimZ, Window::Dimension(0, _input->info()->dimension(2), 1));

    // Setup output slice
    // The first two dimensions of the output are increased by the inner loops
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    // Set biases
    if(_biases != nullptr)
    {
        unsigned int idx = num_arguments_per_3D_tensor() + num_arguments_per_2D_tensor();
        Window       slice_biases;
        slice_biases.use_tensor_dimensions(_biases->info()->tensor_shape());
        add_1D_tensor_argument(idx, _biases, slice_biases);
    }

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_3D(slice) && window.slide_window_slice_2D(slice_out));
}
