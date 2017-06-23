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
#include "arm_compute/core/CL/kernels/CLWeightsReshapeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

CLWeightsReshapeKernel::CLWeightsReshapeKernel(bool is_shared)
    : _is_shared(is_shared), _input(nullptr), _biases(nullptr), _output(nullptr)
{
}

void CLWeightsReshapeKernel::configure(const ICLTensor *input, const ICLTensor *biases, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F16, DataType::F32);
    if(_is_shared)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(4) != (output->info()->dimension(2)));
        ARM_COMPUTE_ERROR_ON(input->info()->num_dimensions() > 5);
        ARM_COMPUTE_ERROR_ON(output->info()->num_dimensions() > 3);
    }
    else
    {
        ARM_COMPUTE_ERROR_ON(input->info()->num_dimensions() > 4);
        ARM_COMPUTE_ERROR_ON(output->info()->num_dimensions() > 2);
    }

    // Check biases
    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::F16, DataType::F32);
    }

    _biases = biases;
    _output = output;
    _input  = input;

    // Create build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.emplace(((biases != nullptr) ? "-DHAS_BIAS" : ""));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("reshape_to_columns", build_opts));

    // Set static arguments
    unsigned int idx = num_arguments_per_3D_tensor() + num_arguments_per_2D_tensor();
    idx += (biases != nullptr) ? num_arguments_per_1D_tensor() : 0;
    _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(0));
    _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(1));
    _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(2));
    _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(3));

    // Configure window
    Window win = calculate_max_window(*input->info(), Steps());
    // The CLWeightsReshapeKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));
    ICLKernel::configure(win);
}

CLConvolutionLayerWeightsReshapeKernel::CLConvolutionLayerWeightsReshapeKernel()
    : CLWeightsReshapeKernel(false)
{
}

void CLConvolutionLayerWeightsReshapeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window out_window;
    out_window.use_tensor_dimensions(_output->info());

    Window in_slice  = window.first_slice_window_3D();
    Window out_slice = out_window.first_slice_window_2D();

    // Set arguments
    unsigned idx = 0;
    add_3D_tensor_argument(idx, _input, in_slice);
    add_2D_tensor_argument(idx, _output, out_slice);
    if(_biases != nullptr)
    {
        Window biases_slice;
        biases_slice.set(Window::DimX, Window::Dimension(0, _biases->info()->tensor_shape().x(), 1));
        add_1D_tensor_argument(idx, _biases, biases_slice);
    }

    // Run kernel
    enqueue(queue, *this, in_slice);
}

CLLocallyConnectedLayerWeightsReshapeKernel::CLLocallyConnectedLayerWeightsReshapeKernel()
    : CLWeightsReshapeKernel(true)
{
}

void CLLocallyConnectedLayerWeightsReshapeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window out_window;
    out_window.use_tensor_dimensions(_output->info());

    Window in_slice  = window.first_slice_window_3D();
    Window out_slice = out_window.first_slice_window_2D();

    Window biases_window;
    Window biases_slice;

    if(_biases != nullptr)
    {
        biases_window.use_tensor_dimensions(_biases->info());
        biases_slice = biases_window.first_slice_window_1D();
    }

    do
    {
        // Set arguments
        unsigned idx = 0;
        add_3D_tensor_argument(idx, _input, in_slice);
        add_2D_tensor_argument(idx, _output, out_slice);
        if(_biases != nullptr)
        {
            add_1D_tensor_argument(idx, _biases, biases_slice);
            biases_window.slide_window_slice_1D(biases_slice);
        }

        // Run kernel
        enqueue(queue, *this, in_slice);
    }
    while(window.slide_window_slice_4D(in_slice) && out_window.slide_window_slice_2D(out_slice));
}
