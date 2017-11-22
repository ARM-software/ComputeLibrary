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

CLWeightsReshapeKernel::CLWeightsReshapeKernel()
    : _input(nullptr), _biases(nullptr), _output(nullptr)
{
}

void CLWeightsReshapeKernel::configure(const ICLTensor *input, const ICLTensor *biases, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    const DataType data_type = input->info()->data_type();

    // Calculate output shape
    TensorShape output_shape{ input->info()->tensor_shape() };
    output_shape.collapse(3);
    const size_t tmp_dim = output_shape[0];
    output_shape.set(0, output_shape[1]);
    output_shape.set(1, tmp_dim + (biases != nullptr ? 1 : 0));

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON(is_data_type_quantized_asymmetric(data_type));
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, biases);
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 4) && (biases->info()->num_dimensions() != 1));
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 5) && (biases->info()->num_dimensions() != 2));
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 4) && (biases->info()->dimension(0) != input->info()->tensor_shape()[3]));
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 5) && (biases->info()->dimension(0) != input->info()->tensor_shape()[3] || biases->info()->dimension(1) != input->info()->tensor_shape()[4]));
    }

    _biases = biases;
    _output = output;
    _input  = input;

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option_if(biases != nullptr, "-DHAS_BIAS");
    build_opts.add_option_if(is_data_type_fixed_point(data_type), "-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("reshape_to_columns", build_opts.options()));

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

void CLWeightsReshapeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window out_window;
    out_window.use_tensor_dimensions(_output->info()->tensor_shape());

    Window in_slice  = window.first_slice_window_3D();
    Window out_slice = out_window.first_slice_window_2D();

    Window biases_window;
    Window biases_slice;

    if(_biases != nullptr)
    {
        biases_window.use_tensor_dimensions(_biases->info()->tensor_shape());
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
