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
#include "arm_compute/core/CL/kernels/CLCol2ImKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <cmath>

using namespace arm_compute;

CLCol2ImKernel::CLCol2ImKernel()
    : _input(nullptr), _output(nullptr), _convolved_dims()
{
}

void CLCol2ImKernel::configure(const ICLTensor *input, ICLTensor *output, std::pair<unsigned int, unsigned int> convolved_dims)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    TensorShape output_shape = input->info()->tensor_shape();
    output_shape.set(0, convolved_dims.first);
    output_shape.set(1, convolved_dims.second);
    output_shape.set(2, input->info()->tensor_shape()[0]);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    _input          = input;
    _output         = output;
    _convolved_dims = convolved_dims;

    // Create kernel
    std::set<std::string> build_opts = { ("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())) };
    build_opts.emplace("-DWIDTH_OUTPUT=" + support::cpp11::to_string(_convolved_dims.first));
    if(is_data_type_fixed_point(input->info()->data_type()))
    {
        build_opts.emplace("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));
    }

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("col2im", build_opts));

    // Configure window
    Window win = calculate_max_window(*input->info(), Steps());

    // The CLCol2ImKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLCol2ImKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);
    // The collapse method rely on the assumption that the third dimension of input buffer is 1
    ARM_COMPUTE_ERROR_ON(window.z().end() != 1);

    Window collapsed_window = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = collapsed_window.first_slice_window_3D();

    // Set static kernel arguments
    unsigned int idx = 2 * num_arguments_per_3D_tensor();
    _kernel.setArg<cl_uint>(idx++, _output->info()->strides_in_bytes()[3]);

    do
    {
        // Set inputs
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(collapsed_window.slide_window_slice_3D(slice));
}
