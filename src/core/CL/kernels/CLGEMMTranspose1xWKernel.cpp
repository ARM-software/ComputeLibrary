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
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"

#include "arm_compute/core/AccessWindowTranspose.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cmath>

using namespace arm_compute;

void CLGEMMTranspose1xWKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QS8, DataType::U16, DataType::S16, DataType::U32, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    TensorShape  output_shape{ input->info()->tensor_shape() };
    const size_t transpose_w = 16 / input->info()->element_size();
    output_shape.set(0, input->info()->dimension(1) * transpose_w);
    output_shape.set(1, static_cast<size_t>(std::ceil((input->info()->dimension(0) / static_cast<float>(transpose_w)))));

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    const unsigned int num_elems_processed_per_iteration = max_cl_vector_width / data_size_from_type(input->info()->data_type());
    const float        scale_x                           = num_elems_processed_per_iteration;
    ARM_COMPUTE_ERROR_ON((0 == static_cast<int>(input->info()->dimension(0) * (1.f / scale_x))));

    _input  = input;
    _output = output;

    /*
     * Following an example of how the transposition1xW works when the input data type is F32
     *
     *         |a00 a01 a02 a03|
     *         |a10 a11 a12 a13|
     *         |a20 a21 a22 a23| = | a00 a01 a02 a03 || a10 a11 a12 a13 || a20 a21 a22 a23 || a30 a31 a32 a33 |
     *         |a30 a31 a32 a33|
     *
     * The output matrix will have the following shape: [ height * W, ceil(width / W) ], where W = (16 / element size of the tensor)
     */
    // Create kernel
    std::string kernel_name = "gemm_transpose1x" + support::cpp11::to_string(num_elems_processed_per_iteration);
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name));

    // Configure window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowTranspose  output_access(output->info(), 0, 0, num_elems_processed_per_iteration, 1, scale_x, 1.f / scale_x);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), input->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLGEMMTranspose1xWKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    // Output is transposed
    Window out_window(window);
    out_window.set(Window::DimX, window.y());
    out_window.set(Window::DimY, window.x());

    Window in_slice  = window.first_slice_window_2D();
    Window out_slice = out_window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, in_slice);
        add_2D_tensor_argument(idx, _output, out_slice);
        enqueue(queue, *this, in_slice, _lws_hint);
    }
    while(window.slide_window_slice_2D(in_slice) && out_window.slide_window_slice_2D(out_slice));
}
