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
#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

CLGEMMInterleave4x4Kernel::CLGEMMInterleave4x4Kernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLGEMMInterleave4x4Kernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QS8, DataType::U16, DataType::S16, DataType::U32, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    TensorShape output_shape = input->info()->tensor_shape();
    output_shape.set(0, input->info()->dimension(0) * 4);
    output_shape.set(1, std::ceil(input->info()->dimension(1) / 4.0f));

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    _input  = input;
    _output = output;

    // Create kernel
    std::string data_type_name;
    data_type_name = support::cpp11::to_string(input->info()->element_size() * 8) + "bit";
    _kernel        = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gemm_interleave4x4_" + data_type_name));

    // Configure kernel window
    const unsigned int     num_elems_processed_per_iteration_x = max_cl_vector_width / data_size_from_type(input->info()->data_type());
    constexpr unsigned int num_elems_processed_per_iteration_y = 4;
    const unsigned int     num_elems_written_per_iteration     = num_elems_processed_per_iteration_x * num_elems_processed_per_iteration_y;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    AccessWindowRectangle input_access(input->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);
    AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_written_per_iteration, 1, 4.f, 0.25f);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    ICLKernel::configure(win);
}

void CLGEMMInterleave4x4Kernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    /*
     * This kernel puts the values in a 4x4 block of Matrix A on the same row (Interleaved values)
     *         |a00 a01 a02 a03|
     *         |a10 a11 a12 a13|
     *         |a20 a21 a22 a23| = | a00 a10 a20 a30 || a01 a11 a21 a31 || a02 a12 a22 a32 || a03 a13 a23 a33 |
     *         |a30 a31 a32 a33|
     *
     * After this operation, the output matrix will have the following shape: [ height * 4, width / 4 ]
     */
    Window in_slice  = window.first_slice_window_2D();
    Window out_slice = window.first_slice_window_2D();

    // Change x and y steps for the slide of output tensor
    out_slice.scale(Window::DimX, 4.f);
    out_slice.scale(Window::DimY, 0.25f);

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, in_slice);
        add_2D_tensor_argument(idx, _output, out_slice);
        enqueue(queue, *this, in_slice);
    }
    while(window.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(out_slice));
}
