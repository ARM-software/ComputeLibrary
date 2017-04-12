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
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(1) != std::ceil(static_cast<float>(input->info()->dimension(0)) / 16.0f)) && (input->info()->data_type() == DataType::U8));
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(1) != std::ceil(static_cast<float>(input->info()->dimension(0)) / 8.0f)) && (input->info()->data_type() == DataType::F16));
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(1) != std::ceil(static_cast<float>(input->info()->dimension(0)) / 4.0f)) && (input->info()->data_type() == DataType::F32));

    _input                                               = input;
    _output                                              = output;
    const unsigned int num_elems_processed_per_iteration = max_cl_vector_width / data_size_from_type(input->info()->data_type());

    /*
     * Following an example of how the transposition1xW works when the input data type is F32
     *
     *         |a00 a01 a02 a03|
     *         |a10 a11 a12 a13|
     *         |a20 a21 a22 a23| = | a00 a01 a02 a03 || a10 a11 a12 a13 || a20 a21 a22 a23 || a30 a31 a32 a33 |
     *         |a30 a31 a32 a33|
     *
     * If the input data type is F32, the output matrix will have the following shape: [ height * 4, width / 4 ]
     * If the input data type is F16, the output matrix will have the following shape: [ height * 8, width / 8 ]
     */
    // Create kernel
    std::string data_type_name = lower_string(string_from_data_type(input->info()->data_type()));
    std::string kernel_name    = "gemm_transpose1x" + val_to_string(num_elems_processed_per_iteration) + "_" + data_type_name;
    _kernel                    = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name));

    // Configure window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowTranspose  output_access(output->info(), 0, 0, num_elems_processed_per_iteration, 1);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}
