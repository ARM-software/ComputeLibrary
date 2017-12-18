/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLLocallyConnectedMatrixMultiplyKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
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

#include <set>
#include <sstream>
#include <string>

using namespace arm_compute;

CLLocallyConnectedMatrixMultiplyKernel::CLLocallyConnectedMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr)
{
}

void CLLocallyConnectedMatrixMultiplyKernel::configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1, output);
    ARM_COMPUTE_ERROR_ON(input0->info()->dimension(0) != input1->info()->dimension(1));

    _input0 = input0;
    _input1 = input1;
    _output = output;

    if(output->info()->dimension(1) == 196)
    {
        _lws_hint = cl::NDRange(1, 7);
    }
    else
    {
        _lws_hint = cl::NDRange(8, 8);
    }

    std::ostringstream    mm_arguments;
    std::set<std::string> build_opts;

    mm_arguments << "-DWIDTH_VECTOR_A=" << input0->info()->dimension(0) << " ";
    build_opts.emplace(mm_arguments.str());

    // Create kernel
    std::string data_type_name = lower_string(string_from_data_type(input0->info()->data_type()));
    _kernel                    = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(("gemm_lc_vm_" + data_type_name), build_opts));

    // Configure window kernel
    const unsigned int num_elems_processed_per_iteration_x = max_cl_vector_width / data_size_from_type(input0->info()->data_type());

    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration_x));

    AccessWindowRectangle input0_access(input0->info(), 0, 0, num_elems_processed_per_iteration_x, 1);
    AccessWindowRectangle input1_access(input1->info(), 0, 0, num_elems_processed_per_iteration_x, 1);
    AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_processed_per_iteration_x, 1);

    update_window_and_padding(win, input0_access, input1_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLLocallyConnectedMatrixMultiplyKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    Window matrix_b_window;
    matrix_b_window.use_tensor_dimensions(_input1->info()->tensor_shape());
    Window slice_matrix_b = matrix_b_window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input0, slice);
        add_3D_tensor_argument(idx, _input1, slice_matrix_b);
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window.slide_window_slice_2D(slice));
}
