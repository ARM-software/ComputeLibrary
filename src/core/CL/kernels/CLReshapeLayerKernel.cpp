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
#include "arm_compute/core/CL/kernels/CLReshapeLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <string>

using namespace arm_compute;

CLReshapeLayerKernel::CLReshapeLayerKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLReshapeLayerKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QS8, DataType::QASYMM8,
                                                  DataType::U16, DataType::S16, DataType::QS16,
                                                  DataType::U32, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    ARM_COMPUTE_ERROR_ON(input->info()->tensor_shape().total_size() != output->info()->tensor_shape().total_size());

    _input  = input;
    _output = output;

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    // Create kernel
    std::set<std::string> build_opts = { "-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()) };
    _kernel                          = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("reshape_layer", build_opts));

    // Add static arguments
    const cl_int2 input_shape =
    {
        {
            static_cast<cl_int>(_input->info()->tensor_shape()[0]),
            static_cast<cl_int>(_input->info()->tensor_shape()[1])
        }
    };
    const cl_int2 output_shape =
    {
        {
            static_cast<cl_int>(_output->info()->tensor_shape()[0]),
            static_cast<cl_int>(_output->info()->tensor_shape()[1])
        }
    };
    unsigned int idx = 2 * num_arguments_per_3D_tensor(); // Skip the input and output parameters
    _kernel.setArg<cl_int2>(idx++, input_shape);
    _kernel.setArg<cl_int2>(idx++, output_shape);

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowStatic     output_access(output->info(), 0, 0, output->info()->tensor_shape().x(), output->info()->tensor_shape().y());
    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLReshapeLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    // Set inputs
    unsigned int idx = 0;
    add_3D_tensor_argument(idx, _input, window_collapsed);
    add_3D_tensor_argument(idx, _output, window_collapsed);
    enqueue(queue, *this, slice);
}
