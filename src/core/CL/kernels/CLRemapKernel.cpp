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
#include "arm_compute/core/CL/kernels/CLRemapKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <algorithm>

using namespace arm_compute;

CLRemapKernel::CLRemapKernel()
    : _input(nullptr), _output(nullptr), _map_x(nullptr), _map_y(nullptr)
{
}

BorderSize CLRemapKernel::border_size() const
{
    return BorderSize(1);
}

void CLRemapKernel::configure(const ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output, InterpolationPolicy policy, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(map_x, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(map_y, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MSG(policy == InterpolationPolicy::AREA, "Area interpolation is not supported!");
    ARM_COMPUTE_UNUSED(border_undefined);

    _input  = input;
    _output = output;
    _map_x  = map_x;
    _map_y  = map_y;

    // Create kernel
    std::set<std::string> build_opts         = { ("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())) };
    std::string           interpolation_name = string_from_interpolation_policy(policy);
    std::transform(interpolation_name.begin(), interpolation_name.end(), interpolation_name.begin(), ::tolower);
    std::string kernel_name = "remap_" + interpolation_name;
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));

    // Configure window
    constexpr unsigned int num_elems_processed_per_iteration = 4;

    const int total_right  = ceil_to_multiple(input->info()->dimension(0), num_elems_processed_per_iteration);
    const int access_right = total_right + (((total_right - input->info()->dimension(0)) == 0) ? border_size().right : 0);

    Window             win = calculate_max_window(*_output->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowStatic input_access(input->info(), -border_size().left, -border_size().top, access_right, input->info()->dimension(1) + border_size().bottom);

    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);

    // Set static arguments
    unsigned int idx = 4 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg<cl_float>(idx++, input->info()->dimension(0));
    _kernel.setArg<cl_float>(idx++, input->info()->dimension(1));
}

void CLRemapKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _output, slice);
        add_2D_tensor_argument(idx, _map_x, slice);
        add_2D_tensor_argument(idx, _map_y, slice);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
