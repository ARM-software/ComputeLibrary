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
#include "arm_compute/core/CL/kernels/CLMinMaxLocationKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <climits>

using namespace arm_compute;

CLMinMaxKernel::CLMinMaxKernel()
    : _input(nullptr), _min_max(), _data_type_max_min()
{
}

void CLMinMaxKernel::configure(const ICLImage *input, cl::Buffer *min_max)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON(min_max == nullptr);

    _input                                               = input;
    _min_max                                             = min_max;
    const unsigned int num_elems_processed_per_iteration = input->info()->dimension(0);

    switch(input->info()->data_type())
    {
        case DataType::U8:
            _data_type_max_min[0] = UCHAR_MAX;
            _data_type_max_min[1] = 0;
            break;
        case DataType::S16:
            _data_type_max_min[0] = SHRT_MAX;
            _data_type_max_min[1] = SHRT_MIN;
            break;
        default:
            ARM_COMPUTE_ERROR("You called with the wrong image data types");
    }

    // Set kernel build options
    std::set<std::string> build_opts;
    build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.emplace("-DDATA_TYPE_MAX=" + val_to_string<int>(_data_type_max_min[0]));
    build_opts.emplace("-DDATA_TYPE_MIN=" + val_to_string<int>(_data_type_max_min[1]));
    build_opts.emplace((0 != (num_elems_processed_per_iteration % max_cl_vector_width)) ? "-DNON_MULTIPLE_OF_16" : "");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("minmax", build_opts));

    // Set fixed arguments
    unsigned int idx = num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, *_min_max);
    _kernel.setArg<cl_uint>(idx++, input->info()->dimension(0));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration));
    ICLKernel::configure(win);
}

void CLMinMaxKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Reset mininum and maximum values
    queue.enqueueWriteBuffer(*_min_max, CL_FALSE /* blocking */, 0, _data_type_max_min.size() * sizeof(int), _data_type_max_min.data());

    Window slice = window.first_slice_window_2D();
    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}

CLMinMaxLocationKernel::CLMinMaxLocationKernel()
    : _input(nullptr), _min_max_count(nullptr)
{
}

void CLMinMaxLocationKernel::configure(const ICLImage *input, cl::Buffer *min_max, cl::Buffer *min_max_count, ICLCoordinates2DArray *min_loc, ICLCoordinates2DArray *max_loc)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON(min_max == nullptr);
    ARM_COMPUTE_ERROR_ON(min_max_count == nullptr && min_loc == nullptr && max_loc == nullptr);

    _input         = input;
    _min_max_count = min_max_count;

    // Set kernel build options
    std::set<std::string> build_opts;
    build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.emplace((min_max_count != nullptr) ? "-DCOUNT_MIN_MAX" : "");
    build_opts.emplace((min_loc != nullptr) ? "-DLOCATE_MIN" : "");
    build_opts.emplace((max_loc != nullptr) ? "-DLOCATE_MAX" : "");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("minmaxloc", build_opts));

    // Set static arguments
    unsigned int idx = num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, *min_max);
    _kernel.setArg(idx++, *min_max_count);
    if(min_loc != nullptr)
    {
        _kernel.setArg(idx++, min_loc->cl_buffer());
        _kernel.setArg<cl_uint>(idx++, min_loc->max_num_values());
    }
    if(max_loc != nullptr)
    {
        _kernel.setArg(idx++, max_loc->cl_buffer());
        _kernel.setArg<cl_uint>(idx++, max_loc->max_num_values());
    }

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 1;
    Window                 win                               = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration));
    ICLKernel::configure(win);
}

void CLMinMaxLocationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    static const unsigned int zero_count = 0;
    queue.enqueueWriteBuffer(*_min_max_count, CL_FALSE, 0 * sizeof(zero_count), sizeof(zero_count), &zero_count);
    queue.enqueueWriteBuffer(*_min_max_count, CL_FALSE, 1 * sizeof(zero_count), sizeof(zero_count), &zero_count);

    Window slice = window.first_slice_window_2D();
    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
