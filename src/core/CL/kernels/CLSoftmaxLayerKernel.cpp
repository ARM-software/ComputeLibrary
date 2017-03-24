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
#include "arm_compute/core/CL/kernels/CLSoftmaxLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>

using namespace arm_compute;

void CLLogits1DMaxKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input                                = input;
    _output                               = output;
    const unsigned int processed_elements = input->info()->dimension(0);

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DUSE_" + string_from_data_type(input->info()->data_type())));
    build_opts.emplace(((processed_elements % max_cl_vector_width) != 0) ? "-DNON_MULTIPLE_OF_16" : "");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("softmax_layer_max", build_opts));

    // Set fixed arguments
    unsigned int idx = 2 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg<cl_uint>(idx++, processed_elements);

    // Configure window
    constexpr unsigned int written_elements(1);
    Window                 win = calculate_max_window(*input->info(), Steps(processed_elements));
    AccessWindowHorizontal input_access(input->info(), 0, processed_elements);
    AccessWindowHorizontal output_access(output->info(), 0, written_elements);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));
    ICLKernel::configure(win);
}

CLLogits1DShiftExpSumKernel::CLLogits1DShiftExpSumKernel()
    : _input(nullptr), _max(nullptr), _output(nullptr), _sum(nullptr)
{
}

void CLLogits1DShiftExpSumKernel::configure(const ICLTensor *input, const ICLTensor *max, ICLTensor *output, ICLTensor *sum)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(max, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(sum, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, max, sum);

    _input                                = input;
    _max                                  = max;
    _output                               = output;
    _sum                                  = sum;
    const unsigned int processed_elements = input->info()->dimension(0);

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DUSE_" + string_from_data_type(input->info()->data_type())));
    build_opts.emplace(((processed_elements % max_cl_vector_width) != 0) ? "-DNON_MULTIPLE_OF_16" : "");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("softmax_layer_shift_exp_sum", build_opts));

    // Set fixed arguments
    unsigned int idx = 4 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg<cl_uint>(idx++, processed_elements);

    // Configure window
    Window                 win = calculate_max_window(*input->info(), Steps(processed_elements));
    AccessWindowHorizontal input_access(input->info(), 0, processed_elements);
    AccessWindowHorizontal max_access(max->info(), 0, 1);
    AccessWindowHorizontal output_access(output->info(), 0, processed_elements);
    AccessWindowHorizontal sum_access(sum->info(), 0, 1);
    update_window_and_padding(win, input_access, max_access, output_access, sum_access);
    output_access.set_valid_region(win, input->info()->valid_region());
    sum_access.set_valid_region(win, ValidRegion(Coordinates(), sum->info()->tensor_shape()));
    ICLKernel::configure(win);
}

void CLLogits1DShiftExpSumKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        // Set inputs
        add_2D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _max, slice);
        add_2D_tensor_argument(idx, _output, slice);
        add_2D_tensor_argument(idx, _sum, slice);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}

CLLogits1DNormKernel::CLLogits1DNormKernel()
    : _input(nullptr), _sum(nullptr), _output(nullptr)
{
}

void CLLogits1DNormKernel::configure(const ICLTensor *input, const ICLTensor *sum, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(sum, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, sum);

    _input  = input;
    _sum    = sum;
    _output = output;

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DUSE_" + string_from_data_type(input->info()->data_type())));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("softmax_layer_norm", build_opts));

    // Configure window
    constexpr unsigned int processed_elements(16);
    Window                 win = calculate_max_window(*input->info(), Steps(processed_elements));
    AccessWindowHorizontal input_access(input->info(), 0, processed_elements);
    AccessWindowStatic     sum_access(sum->info(), 0, 0, 1, sum->info()->dimension(1));
    AccessWindowHorizontal output_access(output->info(), 0, processed_elements);
    update_window_and_padding(win, input_access, sum_access, output_access);
    output_access.set_valid_region(win, input->info()->valid_region());
    ICLKernel::configure(win);
}

void CLLogits1DNormKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        // Set inputs
        add_2D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _sum, slice);
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
