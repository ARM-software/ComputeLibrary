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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCSoftmaxLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>

using namespace arm_compute;

void GCLogits1DMaxKernel::configure(const IGCTensor *input, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    // Softmax across the x dimension
    TensorShape output_shape{ input->info()->tensor_shape() };
    output_shape.set(0, 1);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);

    _input  = input;
    _output = output;

    // Set build options
    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.insert("#define " + dt_name);
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    build_opts.insert("#define SOFTMAX_LAYER_MAX");

    // Tell the kernel that the width is not a multiple of 8
    if((input->info()->dimension(0) % 8) != 0)
    {
        build_opts.insert("#define NON_MULTIPLE_OF_8");
    }

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("softmax_layer_max", build_opts));

    // Set fixed arguments
    unsigned int idx = 2 * num_arguments_per_3D_tensor(); //Skip the input and output parameters
    _kernel.set_argument(idx++, input->info()->dimension(0));

    // Configure kernel window
    // The kernel loops over all elements in steps of 8
    const unsigned int num_elems_processed_per_iteration = ceil_to_multiple(input->info()->dimension(0), 8);
    unsigned int       num_elems_written_per_iteration   = 1;
    if(input->info()->data_type() == DataType::F16)
    {
        num_elems_written_per_iteration = 2;
    }

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    IGCKernel::configure(win);
}

GCLogits1DShiftExpSumKernel::GCLogits1DShiftExpSumKernel()
    : _input(nullptr), _max(nullptr), _output(nullptr), _sum(nullptr)
{
}

void GCLogits1DShiftExpSumKernel::configure(const IGCTensor *input, const IGCTensor *max, IGCTensor *output, IGCTensor *sum)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(max, sum, output);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*sum->info(), max->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, max, sum);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(max, sum);

    _input  = input;
    _max    = max;
    _output = output;
    _sum    = sum;

    // Set build options
    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.insert("#define " + dt_name);
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    build_opts.insert("#define SOFTMAX_LAYER_SHIFT_EXP_SUM");

    // Tell the kernel that the width is not a multiple of 8
    if((input->info()->dimension(0) % 8) != 0)
    {
        build_opts.insert("#define NON_MULTIPLE_OF_8");
    }

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("softmax_layer_shift_exp_sum", build_opts));

    // Set fixed arguments
    unsigned int idx = 4 * num_arguments_per_3D_tensor(); //Skip the input and output parameters
    _kernel.set_argument(idx++, input->info()->dimension(0));

    // Configure window
    // The kernel loops over all elements in steps of 8
    const unsigned int num_elems_processed_per_iteration = ceil_to_multiple(input->info()->dimension(0), 8);
    unsigned int       num_elems_written_per_iteration   = 1;
    if(input->info()->data_type() == DataType::F16)
    {
        num_elems_written_per_iteration = 2;
    }

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal max_access(max->info(), 0, num_elems_written_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal sum_access(sum->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win, input_access, max_access, output_access, sum_access);

    output_access.set_valid_region(win, input->info()->valid_region());
    sum_access.set_valid_region(win, ValidRegion(Coordinates(), sum->info()->tensor_shape()));

    IGCKernel::configure(win);
}

void GCLogits1DShiftExpSumKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(IGCKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    _kernel.use();

    do
    {
        unsigned int idx     = 0;
        unsigned int binding = 1; // SSBO binding starts from 1.
        // Set inputs
        add_3D_tensor_argument(idx, _input, binding++, slice);
        add_3D_tensor_argument(idx, _max, binding++, slice);
        add_3D_tensor_argument(idx, _output, binding++, slice);
        add_3D_tensor_argument(idx, _sum, binding++, slice);
        _kernel.update_shader_params();
        enqueue(*this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}

GCLogits1DNormKernel::GCLogits1DNormKernel()
    : _input(nullptr), _sum(nullptr), _output(nullptr)
{
}

void GCLogits1DNormKernel::configure(const IGCTensor *input, const IGCTensor *sum, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(sum, output);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, sum, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, sum, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);

    _input  = input;
    _sum    = sum;
    _output = output;

    // Set build options
    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.insert("#define " + dt_name);
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    build_opts.insert("#define SOFTMAX_LAYER_NORM");

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("softmax_layer_norm", build_opts));

    // Configure window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    unsigned int           num_elems_written_per_iteration   = 1;
    if(input->info()->data_type() == DataType::F16)
    {
        num_elems_written_per_iteration = 2;
    }

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowStatic     sum_access(sum->info(), 0, 0, num_elems_written_per_iteration, sum->info()->dimension(1));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, sum_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    IGCKernel::configure(win);
}

void GCLogits1DNormKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(IGCKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    _kernel.use();

    do
    {
        Window sum_slice = slice;
        sum_slice.set(Window::DimX, Window::Dimension(0, 1, 1));

        unsigned int idx     = 0;
        unsigned int binding = 1; // SSBO binding starts from 1.
        // Set inputs
        add_3D_tensor_argument(idx, _input, binding++, slice);
        add_3D_tensor_argument(idx, _sum, binding++, slice);
        add_3D_tensor_argument(idx, _output, binding++, slice);

        _kernel.update_shader_params();
        enqueue(*this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
