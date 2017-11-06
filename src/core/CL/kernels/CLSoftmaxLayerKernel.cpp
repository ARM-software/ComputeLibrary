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
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    // Softmax across the x dimension
    TensorShape output_shape{ input->info()->tensor_shape() };
    output_shape.set(0, 1);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);

    _input  = input;
    _output = output;

    // The kernel loops over all elements in steps of 16
    const unsigned int num_elems_processed_per_iteration = ceil_to_multiple(input->info()->dimension(0), 16);

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    if(is_data_type_fixed_point(input->info()->data_type()))
    {
        build_opts.emplace(("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position())));
    }
    else if(input->info()->data_type() == DataType::F16)
    {
        build_opts.emplace("-DUSE_F16");
    }

    // Tell the kernel that the width is not a multiple of 16
    if((input->info()->dimension(0) % max_cl_vector_width) != 0)
    {
        build_opts.emplace("-DNON_MULTIPLE_OF_16");
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("softmax_layer_max", build_opts));

    // Set fixed arguments
    unsigned int idx = 2 * num_arguments_per_3D_tensor(); //Skip the input and output parameters
    _kernel.setArg<cl_uint>(idx++, input->info()->dimension(0));

    // Configure kernel window
    constexpr unsigned int num_elems_written_per_iteration = 1;

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

CLLogits1DShiftExpSumKernel::CLLogits1DShiftExpSumKernel()
    : _input(nullptr), _max(nullptr), _output(nullptr), _sum(nullptr)
{
}

void CLLogits1DShiftExpSumKernel::configure(const ICLTensor *input, const ICLTensor *max, ICLTensor *output, ICLTensor *sum, float beta)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(max, sum, output);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*sum->info(), max->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, max, sum);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output, max, sum);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(max, sum);

    _input  = input;
    _max    = max;
    _output = output;
    _sum    = sum;

    const DataType dt       = input->info()->data_type();
    auto           beta_int = static_cast<int>(lround(beta * (1 << input->info()->fixed_point_position())));

    // The kernel loops over all elements in steps of 16
    const unsigned int num_elems_processed_per_iteration = ceil_to_multiple(input->info()->dimension(0), 16);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option(std::string("-DDATA_TYPE=" + get_cl_type_from_data_type(dt)));
    build_opts.add_option_if(is_data_type_fixed_point(dt),
                             std::string("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position())));
    build_opts.add_option_if(dt == DataType::F16, std::string("-DUSE_F16"));
    // Tell the kernel that the width is not a multiple of 16
    build_opts.add_option_if((input->info()->dimension(0) % max_cl_vector_width) != 0, std::string("-DNON_MULTIPLE_OF_16"));
    build_opts.add_option_if(is_data_type_fixed_point(dt) && (beta != 1.0f), std::string("-DBETA=" + support::cpp11::to_string(beta_int)));
    build_opts.add_option_if(is_data_type_float(dt) && (beta != 1.0f), std::string("-DBETA=" + float_to_string_with_full_precision(beta)));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("softmax_layer_shift_exp_sum", build_opts.options()));

    // Set fixed arguments
    unsigned int idx = 4 * num_arguments_per_3D_tensor(); //Skip the input and output parameters
    _kernel.setArg<cl_uint>(idx++, input->info()->dimension(0));

    // Configure window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal max_access(max->info(), 0, 1);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
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

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        // Set inputs
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _max, slice);
        add_3D_tensor_argument(idx, _output, slice);
        add_3D_tensor_argument(idx, _sum, slice);
        enqueue(queue, *this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}

/**< Grid size (obtained through auto-tuning) */
const unsigned int CLLogits1DMaxShiftExpSumKernel::_grid_size = 64;
/**< Vector size in the serial case (obtained through auto-tuning) */
const unsigned int CLLogits1DMaxShiftExpSumKernel::_serial_vector_size = 8;
/**< Vector size in the parallel case (obtained through auto-tuning, enables the best memory access pattern for Bifrost) .*/
const unsigned int CLLogits1DMaxShiftExpSumKernel::_parallel_vector_size = 4;

CLLogits1DMaxShiftExpSumKernel::CLLogits1DMaxShiftExpSumKernel()
    : _input(nullptr), _max(nullptr), _output(nullptr), _sum(nullptr)
{
}

void CLLogits1DMaxShiftExpSumKernel::configure(const ICLTensor *input, ICLTensor *max, ICLTensor *output, ICLTensor *sum, float beta)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(max, sum, output);
    ARM_COMPUTE_ERROR_ON(beta != 1.0f && input->info()->data_type() != DataType::F32);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*sum->info(), max->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, max, sum);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output, max, sum);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(max, sum);

    _input  = input;
    _max    = max;
    _output = output;
    _sum    = sum;

    const DataType dt                 = input->info()->data_type();
    const size_t   reduction_dim_size = input->info()->dimension(0);
    auto           beta_int           = static_cast<int>(lround(beta * (1 << input->info()->fixed_point_position())));

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(dt));
    build_opts.add_option_if(is_data_type_fixed_point(dt),
                             "-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));
    build_opts.add_option_if(dt == DataType::F16, "-DUSE_F16");
    build_opts.add_option_if(is_data_type_fixed_point(dt) && (beta != 1.0f), "-DBETA=" + support::cpp11::to_string(beta_int));
    build_opts.add_option_if(is_data_type_float(dt) && (beta != 1.0f), "-DBETA=" + float_to_string_with_full_precision(beta));

    // Setting _lws_hint in this way can also communicate grid_size to CLLogits1DMaxShiftExpSumKernel::run().
    // A single workgroup performs reduction in dimension 0 in the parallel case, hence lws[0]==gws[0].
    _lws_hint                                     = cl::NullRange;
    std::string           kernel_name             = std::string("softmax_layer_max_shift_exp_sum_serial");
    ParallelReductionInfo parallel_reduction_info = is_parallel_reduction(reduction_dim_size);
    unsigned int          vector_size             = std::get<1>(parallel_reduction_info);

    build_opts.add_option("-DVECTOR_SIZE=" + support::cpp11::to_string(vector_size));
    build_opts.add_option("-DLOG_VECTOR_SIZE=" + support::cpp11::to_string(lround(log2(vector_size))));
    build_opts.add_option_if((reduction_dim_size % vector_size) != 0, "-DNON_MULTIPLE_OF_VECTOR_SIZE");

    // Configure parallel kernel if needed
    if(std::get<0>(parallel_reduction_info))
    {
        kernel_name            = std::string("softmax_layer_max_shift_exp_sum_parallel");
        bool is_grid_size_pow2 = (_grid_size != 0) && ((_grid_size & (_grid_size - 1)) == 0);
        build_opts.add_option_if(is_grid_size_pow2 && _grid_size <= 256, "-DGRID_SIZE=" + support::cpp11::to_string(_grid_size));

        // Handle boundary conditions.
        const unsigned int multiple_grid_size = (reduction_dim_size / vector_size) % _grid_size;
        build_opts.add_option_if((multiple_grid_size != 0) || ((reduction_dim_size % vector_size) != 0), "-DNON_MULTIPLE_OF_GRID_SIZE");
    }

    // Create kernel.
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Set static arguments. Both the kernels use the same arguments
    unsigned int idx = 4 * num_arguments_per_3D_tensor(); //Skip the input and output parameters
    _kernel.setArg<cl_uint>(idx++, reduction_dim_size);

    // Configure window
    const unsigned int num_elems_x = ceil_to_multiple(input->info()->tensor_shape().x(), vector_size);
    Window             win         = calculate_max_window(*input->info(), Steps(num_elems_x));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_x);
    AccessWindowHorizontal max_access(max->info(), 0, 1);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_x);
    AccessWindowHorizontal sum_access(sum->info(), 0, 1);

    update_window_and_padding(win, input_access, max_access, output_access, sum_access);

    output_access.set_valid_region(win, input->info()->valid_region());
    sum_access.set_valid_region(win, ValidRegion(Coordinates(), sum->info()->tensor_shape()));

    ICLKernel::configure(win);
}

CLLogits1DMaxShiftExpSumKernel::ParallelReductionInfo CLLogits1DMaxShiftExpSumKernel::is_parallel_reduction(size_t size)
{
    bool         is_parallel_reduction = (size >= (_grid_size * _serial_vector_size)) && (_grid_size > 1);
    unsigned int vector_size           = is_parallel_reduction ? _parallel_vector_size : _serial_vector_size;
    return std::make_tuple(is_parallel_reduction, vector_size);
}

void CLLogits1DMaxShiftExpSumKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Collapse window in Z dimension
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);

    // Reconfigure window in case of parallel reduction
    ParallelReductionInfo parallel_reduction_info = is_parallel_reduction(_input->info()->dimension(0));
    if(std::get<0>(parallel_reduction_info))
    {
        // To launch grid_size parallel workitems, steps.x should be modified as follows.
        const unsigned int step = std::get<1>(parallel_reduction_info);
        window_collapsed.set(Window::DimX, Window::Dimension(0, _grid_size * step, step));
    }

    // Get slices
    Window slice = window_collapsed.first_slice_window_3D();
    do
    {
        unsigned int idx = 0;
        // Set inputs
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _max, slice);
        add_3D_tensor_argument(idx, _output, slice);
        add_3D_tensor_argument(idx, _sum, slice);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}

CLLogits1DNormKernel::CLLogits1DNormKernel()
    : _input(nullptr), _sum(nullptr), _output(nullptr)
{
}

void CLLogits1DNormKernel::configure(const ICLTensor *input, const ICLTensor *sum, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
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
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    if(is_data_type_fixed_point(input->info()->data_type()))
    {
        build_opts.emplace(("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position())));
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("softmax_layer_norm", build_opts));

    // Configure window
    constexpr unsigned int num_elems_processed_per_iteration = 16;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowStatic     sum_access(sum->info(), 0, 0, 1, sum->info()->dimension(1));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, sum_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    ICLKernel::configure(win);
}

void CLLogits1DNormKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    do
    {
        Window sum_slice = slice;
        sum_slice.set(Window::DimX, Window::Dimension(0, 1, 1));

        unsigned int idx = 0;
        // Set inputs
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _sum, sum_slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
