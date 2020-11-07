/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "src/core/CL/kernels/CLConvolutionKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

#include <set>
#include <sstream>
#include <string>

namespace arm_compute
{
namespace
{
constexpr unsigned int max_matrix_size = 81;
} // namespace

/****************************************************************************************\
 *                                 Square Convolution                                *
\****************************************************************************************/

template <unsigned int matrix_size>
BorderSize             CLConvolutionKernel<matrix_size>::border_size() const
{
    return BorderSize(matrix_size / 2);
}

template <unsigned int matrix_size>
void CLConvolutionKernel<matrix_size>::configure(const ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t scale, bool border_undefined)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, conv, scale, border_undefined);
}

template <unsigned int matrix_size>
void CLConvolutionKernel<matrix_size>::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t scale, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON(conv == nullptr);

    _input  = input;
    _output = output;

    std::stringstream kernel_name;
    CLBuildOptions    build_opts;
    kernel_name << "convolution" << matrix_size << "x" << matrix_size << "_static";

    if(scale == 0)
    {
        scale = calculate_matrix_scale(conv, matrix_size);
    }

    for(unsigned int i = 0; i < matrix_size * matrix_size; i++)
    {
        std::stringstream mat_str;
        mat_str << "-DMAT" << i << "=" << conv[i];
        build_opts.add_option(mat_str.str());
    }

    build_opts.add_option("-DSCALE=" + support::cpp11::to_string(scale));

    DataType data_type = data_type_for_convolution_matrix(conv, matrix_size * matrix_size);
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));

    std::stringstream out_type;
    out_type << "-DDATA_TYPE_OUT=" << get_cl_type_from_data_type(output->info()->data_type());
    build_opts.add_option(out_type.str());

    _kernel = create_kernel(compile_context, kernel_name.str(), build_opts.options());

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_rows_read_per_iteration       = matrix_size;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());

    AccessWindowRectangle  input_access(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    ICLKernel::configure_internal(win);
}

/****************************************************************************************\
 *                                 Separable Convolution                                *
\****************************************************************************************/
template <unsigned int matrix_size>
CLSeparableConvolutionHorKernel<matrix_size>::CLSeparableConvolutionHorKernel()
    : _border_size(0)
{
}

template <unsigned int matrix_size>
BorderSize             CLSeparableConvolutionHorKernel<matrix_size>::border_size() const
{
    return _border_size;
}

template <unsigned int matrix_size>
void CLSeparableConvolutionHorKernel<matrix_size>::configure(const ICLTensor *input, ICLTensor *output, const int16_t *conv, bool border_undefined)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, conv, border_undefined);
}

template <unsigned int matrix_size>
void CLSeparableConvolutionHorKernel<matrix_size>::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const int16_t *conv, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U16, DataType::S16, DataType::S32);

    ARM_COMPUTE_ERROR_ON((matrix_size != 5) && (matrix_size != 7) && (matrix_size != 9));

    _input       = input;
    _output      = output;
    _border_size = BorderSize(border_undefined ? 0 : matrix_size / 2, matrix_size / 2);

    // Set build options
    std::set<std::string> build_opts;

    std::array<int16_t, matrix_size *matrix_size> mat = { 0 };
    memcpy(mat.data(), conv, matrix_size * sizeof(int16_t));

    for(unsigned int j = 0; j < matrix_size * matrix_size; j++)
    {
        build_opts.insert("-DMAT" + support::cpp11::to_string(j) + "=" + support::cpp11::to_string(mat[j]));
    }

    build_opts.insert("-DSCALE=0");

    build_opts.insert("-DDATA_TYPE=" + get_cl_type_from_data_type(output->info()->data_type()));

    // Create kernel
    const std::string kernel_name = "convolution_separable1x" + support::cpp11::to_string(matrix_size) + "_static";
    _kernel                       = create_kernel(compile_context, kernel_name, build_opts);

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;

    Window win = calculate_max_window_horizontal(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());

    AccessWindowHorizontal input_access(input->info(), -border_size().left, num_elems_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(border_undefined);
}

template <unsigned int matrix_size>
BorderSize             CLSeparableConvolutionVertKernel<matrix_size>::border_size() const
{
    return BorderSize{ matrix_size / 2, 0 };
}

template <unsigned int matrix_size>
void CLSeparableConvolutionVertKernel<matrix_size>::configure(const ICLTensor *input, ICLTensor *output,
                                                              const int16_t *conv, uint32_t scale, bool border_undefined, DataType data_type)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, conv, scale, border_undefined, data_type);
}

template <unsigned int matrix_size>
void CLSeparableConvolutionVertKernel<matrix_size>::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output,
                                                              const int16_t *conv, uint32_t scale, bool border_undefined, DataType data_type)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U16, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON((matrix_size != 5) && (matrix_size != 7) && (matrix_size != 9));
    ARM_COMPUTE_ERROR_ON(scale == 0);

    _input  = input;
    _output = output;

    std::set<std::string> build_opts;

    std::array<int16_t, matrix_size *matrix_size> mat = { 0 };
    memcpy(mat.data() + matrix_size, conv, matrix_size * sizeof(int16_t));

    for(unsigned int j = 0; j < matrix_size * matrix_size; j++)
    {
        build_opts.insert("-DMAT" + support::cpp11::to_string(j) + "=" + support::cpp11::to_string(mat[j]));
    }

    build_opts.insert("-DSCALE=" + support::cpp11::to_string(scale));

    build_opts.insert("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));

    build_opts.insert("-DCOMPUTE_TYPE=" + get_cl_type_from_data_type(data_type));

    std::stringstream out_type;
    out_type << "-DDATA_TYPE_OUT=" << get_cl_type_from_data_type(output->info()->data_type());
    build_opts.insert(out_type.str());

    // Create kernel
    const std::string kernel_name = "convolution_separable" + support::cpp11::to_string(matrix_size) + "x1_static";
    _kernel                       = create_kernel(compile_context, kernel_name, build_opts);

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 8;
    constexpr unsigned int num_rows_read_per_iteration       = matrix_size;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());

    AccessWindowRectangle  input_access(input->info(), 0, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(data_type));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(border_undefined);
}

/****************************************************************************************\
 *                                 Rectangle Convolution                                *
\****************************************************************************************/

CLConvolutionRectangleKernel::CLConvolutionRectangleKernel()
    : _border_size(0), _input(nullptr), _output(nullptr)
{
}

BorderSize CLConvolutionRectangleKernel::border_size() const
{
    return _border_size;
}

void CLConvolutionRectangleKernel::configure(const ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t width, uint32_t height, uint32_t scale, bool border_undefined)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, conv, width, height, scale, border_undefined);
}

void CLConvolutionRectangleKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t width, uint32_t height, uint32_t scale,
                                             bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON(nullptr == conv);
    ARM_COMPUTE_ERROR_ON(3 != width && 5 != width && 7 != width && 9 != width);
    ARM_COMPUTE_ERROR_ON(3 != height && 5 != height && 7 != height && 9 != height);
    ARM_COMPUTE_ERROR_ON(0 == scale);

    _input       = input;
    _output      = output;
    _border_size = BorderSize(height / 2, width / 2);

    std::set<std::string> options;

    std::stringstream output_type;
    output_type << "-DDATA_TYPE_OUT=" << get_cl_type_from_data_type(output->info()->data_type());
    options.insert(output_type.str());

    uint32_t matrix_size = width * height;

    std::array<int16_t, max_matrix_size> mat = { 0 };

    memcpy(mat.data(), conv, matrix_size * sizeof(int16_t));

    for(unsigned int j = 0; j < max_matrix_size; j++)
    {
        options.insert("-DMAT" + support::cpp11::to_string(j) + "=" + support::cpp11::to_string(mat[j]));
    }

    options.insert("-DSCALE=" + support::cpp11::to_string(scale));

    DataType data_type = data_type_for_convolution_matrix(conv, matrix_size);
    options.insert("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));

    options.insert("-DMATRIX_WIDTH=" + support::cpp11::to_string(width));
    options.insert("-DMATRIX_HEIGHT=" + support::cpp11::to_string(height));

    _kernel = create_kernel(compile_context, "convolution_rectangle", options);

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    const unsigned int     num_rows_read_per_iteration       = height;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());

    AccessWindowRectangle  input_access(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    ICLKernel::configure_internal(win);
}

void CLConvolutionRectangleKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}

template class arm_compute::CLConvolutionKernel<3>;
template class arm_compute::CLConvolutionKernel<5>;
template class arm_compute::CLConvolutionKernel<7>;
template class arm_compute::CLConvolutionKernel<9>;
template class arm_compute::CLSeparableConvolutionVertKernel<5>;
template class arm_compute::CLSeparableConvolutionVertKernel<7>;
template class arm_compute::CLSeparableConvolutionVertKernel<9>;
template class arm_compute::CLSeparableConvolutionHorKernel<5>;
template class arm_compute::CLSeparableConvolutionHorKernel<7>;
template class arm_compute::CLSeparableConvolutionHorKernel<9>;
} // namespace arm_compute
