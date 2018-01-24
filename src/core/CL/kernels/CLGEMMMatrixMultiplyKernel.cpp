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
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/AccessWindowTranspose.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>

using namespace arm_compute;

namespace
{
using ElementsProcessed = Steps;

inline Status validate_arguments(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, bool is_interleaved_transposed)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input0, input1, output);
    if(!is_interleaved_transposed)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(0) != input1->dimension(1));
    }

    return Status{};
}

inline std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input0, ITensorInfo *input1, ITensorInfo *output,
                                                               bool is_interleaved_transposed, GPUTarget gpu_target,
                                                               ElementsProcessed &num_elements_processed)
{
    bool   window_changed = false;
    Window win{};

    const DataType data_type                           = input0->data_type();
    unsigned int &num_elems_processed_per_iteration_x = num_elements_processed[0];
    unsigned int &num_elems_processed_per_iteration_y = num_elements_processed[1];

    if(is_interleaved_transposed)
    {
        // Configure kernel window
        num_elems_processed_per_iteration_x = max_cl_vector_width / data_size_from_type(data_type);
        num_elems_processed_per_iteration_y = 4;

        win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

        AccessWindowRectangle input0_access(input0, 0, 0, num_elems_processed_per_iteration_y, 1, 1.f, 0.25f);
        AccessWindowTranspose input1_access(input1, 0, 0, num_elems_processed_per_iteration_x, 1, 0.f, 0.25f);
        AccessWindowRectangle output_access(output, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

        window_changed = update_window_and_padding(win, input0_access, input1_access, output_access);

        output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->tensor_shape()));
    }
    else // The input tensors have not been reshaped
    {
        // Special case for 1xN, 2xN, 3xN and 4xN input0 tensor. num_elems_processed_per_iteration_x is set up for the default case.
        num_elems_processed_per_iteration_x = max_cl_vector_width / data_size_from_type(data_type);
        num_elems_processed_per_iteration_y = std::min(static_cast<int>(output->dimension(1)), 4);

        // Create kernels according to the architecture, data type and input size.
        if(gpu_target == GPUTarget::BIFROST && data_type == DataType::F32)
        {
            num_elems_processed_per_iteration_x = (input1->dimension(0) <= 1000 && input0->num_dimensions() == 1) ? 2 : 4;
        }

        // Configure window
        win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

        AccessWindowStatic    input0_access(input0, 0, 0, input0->dimension(0), ceil_to_multiple(input0->dimension(1), num_elems_processed_per_iteration_y));
        AccessWindowStatic    input1_access(input1, 0, 0, ceil_to_multiple(input1->dimension(0), num_elems_processed_per_iteration_x), input1->dimension(1));
        AccessWindowRectangle output_access(output, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

        window_changed = update_window_and_padding(win, input0_access, input1_access, output_access);

        Coordinates coord;
        coord.set_num_dimensions(output->num_dimensions());
        output_access.set_valid_region(win, ValidRegion(coord, output->tensor_shape()));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLGEMMMatrixMultiplyKernel::CLGEMMMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr)
{
}

void CLGEMMMatrixMultiplyKernel::configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output, float alpha, bool is_interleaved_transposed)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input0, input1, output);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input0->info(), input1->info(), output->info(), is_interleaved_transposed));

    _input0 = input0;
    _input1 = input1;
    _output = output;

    const DataType data_type = input0->info()->data_type();
    const int      fp_pos    = input0->info()->fixed_point_position();

    // Get target architecture
    GPUTarget arch_target = get_arch_from_target(get_target());

    // Configure LWS hint
    if(arch_target == GPUTarget::BIFROST && input1->info()->dimension(1) == 24)
    {
        // LWS optimized for the 11x11 AlexNet convolution on Bifrost.
        _lws_hint = cl::NDRange(2, 2);
    }
    else if(output->info()->dimension(1) == 196)
    {
        _lws_hint = cl::NDRange(1, 7);
    }
    else
    {
        _lws_hint = cl::NDRange(8, 8);
    }

    ElementsProcessed num_elements_processed{};

    // Configure kernel window
    auto win_config = validate_and_configure_window(input0->info(), input1->info(), output->info(), is_interleaved_transposed, arch_target, num_elements_processed);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option_if(is_data_type_fixed_point(data_type), "-DFIXED_POINT_POSITION=" + support::cpp11::to_string(fp_pos));

    // Only define ALPHA when alpha is not 1.0f. This avoids performing unnecessary multiplications.
    if(std::abs(1.0f - alpha) > 0.00001f)
    {
        build_opts.add_option_if_else(is_data_type_fixed_point(data_type),
                                      "-DALPHA=" + support::cpp11::to_string((data_type == DataType::QS8 ? sqcvt_qs8_f32(alpha, fp_pos) : sqcvt_qs16_f32(alpha, fp_pos))),
                                      "-DALPHA=" + float_to_string_with_full_precision(alpha));
    }

    std::string kernel_name;
    if(is_interleaved_transposed)
    {
        build_opts.add_option("-DCOLS_B=" + support::cpp11::to_string(input1->info()->dimension(0)));
        if(data_type == DataType::F32)
        {
            kernel_name = "gemm_mm_interleaved_transposed_f32_" + string_from_target(arch_target);
        }
        else
        {
            kernel_name = "gemm_mm_interleaved_transposed_" + lower_string(string_from_data_type(data_type));
        }
    }
    else // The input tensors have not been reshaped
    {
        build_opts.add_option("-DCOLS_A=" + support::cpp11::to_string(input0->info()->dimension(0)));

        // Create kernels according to the architecture, data type and input size.
        if(arch_target == GPUTarget::BIFROST && data_type == DataType::F32)
        {
            // The first kernel is optimized for the case of 1000 or less output elements (e.g. FC8 of AlexNet and VGG-16, and
            // FC1 of Inception v3). The second kernel is optimized for the case of greater than 1000 output elements (e.g.
            // FC6 and FC7 of AlexNet and VGG-16).
            kernel_name = (input1->info()->dimension(0) <= 1000 && input0->info()->num_dimensions() == 1) ? "gemm_mm_floating_point_f32_bifrost_1000" : "gemm_mm_floating_point_f32_bifrost";

            // The work-group size equal to the Bifrost quad size has been proved to be optimal for these kernels
            // via exhaustive autotuning over a range of representative layer configurations.
            _lws_hint = cl::NDRange(4);
        }
        else if(is_data_type_fixed_point(data_type))
        {
            kernel_name = "gemm_mm_" + lower_string(string_from_data_type(data_type));
        }
        else // (MIDGARD and F32) or (F16)
        {
            build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
            kernel_name = "gemm_mm_floating_point";
        }
        build_opts.add_option("-DNUM_ELEMS_PROCESSED_PER_THREAD_Y=" + support::cpp11::to_string(num_elements_processed.y()));
        build_opts.add_option("-DNUM_ELEMS_PROCESSED_PER_THREAD_X=" + support::cpp11::to_string(num_elements_processed.x()));
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Set config_id for enabling LWS tuning
    _config_id = "gemm_";
    _config_id += (is_interleaved_transposed ? "reshaped_" : "");
    _config_id += lower_string(string_from_data_type(input0->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += (is_interleaved_transposed ? support::cpp11::to_string(input1->info()->dimension(0)) : support::cpp11::to_string(input1->info()->dimension(1)));
}

Status CLGEMMMatrixMultiplyKernel::validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, float alpha, bool is_interleaved_transposed, GPUTarget gpu_target)
{
    ElementsProcessed num_elements_processed{};
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input0, input1, output, is_interleaved_transposed));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input0->clone().get(),
                                                              input1->clone().get(),
                                                              output->clone().get(),
                                                              is_interleaved_transposed,
                                                              gpu_target,
                                                              num_elements_processed)
                                .first);

    return Status{};
}

void CLGEMMMatrixMultiplyKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice          = window.first_slice_window_2D();
    Window slice_matrix_b = slice;

    slice_matrix_b.set(Window::DimX, Window::Dimension(0, 1, 1));
    slice_matrix_b.set(Window::DimY, Window::Dimension(0, 1, 1));

    do
    {
        Window slice_b = slice;
        // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
        // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
        if(_input1->info()->num_dimensions() < 3)
        {
            slice_b = slice_matrix_b;
        }

        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input0, slice);
        add_2D_tensor_argument(idx, _input1, slice_b);
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window.slide_window_slice_2D(slice));
}
