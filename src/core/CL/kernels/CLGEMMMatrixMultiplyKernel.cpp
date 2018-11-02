/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <set>
#include <string>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
using ElementsProcessed = Steps;

inline Status validate_arguments(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input0, input1, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input0, input1);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input1->num_dimensions() > 3, "The number of dimensions for the matrix B must be <= 3");

    if(!is_interleaved_transposed)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(0) != input1->dimension(1));

        if(output->total_size() != 0)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(input1->dimension(0) != output->dimension(0));
            ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(1) != output->dimension(1));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, output);
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input0, output);
        }
    }
    else
    {
        const int m                         = reshape_info.m();
        const int n                         = reshape_info.n();
        const int k                         = reshape_info.k();
        const int mult_transpose1xW_width   = reshape_info.mult_transpose1xW_width();
        const int mult_interleave4x4_height = reshape_info.mult_interleave4x4_height();

        TensorShape tensor_shape0{ input0->tensor_shape() };
        tensor_shape0.set(0, k);
        tensor_shape0.set(1, m);

        TensorShape tensor_shape1{ input1->tensor_shape() };
        tensor_shape1.set(0, n);
        tensor_shape1.set(1, k);

        const TensorInfo tensor_info0 = input0->clone()->set_tensor_shape(tensor_shape0);
        const TensorInfo tensor_info1 = input1->clone()->set_tensor_shape(tensor_shape1);

        const TensorInfo tensor_info_reshaped0 = input0->clone()->set_tensor_shape(compute_interleaved_shape(tensor_info0, mult_interleave4x4_height));
        const TensorInfo tensor_info_reshaped1 = input1->clone()->set_tensor_shape(compute_transpose1xW_with_element_size_shape(tensor_info1, mult_transpose1xW_width));

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input0, &tensor_info_reshaped0);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input1, &tensor_info_reshaped1);

        if(output->total_size() != 0)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(0) != static_cast<size_t>(n));
            ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(1) != static_cast<size_t>(m));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, output);
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input0, output);
        }
    }

    return Status{};
}

inline std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input0, ITensorInfo *input1, ITensorInfo *output,
                                                               bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info, GPUTarget gpu_target,
                                                               ElementsProcessed &num_elements_processed)
{
    bool   window_changed = false;
    Window win{};

    const DataType data_type                           = input0->data_type();
    unsigned int &num_elems_processed_per_iteration_x = num_elements_processed[0];
    unsigned int &num_elems_processed_per_iteration_y = num_elements_processed[1];

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input0->clone()->set_tensor_shape(compute_mm_shape(*input0, *input1, is_interleaved_transposed, reshape_info)));

    if(is_interleaved_transposed)
    {
        // Configure kernel window
        num_elems_processed_per_iteration_x = max_cl_vector_width / data_size_from_type(data_type);
        num_elems_processed_per_iteration_y = 4;

        win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

        AccessWindowRectangle input0_access(input0, 0, 0, num_elems_processed_per_iteration_y, 1, 1.f, 0.25f);
        AccessWindowStatic    input1_access(input1, 0, 0,
                                            ceil_to_multiple(input1->dimension(0), num_elems_processed_per_iteration_x),
                                            ceil_to_multiple(input1->dimension(1), num_elems_processed_per_iteration_y));
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
        GPUTarget arch_target = get_arch_from_target(gpu_target);
        if(arch_target == GPUTarget::BIFROST && data_type == DataType::F32)
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

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window             collapsed             = win;
    const unsigned int dimension_to_collapse = std::min(static_cast<unsigned int>(output->num_dimensions()), 2u);
    collapsed                                = win.collapse(win, dimension_to_collapse);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, collapsed);
}
} // namespace

CLGEMMMatrixMultiplyKernel::CLGEMMMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr), _slide_matrix_b(true)
{
}

void CLGEMMMatrixMultiplyKernel::configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output, float alpha, bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input0, input1, output);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input0->info(), input1->info(), output->info(), is_interleaved_transposed, reshape_info));

    _input0         = input0;
    _input1         = input1;
    _output         = output;
    _slide_matrix_b = _input1->info()->num_dimensions() >= _input0->info()->num_dimensions();

    const DataType data_type = input0->info()->data_type();
    const int      fp_pos    = input0->info()->fixed_point_position();

    // Get target architecture
    GPUTarget gpu_target = get_target();

    // Configure LWS hint
    switch(gpu_target)
    {
        case GPUTarget::MIDGARD:
        case GPUTarget::T600:
        case GPUTarget::T700:
        case GPUTarget::T800:
            if(output->info()->dimension(1) == 196)
            {
                _lws_hint = cl::NDRange(1, 7);
            }
            else
            {
                _lws_hint = cl::NDRange(8, 8);
            }
            break;
        case GPUTarget::G71:
        case GPUTarget::G72:
        case GPUTarget::G51:
        case GPUTarget::G51BIG:
        case GPUTarget::G51LIT:
        case GPUTarget::TNOX:
            if(input1->info()->dimension(1) == 24)
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
            break;
        default:
            _lws_hint = cl::NullRange;
    }

    ElementsProcessed num_elements_processed{};

    // Configure kernel window
    auto win_config = validate_and_configure_window(input0->info(), input1->info(), output->info(), is_interleaved_transposed, reshape_info, gpu_target, num_elements_processed);
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

    // Do not slide matrix B if _slide_matrix_b = false
    build_opts.add_option_if(!_slide_matrix_b, "-DMATRIX_B_DEPTH=" + support::cpp11::to_string(input1->info()->dimension(2)));

    const bool is_bifrost = get_arch_from_target(gpu_target) == GPUTarget::BIFROST;

    std::string kernel_name;
    if(is_interleaved_transposed)
    {
        const int mult_transpose1xW_width   = reshape_info.mult_transpose1xW_width();
        const int mult_interleave4x4_height = reshape_info.mult_interleave4x4_height();

        build_opts.add_option("-DCOLS_B=" + support::cpp11::to_string(input1->info()->dimension(0)));
        build_opts.add_option("-DMULT_TRANSPOSE1XW_WIDTH=" + support::cpp11::to_string(mult_transpose1xW_width));
        build_opts.add_option("-DMULT_INTERLEAVE4X4_HEIGHT=" + support::cpp11::to_string(mult_interleave4x4_height));

        if(is_data_type_float(data_type) && is_bifrost)
        {
            kernel_name = "gemm_mm_interleaved_transposed_" + lower_string(string_from_data_type(data_type)) + "_bifrost";
        }
        else
        {
            kernel_name = "gemm_mm_interleaved_transposed_" + lower_string(string_from_data_type(data_type));
        }
    }
    else // The input tensors have not been reshaped
    {
        build_opts.add_option("-DCOLS_A=" + support::cpp11::to_string(input0->info()->dimension(0)));
        build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));

        // Create kernels according to the architecture, data type and input size.
        if(is_data_type_float(data_type) && is_bifrost)
        {
            kernel_name = "gemm_mm_floating_point";

            if(input0->info()->num_dimensions() != 1)
            {
                kernel_name += "_" + lower_string(string_from_data_type(data_type)) + "_bifrost";
            }
            else if(input1->info()->dimension(0) <= 1000 && data_type == DataType::F32)
            {
                // The first kernel is optimized for the case of 1000 or less output elements (e.g. FC8 of AlexNet and VGG-16, and
                // FC1 of Inception v3). The second kernel is optimized for the case of greater than 1000 output elements (e.g.
                // FC6 and FC7 of AlexNet and VGG-16).
                kernel_name += "_" + lower_string(string_from_data_type(data_type)) + "_bifrost_1000";
            }

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
    _config_id += support::cpp11::to_string(output->info()->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(3));
    _config_id += "_";
    _config_id += (is_interleaved_transposed ? support::cpp11::to_string(input1->info()->dimension(0)) : support::cpp11::to_string(input1->info()->dimension(1)));
}

Status CLGEMMMatrixMultiplyKernel::validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, float alpha, bool is_interleaved_transposed,
                                            const GEMMReshapeInfo &reshape_info, GPUTarget gpu_target)
{
    // Note: num_elements_processed will be set in validate_and_configure_window()
    ElementsProcessed num_elements_processed{};
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input0, input1, output, is_interleaved_transposed, reshape_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input0->clone().get(),
                                                              input1->clone().get(),
                                                              output->clone().get(),
                                                              is_interleaved_transposed,
                                                              reshape_info,
                                                              gpu_target,
                                                              num_elements_processed)
                                .first);

    return Status{};
}

void CLGEMMMatrixMultiplyKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    if(_input1->info()->num_dimensions() < 3)
    {
        // The stride_z for matrix B must be zero if we do not slice
        ARM_COMPUTE_ERROR_ON(_input1->info()->strides_in_bytes()[3] != 0);
    }

    Window slice          = window.first_slice_window_3D();
    Window slice_matrix_b = slice;

    slice_matrix_b.set(Window::DimX, Window::Dimension(0, 1, 1));
    slice_matrix_b.set(Window::DimY, Window::Dimension(0, 1, 1));

    do
    {
        Window slice_b = slice;
        // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
        // This scenario can happen when the matrix multiplication is used to perform a convolution operation
        if(!_slide_matrix_b)
        {
            slice_b = slice_matrix_b;
        }

        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input0, slice);
        add_2D_tensor_argument(idx, _input1, slice_b);
        add_2D_tensor_argument(idx, _output, slice);
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_input0->info()->strides_in_bytes()[2]));
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_input1->info()->strides_in_bytes()[2]));
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_output->info()->strides_in_bytes()[2]));
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window.slide_window_slice_3D(slice));
}
