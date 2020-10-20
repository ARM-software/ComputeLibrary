/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixMultiplyKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/AccessWindowTranspose.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

#include <set>
#include <string>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
using ElementsProcessed = Steps;

inline Status validate_arguments(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_UNUSED(reshape_info);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input0, input1, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input1->num_dimensions() > 3, "The number of dimensions for the matrix B must be <= 3");

    if(!is_interleaved_transposed)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(0) != input1->dimension(1));

        if(output->total_size() != 0)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(input1->dimension(0) != output->dimension(0));
            ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(1) != output->dimension(1));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, output);
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
        }
    }

    return Status{};
}

inline std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input0, ITensorInfo *input1, ITensorInfo *output,
                                                               bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info,
                                                               GPUTarget gpu_target, ElementsProcessed &num_elements_processed)
{
    ARM_COMPUTE_UNUSED(gpu_target);

    // Output tensor auto inizialitation if not yet initialized
    TensorShape tensor_shape{ input0->tensor_shape() };
    tensor_shape.set(0, is_interleaved_transposed ? reshape_info.n() : input1->dimension(0));
    tensor_shape.set(1, is_interleaved_transposed ? reshape_info.m() : input0->dimension(1));

    auto_init_if_empty(*output, input0->clone()->set_tensor_shape(tensor_shape));

    bool   window_changed = false;
    Window win{};

    const DataType data_type                           = input0->data_type();
    unsigned int &num_elems_processed_per_iteration_x = num_elements_processed[0];
    unsigned int &num_elems_processed_per_iteration_y = num_elements_processed[1];

    if(is_interleaved_transposed)
    {
        // Configure window kernel
        num_elems_processed_per_iteration_x = max_gc_vector_width / data_size_from_type(data_type);
        num_elems_processed_per_iteration_y = 4;

        win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

        AccessWindowRectangle input0_access(input0, 0, 0, num_elems_processed_per_iteration_y, 1, 1.f, 0.25f);
        AccessWindowTranspose input1_access(input1, 0, 0, num_elems_processed_per_iteration_x, 1, 0.f, 0.25f);
        AccessWindowRectangle output_access(output, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

        update_window_and_padding(win, input0_access, input1_access, output_access);

        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }
    else // The input tensors have not been reshaped
    {
        // Special case for 1xN, 2xN, 3xN and 4xN input0 tensor.
        num_elems_processed_per_iteration_y = std::min(static_cast<int>(output->dimension(1)), 4);

        switch(data_type)
        {
            case DataType::F16:
                num_elems_processed_per_iteration_x = 4;
                break;

            case DataType::F32:
                num_elems_processed_per_iteration_x = max_gc_vector_width / data_size_from_type(data_type);
                break;

            default:
                ARM_COMPUTE_ERROR("Current data type is not supported");
                break;
        }

        win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

        AccessWindowStatic    input0_access(input0, 0, 0, ceil_to_multiple(input0->dimension(0), 8), ceil_to_multiple(input0->dimension(1), num_elems_processed_per_iteration_y));
        AccessWindowStatic    input1_access(input1, 0, 0, ceil_to_multiple(input1->dimension(0), num_elems_processed_per_iteration_x), input1->dimension(1));
        AccessWindowRectangle output_access(output, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

        update_window_and_padding(win, input0_access, input1_access, output_access);

        Coordinates coord;
        coord.set_num_dimensions(output->num_dimensions());
        output_access.set_valid_region(win, ValidRegion(coord, output->tensor_shape()));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

GCGEMMMatrixMultiplyKernel::GCGEMMMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr)
{
}

void GCGEMMMatrixMultiplyKernel::configure(const IGCTensor *input0, const IGCTensor *input1, IGCTensor *output, float alpha, bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input0, input1, output);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input0->info(), input1->info(), output->info(), is_interleaved_transposed, reshape_info));

    _input0 = input0;
    _input1 = input1;
    _output = output;

    // Get target architecture
    GPUTarget gpu_target = get_target();

    ElementsProcessed num_elements_processed{};

    // Configure kernel window
    auto win_config = validate_and_configure_window(input0->info(), input1->info(), output->info(), is_interleaved_transposed, reshape_info, gpu_target, num_elements_processed);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    IGCKernel::configure(win_config.second);

    // Create build options
    std::set<std::string> build_opts;
    std::string           kernel_name;

    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    build_opts.emplace("#define COLS_A " + support::cpp11::to_string(input0->info()->dimension(0)));
    build_opts.emplace("#define COLS_B " + support::cpp11::to_string(input1->info()->dimension(0)));
    build_opts.emplace("#define ALPHA " + float_to_string_with_full_precision(alpha));

    // Check if the output tensor is a vector. If so,the kernel runs the vector-matrix multiplication
    if(is_interleaved_transposed)
    {
        const int mult_transpose1xW_width   = reshape_info.mult_transpose1xW_width();
        const int mult_interleave4x4_height = reshape_info.mult_interleave4x4_height();

        build_opts.emplace("#define MULT_TRANSPOSE1XW_WIDTH " + support::cpp11::to_string(mult_transpose1xW_width));
        build_opts.emplace("#define MULT_INTERLEAVE4X4_HEIGHT " + support::cpp11::to_string(mult_interleave4x4_height));

        switch(input0->info()->data_type())
        {
            case DataType::F16:
                build_opts.emplace("#define DATA_TYPE_FP16");
                break;

            case DataType::F32:
                build_opts.emplace("#define DATA_TYPE_FP32");
                break;

            default:
                ARM_COMPUTE_ERROR("Current data type is not supported");
                break;
        }

        build_opts.emplace("#define GEMM_MM_INTERLEAVED_TRANSPOSED");

        kernel_name = "gemm_mm_interleaved_transposed";
    }
    else
    {
        // Special case for 1xN, 2xN, 3xN and 4xN input0 tensor

        GPUTarget arch_target = get_arch_from_target(gpu_target);
        switch(input0->info()->data_type())
        {
            case DataType::F16:
                build_opts.emplace("#define DATA_TYPE_FP16");
                build_opts.emplace("#define MM_PROCESS_4X_OPTIMIZED");
                build_opts.emplace("#define GEMM_MM_FLOATING_POINT");
                break;

            case DataType::F32:
                build_opts.emplace("#define DATA_TYPE_FP32");

                if(arch_target == GPUTarget::BIFROST && input0->info()->num_dimensions() != 1)
                {
                    build_opts.emplace("#define GEMM_MM_FLOATING_POINT_BIFROST");
                }
                else
                {
                    build_opts.emplace("#define GEMM_MM_FLOATING_POINT");
                }
                break;

            default:
                ARM_COMPUTE_ERROR("Current data type is not supported");
                break;
        }

        build_opts.emplace("#define NUM_ELEMS_PROCESSED_PER_THREAD_X " + support::cpp11::to_string(num_elements_processed.x()));
        build_opts.emplace("#define NUM_ELEMS_PROCESSED_PER_THREAD_Y " + support::cpp11::to_string(num_elements_processed.y()));

        kernel_name = "gemm_mm_floating_point";
    }

    // Create kernel
    _kernel = GCKernelLibrary::get().create_kernel(kernel_name, build_opts);
}

Status GCGEMMMatrixMultiplyKernel::validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, float alpha, bool is_interleaved_transposed,
                                            const GEMMReshapeInfo &reshape_info, GPUTarget gpu_target)
{
    ARM_COMPUTE_UNUSED(alpha);
    ElementsProcessed num_elements_processed{};
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

void GCGEMMMatrixMultiplyKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IGCKernel::window(), window);

    _kernel.use();

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

        add_2D_tensor_argument(idx, _input0, 1, slice);
        add_2D_tensor_argument(idx, _input1, 2, slice_b);
        add_2D_tensor_argument(idx, _output, 3, slice);
        _kernel.update_shader_params();
        enqueue(*this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
