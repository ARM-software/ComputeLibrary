/*
 * Copyright (c) 2017, 2018 ARM Limited.
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

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/AccessWindowTranspose.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>

using namespace arm_compute;
using namespace arm_compute::gles_compute;

GCGEMMMatrixMultiplyKernel::GCGEMMMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr)
{
}

void GCGEMMMatrixMultiplyKernel::configure(const IGCTensor *input0, const IGCTensor *input1, IGCTensor *output, float alpha, bool is_interleaved_transposed)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::F32, DataType::F16);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1, output);

    if(!is_interleaved_transposed)
    {
        ARM_COMPUTE_ERROR_ON(input0->info()->dimension(0) != input1->info()->dimension(1));
    }

    _input0 = input0;
    _input1 = input1;
    _output = output;

    std::set<std::string> build_opts;
    Window                win;

    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    build_opts.emplace("#define COLS_A " + support::cpp11::to_string(input0->info()->dimension(0)));
    build_opts.emplace("#define COLS_B " + support::cpp11::to_string(input1->info()->dimension(0)));
    build_opts.emplace("#define ALPHA " + float_to_string_with_full_precision(alpha));

    // Check if the output tensor is a vector. If so,the kernel runs the vector-matrix multiplication
    if(is_interleaved_transposed)
    {
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

        // Create kernel
        _kernel = GCKernelLibrary::get().create_kernel(("gemm_mm_interleaved_transposed"), build_opts);

        // Configure window kernel
        const unsigned int     num_elems_processed_per_iteration_x = max_gc_vector_width / data_size_from_type(input0->info()->data_type());
        constexpr unsigned int num_elems_processed_per_iteration_y = 4;

        win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

        AccessWindowRectangle input0_access(input0->info(), 0, 0, num_elems_processed_per_iteration_y, 1, 1.f, 0.25f);
        AccessWindowTranspose input1_access(input1->info(), 0, 0, num_elems_processed_per_iteration_x, 1, 0.f, 0.25f);
        AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

        update_window_and_padding(win, input0_access, input1_access, output_access);

        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));
    }
    else
    {
        ARM_COMPUTE_ERROR_ON(input0->info()->dimension(0) != input1->info()->dimension(1));

        // Special case for 1xN, 2xN, 3xN and 4xN input0 tensor
        unsigned int num_elems_processed_per_iteration_x;
        unsigned int num_elems_processed_per_iteration_y;

        switch(input0->info()->data_type())
        {
            case DataType::F16:
                build_opts.emplace("#define DATA_TYPE_FP16");

#define MM_PROCESS_4X_OPTIMIZED

#if defined(MM_PROCESS_4X)
                num_elems_processed_per_iteration_x = 4;
                num_elems_processed_per_iteration_y = std::min(static_cast<int>(output->info()->dimension(1)), 4);
                build_opts.emplace("#define MM_PROCESS_4X");
#elif defined(MM_PROCESS_4X_OPTIMIZED) /* MM_PROCESS_4X */
                num_elems_processed_per_iteration_x = 4;
                num_elems_processed_per_iteration_y = std::min(static_cast<int>(output->info()->dimension(1)), 4);
                build_opts.emplace("#define MM_PROCESS_4X_OPTIMIZED");
#elif defined(MM_PROCESS_8X)           /* MM_PROCESS_4X */
                num_elems_processed_per_iteration_x = 8;
                num_elems_processed_per_iteration_y = 1;
                build_opts.emplace("#define MM_PROCESS_8X");
#endif                                 /* MM_PROCESS_4X */
                break;

            case DataType::F32:
                num_elems_processed_per_iteration_x = max_gc_vector_width / data_size_from_type(input0->info()->data_type());
                num_elems_processed_per_iteration_y = std::min(static_cast<int>(output->info()->dimension(1)), 4);
                build_opts.emplace("#define DATA_TYPE_FP32");
                break;

            default:
                ARM_COMPUTE_ERROR("Current data type is not supported");
                break;
        }

        build_opts.emplace("#define GEMM_MM_FLOATING_POINT");
        build_opts.emplace("#define NUM_ELEMS_PROCESSED_PER_THREAD_X " + support::cpp11::to_string(num_elems_processed_per_iteration_x));
        build_opts.emplace("#define NUM_ELEMS_PROCESSED_PER_THREAD_Y " + support::cpp11::to_string(num_elems_processed_per_iteration_y));

        // Create kernel
        _kernel = GCKernelLibrary::get().create_kernel("gemm_mm_floating_point", build_opts);

        win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

#if defined(MM_PROCESS_4X_OPTIMIZED)
        AccessWindowStatic input0_access(input0->info(), 0, 0, ceil_to_multiple(input0->info()->dimension(0), 8), ceil_to_multiple(input0->info()->dimension(1), num_elems_processed_per_iteration_y));
#else  /* MM_PROCESS_4X_OPTIMIZED */
        AccessWindowStatic input0_access(input0->info(), 0, 0, ceil_to_multiple(input0->info()->dimension(0), num_elems_processed_per_iteration_x), ceil_to_multiple(input0->info()->dimension(1),
                                         num_elems_processed_per_iteration_y));
#endif /* MM_PROCESS_4X_OPTIMIZED */
        AccessWindowStatic    input1_access(input1->info(), 0, 0, ceil_to_multiple(input1->info()->dimension(0), num_elems_processed_per_iteration_x), input1->info()->dimension(1));
        AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

        update_window_and_padding(win, input0_access, input1_access, output_access);

        Coordinates coord;
        coord.set_num_dimensions(output->info()->num_dimensions());
        output_access.set_valid_region(win, ValidRegion(coord, output->info()->tensor_shape()));
    }

    IGCKernel::configure(win);
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
