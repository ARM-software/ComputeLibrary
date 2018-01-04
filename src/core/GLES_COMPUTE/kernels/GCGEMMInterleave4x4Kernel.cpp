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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMInterleave4x4Kernel.h"

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

using namespace arm_compute;
using namespace arm_compute::gles_compute;

GCGEMMInterleave4x4Kernel::GCGEMMInterleave4x4Kernel()
    : _input(nullptr), _output(nullptr)
{
}

void GCGEMMInterleave4x4Kernel::configure(const IGCTensor *input, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    TensorShape output_shape = input->info()->tensor_shape();
    output_shape.set(0, input->info()->dimension(0) * 4);
    output_shape.set(1, std::ceil(input->info()->dimension(1) / 4.0f));

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input  = input;
    _output = output;

    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace(("#define " + dt_name));
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));

    // Create kernel
    build_opts.emplace("#define GEMM_INTERLEAVE4x4");
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("gemm_interleave4x4", build_opts));

    // Configure kernel window
    const unsigned int     num_elems_processed_per_iteration_x = max_gc_vector_width / data_size_from_type(input->info()->data_type());
    constexpr unsigned int num_elems_processed_per_iteration_y = 4;
    const unsigned int     num_elems_written_per_iteration     = num_elems_processed_per_iteration_x * num_elems_processed_per_iteration_y;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    AccessWindowRectangle input_access(input->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);
    AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_written_per_iteration, 1, 4.f, 0.25f);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    IGCKernel::configure(win);
}

void GCGEMMInterleave4x4Kernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IGCKernel::window(), window);

    _kernel.use();

    /*
     * This kernel puts the values in a 4x4 block of Matrix A on the same row (Interleaved values)
     *         |a00 a01 a02 a03|
     *         |a10 a11 a12 a13|
     *         |a20 a21 a22 a23| = | a00 a10 a20 a30 || a01 a11 a21 a31 || a02 a12 a22 a32 || a03 a13 a23 a33 |
     *         |a30 a31 a32 a33|
     *
     * After this operation, the output matrix will have the following shape: [ height * 4, width / 4 ]
     */
    Window in_slice  = window.first_slice_window_2D();
    Window out_slice = window.first_slice_window_2D();

    // Change x and y steps for the slide of output tensor
    out_slice.scale(Window::DimX, 4.f);
    out_slice.scale(Window::DimY, 0.25f);

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, 1, in_slice);
        add_2D_tensor_argument(idx, _output, 2, out_slice);

        _kernel.update_shader_params();

        enqueue(*this, in_slice);
    }
    while(window.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(out_slice));
}
