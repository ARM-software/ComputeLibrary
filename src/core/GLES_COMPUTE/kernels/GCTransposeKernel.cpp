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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCTransposeKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"

#include <set>
#include <string>

using namespace arm_compute;

void GCTransposeKernel::configure(const IGCTensor *input, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    TensorShape  output_shape{ input->info()->tensor_shape() };
    const size_t w_out = input->info()->dimension(1);
    const size_t h_out = input->info()->dimension(0);
    output_shape.set(0, w_out);
    output_shape.set(1, h_out);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input  = input;
    _output = output;

    // for better performance
    if(w_out < 512 && h_out < 512)
    {
        _lws_hint = gles::NDRange(8U, 1U, 1U);
    }
    else
    {
        _lws_hint = gles::NDRange(1U, 8U, 1U);
    }

    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace(("#define " + dt_name));
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(_lws_hint[0]));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(_lws_hint[1]));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(_lws_hint[2]));

    // Configure kernel window
    unsigned int num_elems_processed_per_iteration = 4;

    if(input->info()->data_type() == DataType::F16)
    {
#define TRANSPOSE_8X8

#if defined(TRANSPOSE_4X4)
        build_opts.emplace(("#define TRANSPOSE_4X4"));
        num_elems_processed_per_iteration = 4;
#elif defined(TRANSPOSE_8X8) /* TRANSPOSE_4X4 */
        if(w_out != h_out)
        {
            build_opts.emplace("#define TRANSPOSE_8X8");
            num_elems_processed_per_iteration = 8;
        }
        else
        {
            build_opts.emplace("#define TRANSPOSE_8X8_SQUARE");
            num_elems_processed_per_iteration = 8;
        }
#endif                       /* TRANSPOSE_4X4 */
    }

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("transpose", build_opts));

    const unsigned int width_aligned  = num_elems_processed_per_iteration * static_cast<unsigned int>(_lws_hint[0]);
    const unsigned int height_aligned = num_elems_processed_per_iteration * static_cast<unsigned int>(_lws_hint[1]);

    AccessWindowStatic input_access(input->info(), 0, 0,
                                    ceil_to_multiple(input->info()->dimension(0), width_aligned),
                                    ceil_to_multiple(input->info()->dimension(1), height_aligned));
    AccessWindowStatic output_access(output->info(), 0, 0,
                                     ceil_to_multiple(output->info()->dimension(0), height_aligned),
                                     ceil_to_multiple(output->info()->dimension(1), width_aligned));

    Window win = calculate_max_window(*input->info(), Steps(width_aligned, height_aligned));
    win.set_dimension_step(Window::DimX, num_elems_processed_per_iteration);
    win.set_dimension_step(Window::DimY, num_elems_processed_per_iteration);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, output->info()->valid_region());

    IGCKernel::configure(win);
}

void GCTransposeKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IGCKernel::window(), window);

    _kernel.use();

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;

        add_2D_tensor_argument(idx, _input, 1, slice);
        add_2D_tensor_argument(idx, _output, 2, slice);
        _kernel.update_shader_params();
        enqueue(*this, slice, _lws_hint);
    }
    while(window.slide_window_slice_2D(slice));
}
