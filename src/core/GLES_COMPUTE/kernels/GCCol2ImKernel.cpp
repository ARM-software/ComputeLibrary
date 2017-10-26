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

#include "arm_compute/core/GLES_COMPUTE/kernels/GCCol2ImKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

GCCol2ImKernel::GCCol2ImKernel()
    : _input(nullptr), _output(nullptr), _convolved_dims()
{
}

void GCCol2ImKernel::configure(const IGCTensor *input, IGCTensor    *output,
                               std::pair<unsigned int, unsigned int> convolved_dims)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _kernel.clear_params();

    _input          = input;
    _output         = output;
    _convolved_dims = convolved_dims;

    // Create kernel
    std::set<std::string>  build_opts;
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    build_opts.insert("#define COL2IM");
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("col2im", build_opts));

    // Set static kernel arguments
    unsigned int idx = num_arguments_per_2D_tensor() + num_arguments_per_3D_tensor();
    _kernel.set_params(idx++, _convolved_dims.first);

    // Configure window
    Window win = calculate_max_window(*input->info(), Steps());

    // The GCCol2ImKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    // set shader params binding point
    _kernel.set_shader_params_binding_point(0);

    IGCKernel::configure(win);
}

void GCCol2ImKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IGCKernel::window(), window);

    Window slice_in  = window.first_slice_window_2D();
    Window slice_out = window.first_slice_window_3D();

    _kernel.use();

    do
    {
        // Set inputs
        unsigned int idx     = 0;
        unsigned int binding = 1;
        add_2D_tensor_argument(idx, _input, binding++, slice_in);
        add_3D_tensor_argument(idx, _output, binding++, slice_out);
        _kernel.update_shader_params();
        enqueue(*this, slice_in);
    }
    while(window.slide_window_slice_2D(slice_in) && window.slide_window_slice_3D(slice_out));
}
