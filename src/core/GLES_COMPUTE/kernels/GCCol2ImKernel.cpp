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

#include "arm_compute/core/GLES_COMPUTE/kernels/GCCol2ImKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
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
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    TensorShape output_shape = input->info()->tensor_shape();
    output_shape.set(0, convolved_dims.first);
    output_shape.set(1, convolved_dims.second);
    output_shape.set(2, input->info()->tensor_shape()[0]);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input          = input;
    _output         = output;
    _convolved_dims = convolved_dims;

    unsigned int num_elems_processed_per_iteration = 1;

    // Create kernel
    std::set<std::string> build_opts;
    build_opts.emplace("#define WIDTH_OUTPUT " + support::cpp11::to_string(_convolved_dims.first));
    std::string dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace(("#define " + dt_name));
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(num_elems_processed_per_iteration));

    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("col2im", build_opts));

    // Configure window
    unsigned int nums = 2;
    Window       win  = calculate_max_window(*output->info(), Steps(nums));

    AccessWindowHorizontal output_access(output->info(), 0, 2);
    const int              input_padding = ceil_to_multiple(input->info()->dimension(0), 2) - input->info()->dimension(0);

    AccessWindowStatic input_access(input->info(), 0, 0, input->info()->dimension(0) + input_padding, input->info()->dimension(1) + 1);

    update_window_and_padding(win, input_access,
                              output_access);

    output_access.set_valid_region(win, output->info()->valid_region());

    IGCKernel::configure(win);
}

void GCCol2ImKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IGCKernel::window(), window);

    _kernel.use();

    Window collapsed_window = window.collapse_if_possible(IGCKernel::window(), Window::DimZ);
    Window slice            = collapsed_window.first_slice_window_3D();

    // Set static kernel arguments
    unsigned int idx = 2 * num_arguments_per_3D_tensor();
    //_kernel.set_argument(idx++, _output->info()->strides_in_bytes()[3]);
    _kernel.set_argument(idx++, uint(_output->info()->dimension(2)));
    _kernel.set_argument(idx++, _input->info()->strides_in_bytes()[2]);

    do
    {
        // Set inputs
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, 1, slice);
        add_3D_tensor_argument(idx, _output, 2, slice);
        _kernel.update_shader_params();
        enqueue(*this, slice);
    }
    while(collapsed_window.slide_window_slice_3D(slice));
}
