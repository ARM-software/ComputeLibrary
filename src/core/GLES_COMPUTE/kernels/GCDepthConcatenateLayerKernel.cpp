/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCDepthConcatenateLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

using namespace arm_compute;

GCDepthConcatenateLayerKernel::GCDepthConcatenateLayerKernel()
    : _input(nullptr), _output(nullptr), _depth_offset(0)
{
}
void GCDepthConcatenateLayerKernel::configure(const IGCTensor *input, unsigned int depth_offset, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    ARM_COMPUTE_ERROR_ON(input->info()->dimension(Window::DimX) != output->info()->dimension(Window::DimX));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(Window::DimY) != output->info()->dimension(Window::DimY));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) + depth_offset > output->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(3, input, output);

    _input        = input;
    _output       = output;
    _depth_offset = depth_offset;

    // Add build options
    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace(("#define " + dt_name));
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("concatenate_depth", build_opts));

    unsigned int num_elems_processed_per_iteration = 1;
    if(input->info()->data_type() == DataType::F16)
    {
        num_elems_processed_per_iteration = 4;
    }

    // The window needs to be based on input as we copy all the depths of input
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    win.set(Window::DimZ, Window::Dimension(0, input->info()->tensor_shape().z(), 1));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    IGCKernel::configure(win);
}

void GCDepthConcatenateLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IGCKernel::window(), window);

    _kernel.use();

    _output->set_needs_shifting(true);

    Window slice_in  = window.first_slice_window_3D();
    Window slice_out = window.first_slice_window_3D();

    slice_out.set(Window::DimZ, Window::Dimension(_depth_offset));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, 1, slice_in);
        add_3D_tensor_argument(idx, _output, 2, slice_out);

        _kernel.update_shader_params();

        enqueue(*this, slice_in);
    }
    while(window.slide_window_slice_3D(slice_in));
}
