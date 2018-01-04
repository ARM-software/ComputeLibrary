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
    : _input(nullptr), _output(nullptr), _top_bottom(0), _left_right(0)
{
}

BorderSize GCDepthConcatenateLayerKernel::border_size() const
{
    return BorderSize(_top_bottom, _left_right);
}

void GCDepthConcatenateLayerKernel::configure(const IGCTensor *input, unsigned int depth_offset, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) + depth_offset > output->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) > output->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) > output->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(3, input, output);

    // The gaps between the two lowest dimensions of input and output need to be divisible by 2
    // Otherwise it is not clear how the padding should be added onto the input tensor
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(0) - input->info()->dimension(0)) % 2);
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(1) - input->info()->dimension(1)) % 2);

    _input  = input;
    _output = output;

    // Add build options
    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace(("#define " + dt_name));
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));

    // Configure kernel window
    _left_right = (output->info()->dimension(0) - input->info()->dimension(0)) / 2;
    _top_bottom = (output->info()->dimension(1) - input->info()->dimension(1)) / 2;

    const int offset_to_first_elements_in_bytes = depth_offset * output->info()->strides_in_bytes()[2];

    build_opts.emplace("#define OFFSETS_X " + support::cpp11::to_string(_left_right));
    build_opts.emplace("#define OFFSETS_Y " + support::cpp11::to_string(_top_bottom));
    build_opts.emplace("#define OFFSETS_Z " + support::cpp11::to_string(offset_to_first_elements_in_bytes));

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("concatenate_depth", build_opts));

    unsigned int num_elems_processed_per_iteration = 1;
    unsigned int num_elems_read_per_iteration      = 1;
    if(input->info()->data_type() == DataType::F32)
    {
        num_elems_processed_per_iteration = 1;
        num_elems_read_per_iteration      = 1;
    }
    else if(input->info()->data_type() == DataType::F16)
    {
        num_elems_processed_per_iteration = 4;
        num_elems_read_per_iteration      = 4;
    }
    const unsigned int num_rows_read_per_iteration = 1;

    // The window needs to be based on input as we copy all the depths of input
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    win.set(Window::DimZ, Window::Dimension(0, input->info()->tensor_shape().z(), 1));

    AccessWindowRectangle  input_access(input->info(), -_left_right, -_top_bottom, num_elems_read_per_iteration, num_rows_read_per_iteration);
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

    Window slice = window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, 1, slice);
        add_3D_tensor_argument(idx, _output, 2, slice);

        _kernel.update_shader_params();

        enqueue(*this, slice);
    }
    while(window.slide_window_slice_3D(slice));
}
