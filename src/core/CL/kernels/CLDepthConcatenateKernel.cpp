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
#include "arm_compute/core/CL/kernels/CLDepthConcatenateKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

CLDepthConcatenateKernel::CLDepthConcatenateKernel()
    : _input(nullptr), _output(nullptr), _top_bottom(0), _left_right(0)
{
}

BorderSize CLDepthConcatenateKernel::border_size() const
{
    return BorderSize(_top_bottom, _left_right);
}

void CLDepthConcatenateKernel::configure(const ICLTensor *input, unsigned int depth_offset, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
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

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("concatenate_depth"));

    // Configure kernel window
    _left_right = (output->info()->dimension(0) - input->info()->dimension(0)) / 2;
    _top_bottom = (output->info()->dimension(1) - input->info()->dimension(1)) / 2;

    const unsigned int offset_to_first_elements_in_bytes = depth_offset * output->info()->strides_in_bytes()[2] + _left_right * output->info()->strides_in_bytes()[0] + _top_bottom *
                                                           output->info()->strides_in_bytes()[1];

    const unsigned int num_elems_processed_per_iteration = 4;
    const unsigned int num_elems_read_per_iteration      = 4;
    const unsigned int num_rows_read_per_iteration       = 1;

    // The window needs to be based on input as we copy all the depths of input
    Window win = calculate_max_enlarged_window(*input->info(), Steps(num_elems_processed_per_iteration), border_size());

    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), -_left_right, -_top_bottom, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));

    unsigned int idx = 2 * num_arguments_per_2D_tensor(); // Skip the input and output parameters
    _kernel.setArg<unsigned int>(idx, offset_to_first_elements_in_bytes);

    ICLKernel::configure(win);
}

void CLDepthConcatenateKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
