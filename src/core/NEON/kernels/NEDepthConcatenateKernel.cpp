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
#include "arm_compute/core/NEON/kernels/NEDepthConcatenateKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

using namespace arm_compute;

NEDepthConcatenateKernel::NEDepthConcatenateKernel()
    : _input(nullptr), _output(nullptr), _top_bottom(0), _left_right(0), _depth_offset(0)
{
}

BorderSize NEDepthConcatenateKernel::border_size() const
{
    return BorderSize(_top_bottom, _left_right);
}

void NEDepthConcatenateKernel::configure(const ITensor *input, unsigned int depth_offset, ITensor *output)
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

    _input        = input;
    _output       = output;
    _depth_offset = depth_offset;
    _left_right   = (output->info()->dimension(0) - input->info()->dimension(0)) / 2;
    _top_bottom   = (output->info()->dimension(1) - input->info()->dimension(1)) / 2;

    const unsigned int num_elems_processed_per_iteration = 4;
    const unsigned int num_elems_read_per_iteration      = 4;
    const unsigned int num_rows_read_per_iteration       = 1;

    // The window needs to be based on input as we copy all the depths of input
    Window win = calculate_max_enlarged_window(*input->info(), Steps(num_elems_processed_per_iteration), border_size());

    AccessWindowRectangle  input_access(input->info(), -_left_right, -_top_bottom, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEDepthConcatenateKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    // Offset output
    const unsigned int offset_to_first_elements_in_bytes = _output->info()->offset_first_element_in_bytes() + _left_right * _output->info()->strides_in_bytes()[0] + _top_bottom *
                                                           _output->info()->strides_in_bytes()[1] + _depth_offset * _output->info()->strides_in_bytes()[2];
    uint8_t           *output_ptr                        = _output->buffer() + offset_to_first_elements_in_bytes;

    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto in_ptr  = reinterpret_cast<const float *>(input.ptr());
        const auto out_ptr = reinterpret_cast<float *>(output_ptr + output.offset());

        vst1q_f32(out_ptr, vld1q_f32(in_ptr));
    },
    input, output);
}
