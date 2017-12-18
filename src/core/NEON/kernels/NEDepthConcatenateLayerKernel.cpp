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
#include "arm_compute/core/NEON/kernels/NEDepthConcatenateLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstdint>

using namespace arm_compute;

namespace
{
// Overloads of 128-bit vector loads
uint8x16_t loadq(const uint8_t *ptr)
{
    return vld1q_u8(ptr);
}
uint16x8_t loadq(const uint16_t *ptr)
{
    return vld1q_u16(ptr);
}
uint32x4_t loadq(const uint32_t *ptr)
{
    return vld1q_u32(ptr);
}
// Overloads of 128-bit vector stores
void storeq(uint8_t *ptr, uint8x16_t val)
{
    return vst1q_u8(ptr, val);
}
void storeq(uint16_t *ptr, uint16x8_t val)
{
    return vst1q_u16(ptr, val);
}
void storeq(uint32_t *ptr, uint32x4_t val)
{
    return vst1q_u32(ptr, val);
}

template <typename T>
void depth_concat(const ITensor *in, ITensor *out, std::pair<int, int> start_xy, int depth_offset, const Window &window)
{
    const int start_x = start_xy.first;
    const int start_y = start_xy.second;

    // Offset input
    const int input_offset_to_first_elements_in_bytes = in->info()->offset_first_element_in_bytes() - start_x * in->info()->strides_in_bytes()[0] - start_y * in->info()->strides_in_bytes()[1];
    uint8_t *input_ptr                               = in->buffer() + input_offset_to_first_elements_in_bytes;

    // Offset output
    const unsigned int output_offset_to_first_elements_in_bytes = out->info()->offset_first_element_in_bytes() + depth_offset * out->info()->strides_in_bytes()[2];
    uint8_t           *output_ptr                               = out->buffer() + output_offset_to_first_elements_in_bytes;

    Iterator input(in, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto in_ptr  = reinterpret_cast<const T *>(input_ptr + input.offset());
        const auto out_ptr = reinterpret_cast<T *>(output_ptr + output.offset());

        storeq(out_ptr, loadq(in_ptr));
    },
    input, output);
}
} // namespace

NEDepthConcatenateLayerKernel::NEDepthConcatenateLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _top_bottom(0), _left_right(0), _depth_offset(0)
{
}

BorderSize NEDepthConcatenateLayerKernel::border_size() const
{
    return BorderSize(_top_bottom, _left_right);
}

void NEDepthConcatenateLayerKernel::configure(const ITensor *input, unsigned int depth_offset, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) + depth_offset > output->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) > output->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) > output->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(3, input, output);

    // The gaps between the two lowest dimensions of input and output need to be divisible by 2
    // Otherwise it is not clear how the padding should be added onto the input tensor
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(0) - input->info()->dimension(0)) % 2);
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(1) - input->info()->dimension(1)) % 2);

    _func         = nullptr;
    _input        = input;
    _output       = output;
    _depth_offset = depth_offset;
    _left_right   = (output->info()->dimension(0) - input->info()->dimension(0)) / 2;
    _top_bottom   = (output->info()->dimension(1) - input->info()->dimension(1)) / 2;

    switch(input->info()->data_type())
    {
        case DataType::QS8:
            _func = &depth_concat<uint8_t>;
            break;
        case DataType::QS16:
        case DataType::F16:
            _func = &depth_concat<uint16_t>;
            break;
        case DataType::F32:
            _func = &depth_concat<uint32_t>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();
    const unsigned int num_elems_read_per_iteration      = 16 / input->info()->element_size();
    const unsigned int num_rows_read_per_iteration       = 1;

    // The window needs to be based on input as we copy all the depths of input
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    win.set(Window::DimZ, Window::Dimension(0, input->info()->tensor_shape().z(), 1));

    AccessWindowRectangle  input_access(input->info(), -_left_right, -_top_bottom, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEDepthConcatenateLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _output, std::make_pair(_left_right, _top_bottom), _depth_offset, window);
}
