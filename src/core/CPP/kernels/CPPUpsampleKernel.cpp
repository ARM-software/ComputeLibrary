/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/core/CPP/kernels/CPPUpsampleKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
CPPUpsampleKernel::CPPUpsampleKernel()
    : _input(nullptr), _output(nullptr), _info()
{
}

bool CPPUpsampleKernel::is_parallelisable() const
{
    return false;
}

void CPPUpsampleKernel::configure(const ITensor *input, ITensor *output, const PadStrideInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _input  = input;
    _output = output;
    _info   = info;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());

    // The CPPUpsampleKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    ICPPKernel::configure(win);
}

void CPPUpsampleKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

    const DataLayout data_layout = _input->info()->data_layout();
    const size_t     idx_w       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t     idx_h       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Initialize _scaled_output buffer
    const int    width_scaled  = _output->info()->dimension(idx_w);
    const int    height_scaled = _output->info()->dimension(idx_h);
    const int    stride_width  = _info.stride().first;
    const int    stride_height = _info.stride().second;
    const int    start_width   = _info.pad_left();
    const int    start_height  = _info.pad_top();
    const int    end_width     = width_scaled - _info.pad_right();
    const int    end_height    = height_scaled - _info.pad_bottom();
    const size_t element_size  = _input->info()->element_size();

    // The fill value is normally 0, but for quantized types '0' corresponds to the offset
    switch(_output->info()->data_type())
    {
        case DataType::QASYMM8:
        {
            const uint8_t fill_value = _output->info()->quantization_info().uniform().offset;
            std::fill_n(_output->buffer(), _output->info()->total_size(), fill_value);
        }
        break;
        case DataType::QASYMM8_SIGNED:
        {
            const int8_t fill_value = _output->info()->quantization_info().uniform().offset;
            std::fill_n(_output->buffer(), _output->info()->total_size(), fill_value);
        }
        break;
        default:
            std::fill_n(_output->buffer(), _output->info()->total_size(), 0);
    }

    // Create window
    Window window_out(window);
    if(data_layout == DataLayout::NCHW)
    {
        window_out.set(Window::DimX, Window::Dimension(start_width, end_width, stride_width));
        window_out.set(Window::DimY, Window::Dimension(start_height, end_height, stride_height));
    }
    else
    {
        window_out.set(Window::DimY, Window::Dimension(start_width, end_width, stride_width));
        window_out.set(Window::DimZ, Window::Dimension(start_height, end_height, stride_height));
    }

    // Create iterators
    Iterator in(_input, window);
    Iterator out(_output, window_out);

    execute_window_loop(window, [&](const Coordinates &)
    {
        memcpy(out.ptr(), in.ptr(), element_size);
    },
    in, out);
}
} // namespace arm_compute