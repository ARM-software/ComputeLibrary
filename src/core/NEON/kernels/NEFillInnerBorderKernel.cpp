/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEFillInnerBorderKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

NEFillInnerBorderKernel::NEFillInnerBorderKernel()
    : _tensor(nullptr), _border_size(0), _constant_border_value(static_cast<float>(0.f))
{
}

void NEFillInnerBorderKernel::configure(ITensor *input, BorderSize border_size, const PixelValue &constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16, DataType::S32, DataType::F32);

    _tensor                = input;
    _border_size           = border_size;
    _constant_border_value = constant_border_value;

    Window win;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    win.set(Window::DimY, Window::Dimension(0, 1, 1));
    win.use_tensor_dimensions(_tensor->info()->tensor_shape(), Window::DimZ);
    INEKernel::configure(win);
}

void NEFillInnerBorderKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    // If there is no border: early exit
    if(_border_size.empty())
    {
        return;
    }

    switch(_tensor->info()->data_type())
    {
        case DataType::U8:
            fill_value_single_channel<uint8_t>(window);
            break;
        case DataType::S16:
            fill_value_single_channel<int16_t>(window);
            break;
        case DataType::S32:
            fill_value_single_channel<int32_t>(window);
            break;
        case DataType::F32:
            static_assert(sizeof(float) == 4, "Float must be 32 bit");
            fill_value_single_channel<float>(window);
            break;
        default:
            ARM_COMPUTE_ERROR("Not handled");
            break;
    }
}

template <typename T>
void NEFillInnerBorderKernel::fill_value_single_channel(const Window &window)
{
    const size_t stride = _tensor->info()->strides_in_bytes()[1];
    const size_t width  = _tensor->info()->dimension(0);
    const size_t height = _tensor->info()->dimension(1);

    T constant_border_value;
    _constant_border_value.get(constant_border_value);

    // Left and right border
    // All X values are set at once
    Window vertical(window);
    vertical.set(Window::DimY, Window::Dimension(0, height, 1));

    Iterator vertical_it(_tensor, vertical);

    execute_window_loop(vertical, [&](const Coordinates & id)
    {
        std::fill_n(reinterpret_cast<T *>(vertical_it.ptr()), _border_size.left, constant_border_value);
        std::fill_n(reinterpret_cast<T *>(vertical_it.ptr()) + width - _border_size.right, _border_size.right, constant_border_value);
    },
    vertical_it);

    // Top and bottom border
    // All values are set at once
    Iterator horizontal_it(_tensor, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        for(size_t i = 0; i < _border_size.top; ++i)
        {
            std::fill_n(reinterpret_cast<T *>(horizontal_it.ptr() + i * stride), width, constant_border_value);
        }

        for(size_t i = 0; i < _border_size.bottom; ++i)
        {
            std::fill_n(reinterpret_cast<T *>(horizontal_it.ptr() + (height - i - 1) * stride), width, constant_border_value);
        }
    },
    horizontal_it);
}
