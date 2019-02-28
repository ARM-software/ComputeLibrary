/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <cstdint>

namespace arm_compute
{
class Coordinates;
namespace
{
inline void fill_constant_value_single_channel_special(ITensor *tensor, const Window &window, unsigned int right, unsigned int bottom, const PixelValue &constant_border_value)
{
    float border_value;
    constant_border_value.get(border_value);
    uint8_t *const start_valid_region = tensor->ptr_to_element(tensor->info()->valid_region().anchor);
    const size_t   width              = tensor->info()->valid_region().shape[0];
    const size_t   height             = tensor->info()->valid_region().shape[1];
    const int      stridey            = tensor->info()->strides_in_bytes()[1];

    // Left and right border
    Window vertical(window);
    vertical.set(Window::DimY, Window::Dimension(0, height, 1));

    Iterator vertical_it(tensor, vertical);

    execute_window_loop(vertical, [&](const Coordinates &)
    {
        const auto row_start = reinterpret_cast<float *>(start_valid_region + vertical_it.offset());

        // Fill left and right borders
        *(row_start - 1) = border_value;
        std::fill_n(row_start + width, right, border_value);
    },
    vertical_it);

    // Top and bottom border
    Iterator plane_it(tensor, window);

    // Iterate over all XY planes
    execute_window_loop(window, [&](const Coordinates &)
    {
        uint8_t *base_addr = start_valid_region + plane_it.offset();
        // Top border
        const auto row_start = reinterpret_cast<float *>(base_addr - stridey);
        // Fill top rows including left/right borders
        std::fill_n(row_start - 1, 1 + width + right, border_value);

        // Bottom border
        const unsigned low_border_size = height + bottom;
        for(unsigned int i = height; i < low_border_size; ++i)
        {
            const auto row_start = reinterpret_cast<float *>(base_addr + i * stridey);

            // Fill bottom rows including left/right borders
            std::fill_n(row_start - 1, 1 + width + right, border_value);
        }
    },
    plane_it);
}
} // namespace

NEFillBorderKernel::NEFillBorderKernel()
    : _tensor(nullptr), _border_size(0), _mode(BorderMode::UNDEFINED), _constant_border_value(static_cast<float>(0.f))
{
}

void NEFillBorderKernel::configure(ITensor *tensor, BorderSize border_size, BorderMode border_mode, const PixelValue &constant_border_value)
{
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(tensor, 1, DataType::U8, DataType::QASYMM8,
                                                  DataType::U16, DataType::S16,
                                                  DataType::U32, DataType::S32,
                                                  DataType::F16, DataType::F32);

    _tensor                = tensor;
    _border_size           = border_size;
    _mode                  = border_mode;
    _constant_border_value = constant_border_value;

    _border_size.limit(tensor->info()->padding());

    Window win;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    win.set(Window::DimY, Window::Dimension(0, 1, 1));
    win.use_tensor_dimensions(_tensor->info()->tensor_shape(), Window::DimZ);
    INEKernel::configure(win);
}

void NEFillBorderKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    // If there is no border: early exit
    if(_border_size.empty())
    {
        return;
    }

    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_mode)
    {
        case BorderMode::CONSTANT:
        {
            if(_border_size.left == 1 && _border_size.top == 1 && _tensor->info()->data_type() == DataType::F32)
            {
                fill_constant_value_single_channel_special(_tensor, window, _border_size.right, _border_size.bottom, _constant_border_value);
            }
            else
            {
                fill_constant_value_single_channel(window);
            }
            break;
        }
        case BorderMode::REPLICATE:
        {
            fill_replicate_single_channel(window);
            break;
        }
        case BorderMode::UNDEFINED:
            break; // Nothing to do here
        default:
            ARM_COMPUTE_ERROR("Unknown border mode");
    }
}

void NEFillBorderKernel::fill_replicate_single_channel(const Window &window)
{
    uint8_t *const start_valid_region = _tensor->ptr_to_element(_tensor->info()->valid_region().anchor);
    const size_t   width              = _tensor->info()->valid_region().shape[0];
    const size_t   height             = _tensor->info()->valid_region().shape[1];
    const size_t   element_size       = _tensor->info()->element_size();
    // Left and right border
    Window vertical(window);
    vertical.set(Window::DimY, Window::Dimension(0, height, 1));

    Iterator vertical_it(_tensor, vertical);

    execute_window_loop(vertical, [&](const Coordinates & id)
    {
        uint8_t *base_addr = start_valid_region + vertical_it.offset();
        // Fill left and right borders
        for(unsigned int i = 0; i < _border_size.left; ++i)
        {
            std::memcpy(base_addr + static_cast<int>(i - _border_size.left) * element_size, vertical_it.ptr(), element_size);
        }

        for(unsigned int i = 0; i < _border_size.right; ++i)
        {
            std::memcpy(base_addr + (width + i) * element_size, vertical_it.ptr() + (width - 1) * element_size, element_size);
        }
    },
    vertical_it);

    // Top and bottom border
    Iterator plane_it(_tensor, window);

    // Iterate over all XY planes
    execute_window_loop(window, [&](const Coordinates & id)
    {
        uint8_t *base_addr = start_valid_region + plane_it.offset();
        // Top border
        for(int i = -_border_size.top; i < 0; ++i)
        {
            // Copy top rows including left/right borders
            std::memcpy(base_addr + i * _tensor->info()->strides_in_bytes()[1] - _border_size.left * element_size,
                        base_addr - _border_size.left * element_size, (_border_size.left + width + _border_size.right) * element_size);
        }

        // Bottom border
        for(unsigned int i = height; i < height + _border_size.bottom; ++i)
        {
            // Copy bottom rows including left/right borders
            std::memcpy(base_addr + i * _tensor->info()->strides_in_bytes()[1] - _border_size.left * element_size,
                        base_addr + (height - 1) * _tensor->info()->strides_in_bytes()[1] - _border_size.left * element_size, (_border_size.left + width + _border_size.right) * element_size);
        }
    },
    plane_it);
}

void NEFillBorderKernel::fill_constant_value_single_channel(const Window &window)
{
    uint8_t *const start_valid_region = _tensor->ptr_to_element(_tensor->info()->valid_region().anchor);
    const size_t   width              = _tensor->info()->valid_region().shape[0];
    const size_t   height             = _tensor->info()->valid_region().shape[1];
    const int      stridey            = _tensor->info()->strides_in_bytes()[1];
    const size_t   element_size       = _tensor->info()->element_size();

    // Left and right border
    Window vertical(window);
    vertical.set(Window::DimY, Window::Dimension(0, height, 1));

    Iterator vertical_it(_tensor, vertical);

    execute_window_loop(vertical, [&](const Coordinates & id)
    {
        uint8_t *base_addr = start_valid_region + vertical_it.offset();
        // Fill left and right borders
        for(unsigned int i = 0; i < _border_size.left; ++i)
        {
            std::memcpy(base_addr + static_cast<int>(i - _border_size.left) * element_size, &_constant_border_value, element_size);
        }

        for(unsigned int i = 0; i < _border_size.right; ++i)
        {
            std::memcpy(base_addr + (width + i) * element_size, &_constant_border_value, element_size);
        }
    },
    vertical_it);

    // Top and bottom border
    Iterator plane_it(_tensor, window);

    // Iterate over all XY planes
    execute_window_loop(window, [&](const Coordinates & id)
    {
        uint8_t *base_addr = start_valid_region + plane_it.offset();
        // Top border
        for(int i = -_border_size.top; i < 0; ++i)
        {
            // Fill top rows including left/right borders
            for(unsigned int j = 0; j < (_border_size.left + width + _border_size.right); ++j)
            {
                std::memcpy(base_addr + i * stridey + static_cast<int>(j - _border_size.left) * element_size, &_constant_border_value, element_size);
            }
        }

        // Bottom border
        const unsigned low_border_size = height + _border_size.bottom;
        for(unsigned int i = height; i < low_border_size; ++i)
        {
            // Fill bottom rows including left/right borders
            for(unsigned int j = 0; j < (_border_size.left + width + _border_size.right); ++j)
            {
                std::memcpy(base_addr + i * stridey + static_cast<int>(j - _border_size.left) * element_size, &_constant_border_value, element_size);
            }
        }
    },
    plane_it);
}
} // namespace arm_compute
