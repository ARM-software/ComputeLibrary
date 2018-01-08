/*
 * Copyright (c) 2016-2018 ARM Limited.
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

using namespace arm_compute;

namespace
{
template <typename T, unsigned int leftx, unsigned int rightx>
void fill_constant_value_single_channel_special(ITensor *tensor, const Window &window, unsigned int right, unsigned int bottom, const PixelValue &constant_border_value);

template <>
inline void fill_constant_value_single_channel_special<float, 1u, 1u>(ITensor *tensor, const Window &window, unsigned int right, unsigned int bottom, const PixelValue &constant_border_value)
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

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

NEFillBorderKernel::NEFillBorderKernel()
    : _tensor(nullptr), _border_size(0), _mode(BorderMode::UNDEFINED), _constant_border_value(static_cast<float>(0.f))
{
}

void NEFillBorderKernel::configure(ITensor *tensor, BorderSize border_size, BorderMode border_mode, const PixelValue &constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(tensor, 1, DataType::U8, DataType::QS8, DataType::QASYMM8,
                                                  DataType::QS16, DataType::U16, DataType::S16,
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
            switch(_tensor->info()->data_type())
            {
                case DataType::QASYMM8:
                case DataType::U8:
                    fill_constant_value_single_channel<uint8_t>(window);
                    break;
                case DataType::QS8:
                case DataType::S8:
                    fill_constant_value_single_channel<int8_t>(window);
                    break;
                case DataType::U16:
                    fill_constant_value_single_channel<uint16_t>(window);
                    break;
                case DataType::S16:
                case DataType::QS16:
                    fill_constant_value_single_channel<int16_t>(window);
                    break;
                case DataType::U32:
                    fill_constant_value_single_channel<uint32_t>(window);
                    break;
                case DataType::S32:
                    fill_constant_value_single_channel<int32_t>(window);
                    break;
                case DataType::F16:
                    static_assert(sizeof(half) == 2, "Float16_t must be 16 bit");
                    fill_constant_value_single_channel<half>(window);
                    break;
                case DataType::F32:
                    static_assert(sizeof(float) == 4, "Float must be 32 bit");
                    if(_border_size.left == 1 && _border_size.top == 1)
                    {
                        fill_constant_value_single_channel_special<float, 1u, 1u>(_tensor, window, _border_size.right, _border_size.bottom, _constant_border_value);
                    }
                    else
                    {
                        fill_constant_value_single_channel<float>(window);
                    }
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not handled");
            }
            break;
        }
        case BorderMode::REPLICATE:
        {
            switch(_tensor->info()->data_type())
            {
                case DataType::QASYMM8:
                case DataType::U8:
                    fill_replicate_single_channel<uint8_t>(window);
                    break;
                case DataType::QS8:
                case DataType::S8:
                    fill_replicate_single_channel<int8_t>(window);
                    break;
                case DataType::U16:
                    fill_replicate_single_channel<uint16_t>(window);
                    break;
                case DataType::S16:
                case DataType::QS16:
                    fill_replicate_single_channel<int16_t>(window);
                    break;
                case DataType::U32:
                    fill_replicate_single_channel<uint32_t>(window);
                    break;
                case DataType::S32:
                    fill_replicate_single_channel<int32_t>(window);
                    break;
                case DataType::F16:
                    static_assert(sizeof(half) == 2, "Float16_t must be 16 bit");
                    fill_replicate_single_channel<half>(window);
                    break;
                case DataType::F32:
                    static_assert(sizeof(float) == 4, "Float must be 32 bit");
                    fill_replicate_single_channel<float>(window);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not handled");
            }
            break;
        }
        case BorderMode::UNDEFINED:
            break; // Nothing to do here
        default:
            ARM_COMPUTE_ERROR("Unknown border mode");
    }
}

template <typename T>
void NEFillBorderKernel::fill_replicate_single_channel(const Window &window)
{
    uint8_t *const start_valid_region = _tensor->ptr_to_element(_tensor->info()->valid_region().anchor);
    const size_t   width              = _tensor->info()->valid_region().shape[0];
    const size_t   height             = _tensor->info()->valid_region().shape[1];

    // Left and right border
    Window vertical(window);
    vertical.set(Window::DimY, Window::Dimension(0, height, 1));

    Iterator vertical_it(_tensor, vertical);

    execute_window_loop(vertical, [&](const Coordinates & id)
    {
        const auto row_start = reinterpret_cast<T *>(start_valid_region + vertical_it.offset());
        const auto left_val  = *reinterpret_cast<T *>(vertical_it.ptr());
        const auto right_val = *(reinterpret_cast<T *>(vertical_it.ptr()) + width - 1);

        // Fill left and right borders
        std::fill_n(row_start - _border_size.left, _border_size.left, left_val);
        std::fill_n(row_start + width, _border_size.right, right_val);
    },
    vertical_it);

    // Top and bottom border
    Iterator plane_it(_tensor, window);

    // Iterate over all XY planes
    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto first_row = reinterpret_cast<T *>(start_valid_region + plane_it.offset());

        // Top border
        for(int i = -_border_size.top; i < 0; ++i)
        {
            const auto row_start = reinterpret_cast<T *>(start_valid_region + plane_it.offset() + i * _tensor->info()->strides_in_bytes()[1]);

            // Copy top rows including left/right borders
            std::copy_n(first_row - _border_size.left, _border_size.left + width + _border_size.right, row_start - _border_size.left);
        }

        const auto last_row = reinterpret_cast<T *>(start_valid_region + plane_it.offset() + (height - 1) * _tensor->info()->strides_in_bytes()[1]);

        // Bottom border
        for(unsigned int i = height; i < height + _border_size.bottom; ++i)
        {
            const auto row_start = reinterpret_cast<T *>(start_valid_region + plane_it.offset() + i * _tensor->info()->strides_in_bytes()[1]);

            // Copy bottom rows including left/right borders
            std::copy_n(last_row - _border_size.left, _border_size.left + width + _border_size.right, row_start - _border_size.left);
        }
    },
    plane_it);
}

template <typename T>
void NEFillBorderKernel::fill_constant_value_single_channel(const Window &window)
{
    T constant_border_value;
    _constant_border_value.get(constant_border_value);

    uint8_t *const start_valid_region = _tensor->ptr_to_element(_tensor->info()->valid_region().anchor);
    const size_t   width              = _tensor->info()->valid_region().shape[0];
    const size_t   height             = _tensor->info()->valid_region().shape[1];
    const int      stridey            = _tensor->info()->strides_in_bytes()[1];

    // Left and right border
    Window vertical(window);
    vertical.set(Window::DimY, Window::Dimension(0, height, 1));

    Iterator vertical_it(_tensor, vertical);

    execute_window_loop(vertical, [&](const Coordinates & id)
    {
        const auto row_start = reinterpret_cast<T *>(start_valid_region + vertical_it.offset());

        // Fill left and right borders
        std::fill_n(row_start - _border_size.left, _border_size.left, constant_border_value);
        std::fill_n(row_start + width, _border_size.right, constant_border_value);
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
            const auto row_start = reinterpret_cast<T *>(base_addr + i * stridey);

            // Fill top rows including left/right borders
            std::fill_n(row_start - _border_size.left, _border_size.left + width + _border_size.right, constant_border_value);
        }

        // Bottom border
        const unsigned low_border_size = height + _border_size.bottom;
        for(unsigned int i = height; i < low_border_size; ++i)
        {
            const auto row_start = reinterpret_cast<T *>(base_addr + i * stridey);

            // Fill bottom rows including left/right borders
            std::fill_n(row_start - _border_size.left, _border_size.left + width + _border_size.right, constant_border_value);
        }
    },
    plane_it);
}
