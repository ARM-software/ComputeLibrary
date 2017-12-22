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
#include "arm_compute/core/NEON/kernels/NEHistogramKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IDistribution1D.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <arm_neon.h>
#include <array>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

inline void NEHistogramKernel::merge_histogram(uint32_t *global_hist, const uint32_t *local_hist, size_t bins)
{
    std::lock_guard<arm_compute::Mutex> lock(_hist_mtx);

    const unsigned int v_end = (bins / 4) * 4;

    for(unsigned int b = 0; b < v_end; b += 4)
    {
        const uint32x4_t tmp_global = vld1q_u32(global_hist + b);
        const uint32x4_t tmp_local  = vld1q_u32(local_hist + b);
        vst1q_u32(global_hist + b, vaddq_u32(tmp_global, tmp_local));
    }

    for(unsigned int b = v_end; b < bins; ++b)
    {
        global_hist[b] += local_hist[b];
    }
}

NEHistogramKernel::NEHistogramKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _local_hist(nullptr), _window_lut(nullptr), _hist_mtx()
{
}

void NEHistogramKernel::histogram_U8(Window win, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON(_output->buffer() == nullptr);

    const size_t          bins       = _output->num_bins();
    const int32_t         offset     = _output->offset();
    const uint32_t        offrange   = offset + _output->range();
    const uint32_t *const w_lut      = _window_lut;
    uint32_t *const       local_hist = _local_hist + info.thread_id * bins;

    // Clear local_histogram
    std::fill_n(local_hist, bins, 0);

    auto update_local_hist = [&](uint8_t p)
    {
        if(offset <= p && p < offrange)
        {
            ++local_hist[w_lut[p]];
        }
    };

    const int x_start = win.x().start();
    const int x_end   = win.x().end();

    // Handle X dimension manually to split into two loops
    // First one will use vector operations, second one processes the left over
    // pixels
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win);

    // Calculate local histogram
    execute_window_loop(win, [&](const Coordinates &)
    {
        int x = x_start;

        // Vector loop
        for(; x <= x_end - 8; x += 8)
        {
            const uint8x8_t pixels = vld1_u8(input.ptr() + x);

            update_local_hist(vget_lane_u8(pixels, 0));
            update_local_hist(vget_lane_u8(pixels, 1));
            update_local_hist(vget_lane_u8(pixels, 2));
            update_local_hist(vget_lane_u8(pixels, 3));
            update_local_hist(vget_lane_u8(pixels, 4));
            update_local_hist(vget_lane_u8(pixels, 5));
            update_local_hist(vget_lane_u8(pixels, 6));
            update_local_hist(vget_lane_u8(pixels, 7));
        }

        // Process leftover pixels
        for(; x < x_end; ++x)
        {
            update_local_hist(input.ptr()[x]);
        }
    },
    input);

    // Merge histograms
    merge_histogram(_output->buffer(), local_hist, bins);
}

void NEHistogramKernel::histogram_fixed_U8(Window win, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON(_output->buffer() == nullptr);

    std::array<uint32_t, _max_range_size> local_hist{ { 0 } };

    const int x_start = win.x().start();
    const int x_end   = win.x().end();

    // Handle X dimension manually to split into two loops
    // First one will use vector operations, second one processes the left over
    // pixels
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win);

    // Calculate local histogram
    execute_window_loop(win, [&](const Coordinates &)
    {
        int x = x_start;

        // Vector loop
        for(; x <= x_end - 8; x += 8)
        {
            const uint8x8_t pixels = vld1_u8(input.ptr() + x);

            ++local_hist[vget_lane_u8(pixels, 0)];
            ++local_hist[vget_lane_u8(pixels, 1)];
            ++local_hist[vget_lane_u8(pixels, 2)];
            ++local_hist[vget_lane_u8(pixels, 3)];
            ++local_hist[vget_lane_u8(pixels, 4)];
            ++local_hist[vget_lane_u8(pixels, 5)];
            ++local_hist[vget_lane_u8(pixels, 6)];
            ++local_hist[vget_lane_u8(pixels, 7)];
        }

        // Process leftover pixels
        for(; x < x_end; ++x)
        {
            ++local_hist[input.ptr()[x]];
        }
    },
    input);

    // Merge histograms
    merge_histogram(_output->buffer(), local_hist.data(), _max_range_size);
}

void NEHistogramKernel::calculate_window_lut() const
{
    const int32_t  offset = _output->offset();
    const size_t   bins   = _output->num_bins();
    const uint32_t range  = _output->range();

    std::fill_n(_window_lut, offset, 0);

    for(unsigned int p = offset; p < _max_range_size; ++p)
    {
        _window_lut[p] = ((p - offset) * bins) / range;
    }
}

void NEHistogramKernel::configure(const IImage *input, IDistribution1D *output, uint32_t *local_hist, uint32_t *window_lut)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    ARM_COMPUTE_ERROR_ON(nullptr == local_hist);
    ARM_COMPUTE_ERROR_ON(nullptr == window_lut);

    _input      = input;
    _output     = output;
    _local_hist = local_hist;
    _window_lut = window_lut;

    //Check offset
    ARM_COMPUTE_ERROR_ON_MSG(0 > _output->offset() || _output->offset() > static_cast<int32_t>(_max_range_size), "Offset is larger than the image value range.");

    //Check range
    ARM_COMPUTE_ERROR_ON_MSG(static_cast<int32_t>(_output->range()) > static_cast<int32_t>(_max_range_size) /* max range */, "Range larger than the image value range.");

    // Calculate LUT
    calculate_window_lut();

    // Set appropriate function
    _func = &NEHistogramKernel::histogram_U8;

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    INEKernel::configure(win);
}

void NEHistogramKernel::configure(const IImage *input, IDistribution1D *output)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    _input  = input;
    _output = output;

    // Set appropriate function
    _func = &NEHistogramKernel::histogram_fixed_U8;

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    INEKernel::configure(win);
}

void NEHistogramKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window, info);
}
