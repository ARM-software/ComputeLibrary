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

#include "arm_compute/core/AccessWindowAutoPadding.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IDistribution1D.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
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
    std::lock_guard<std::mutex> lock(_hist_mtx);

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

void NEHistogramKernel::histogram_U8(const Window &win)
{
    ARM_COMPUTE_ERROR_ON(_output->buffer() == nullptr);

    const size_t          bins       = _output->num_bins();
    const int32_t         offset     = _output->offset();
    const uint32_t        offrange   = offset + _output->range();
    const uint32_t *const w_lut      = _window_lut;
    uint32_t *const       local_hist = _local_hist + win.thread_id() * bins;

    // Clear local_histogram
    std::fill_n(local_hist, bins, 0);

    auto update_local_hist = [&](uint8_t p)
    {
        if(offset <= p && p < offrange)
        {
            ++local_hist[w_lut[p]];
        }
    };

    Iterator input(_input, win);

    // Calculate local histogram
    execute_window_loop(win, [&](const Coordinates &)
    {
        const uint8x8_t pixels = vld1_u8(input.ptr());

        update_local_hist(vget_lane_u8(pixels, 0));
        update_local_hist(vget_lane_u8(pixels, 1));
        update_local_hist(vget_lane_u8(pixels, 2));
        update_local_hist(vget_lane_u8(pixels, 3));
        update_local_hist(vget_lane_u8(pixels, 4));
        update_local_hist(vget_lane_u8(pixels, 5));
        update_local_hist(vget_lane_u8(pixels, 6));
        update_local_hist(vget_lane_u8(pixels, 7));
    },
    input);

    // Merge histograms
    merge_histogram(_output->buffer(), local_hist, bins);
}

void NEHistogramKernel::histogram_fixed_U8(const Window &win)
{
    ARM_COMPUTE_ERROR_ON(_output->buffer() == nullptr);

    std::array<uint32_t, _max_range_size> local_hist{ { 0 } };

    Iterator input(_input, win);

    // Calculate local histogram
    execute_window_loop(win, [&](const Coordinates & id)
    {
        const uint8x8_t pixels = vld1_u8(input.ptr());

        ++local_hist[vget_lane_u8(pixels, 0)];
        ++local_hist[vget_lane_u8(pixels, 1)];
        ++local_hist[vget_lane_u8(pixels, 2)];
        ++local_hist[vget_lane_u8(pixels, 3)];
        ++local_hist[vget_lane_u8(pixels, 4)];
        ++local_hist[vget_lane_u8(pixels, 5)];
        ++local_hist[vget_lane_u8(pixels, 6)];
        ++local_hist[vget_lane_u8(pixels, 7)];
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

void NEHistogramKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
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

    constexpr unsigned int processed_elements = 8;

    // We only run histogram on Image, therefore only 2 dimensions here
    const unsigned int end_position = floor_to_multiple(_input->info()->dimension(0), processed_elements);

    Window win;
    win.set(0, Window::Dimension(0, end_position, processed_elements));
    win.set(1, Window::Dimension(0, _input->info()->dimension(1)));

    update_window_and_padding(win, AccessWindowAutoPadding(input->info()));

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

    constexpr unsigned int processed_elements = 8;

    // We only run histogram on Image, therefore only 2 dimensions here
    const unsigned int end_position = floor_to_multiple(_input->info()->dimension(0), processed_elements);

    Window win;
    win.set(0, Window::Dimension(0, end_position, processed_elements));
    win.set(1, Window::Dimension(0, _input->info()->dimension(1)));

    update_window_and_padding(win, AccessWindowAutoPadding(input->info()));

    INEKernel::configure(win);
}

NEHistogramBorderKernel::NEHistogramBorderKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _window_lut(nullptr)
{
}

bool NEHistogramBorderKernel::is_parallelisable() const
{
    return false;
}

void NEHistogramBorderKernel::histogram_U8(const Window &win)
{
    const int32_t         offset   = _output->offset();
    const uint32_t        offrange = offset + _output->range();
    const uint32_t *const w_lut    = _window_lut;
    uint32_t *const       out_ptr  = _output->buffer();

    ARM_COMPUTE_ERROR_ON(out_ptr == nullptr);

    Iterator input(_input, win);

    // Calculate local histogram
    execute_window_loop(win, [&](const Coordinates & id)
    {
        const uint8_t pixel = *input.ptr();

        if(offset <= pixel && pixel < offrange)
        {
            ++out_ptr[w_lut[pixel]];
        }
    },
    input);
}

void NEHistogramBorderKernel::histogram_fixed_U8(const Window &win)
{
    uint32_t *const out_ptr = _output->buffer();
    ARM_COMPUTE_ERROR_ON(out_ptr == nullptr);

    Iterator input(_input, win);

    // Calculate local histogram
    execute_window_loop(win, [&](const Coordinates & id)
    {
        ++out_ptr[*input.ptr()];
    },
    input);
}

void NEHistogramBorderKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}

void NEHistogramBorderKernel::configure(const IImage *input, IDistribution1D *output, uint32_t *window_lut, const unsigned int hist_elements_per_thread)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    ARM_COMPUTE_ERROR_ON(nullptr == window_lut);

    _input      = input;
    _output     = output;
    _window_lut = window_lut;

    //Check offset
    ARM_COMPUTE_ERROR_ON_MSG(0 > _output->offset() || _output->offset() > static_cast<int32_t>(_max_range_size), "Offset is larger than the image value range.");

    //Check range
    ARM_COMPUTE_ERROR_ON_MSG(static_cast<int32_t>(_output->range()) > static_cast<int32_t>(_max_range_size) /* max range */, "Range larger than the image value range.");

    // Set appropriate function
    _func = &NEHistogramBorderKernel::histogram_U8;

    // We only run histogram on Image, therefore only 2 dimensions here
    const unsigned int start_position = floor_to_multiple(input->info()->dimension(0), hist_elements_per_thread);

    ARM_COMPUTE_ERROR_ON(start_position >= input->info()->dimension(0));

    Window win;
    win.set(0, Window::Dimension(start_position, _input->info()->dimension(0)));
    win.set(1, Window::Dimension(0, _input->info()->dimension(1)));

    update_window_and_padding(win, AccessWindowAutoPadding(input->info()));

    INEKernel::configure(win);
}

void NEHistogramBorderKernel::configure(const IImage *input, IDistribution1D *output, unsigned int hist_elements_per_thread)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    _input  = input;
    _output = output;

    // Set appropriate function
    _func = &NEHistogramBorderKernel::histogram_fixed_U8;

    // We only run histogram on Image, therefore only 2 dimensions here
    const unsigned int start_position = floor_to_multiple(input->info()->dimension(0), hist_elements_per_thread);

    Window win;
    win.set(0, Window::Dimension(start_position, _input->info()->dimension(0)));
    win.set(1, Window::Dimension(0, _input->info()->dimension(1)));

    INEKernel::configure(win);
}
