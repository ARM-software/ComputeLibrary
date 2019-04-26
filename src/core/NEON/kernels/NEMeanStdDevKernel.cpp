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
#include "arm_compute/core/NEON/kernels/NEMeanStdDevKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cmath>
#include <tuple>
#include <utility>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
template <bool calc_sum_squared>
std::pair<uint64x1_t, uint64x1_t> accumulate(const Window &window, Iterator &iterator)
{
    uint64x1_t sum         = vdup_n_u64(0);
    uint64x1_t sum_squared = vdup_n_u64(0);

    // Calculate sum
    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t in_data = vld1q_u8(iterator.ptr());

        // Sum of the low and high elements of data
        const uint16x8_t tmp0 = vaddl_u8(vget_low_u8(in_data), vget_high_u8(in_data));
        const uint32x4_t tmp1 = vaddl_u16(vget_low_u16(tmp0), vget_high_u16(tmp0));
        const uint32x2_t tmp2 = vadd_u32(vget_low_u32(tmp1), vget_high_u32(tmp1));

        // Update sum
        sum = vpadal_u32(sum, tmp2);

        if(calc_sum_squared)
        {
            const uint16x8_t square_data_low  = vmull_u8(vget_low_u8(in_data), vget_low_u8(in_data));
            const uint16x8_t square_data_high = vmull_u8(vget_high_u8(in_data), vget_high_u8(in_data));

            // Sum of the low and high elements of data
            const uint32x4_t tmp0_low  = vaddl_u16(vget_low_u16(square_data_low), vget_high_u16(square_data_low));
            const uint32x4_t tmp0_high = vaddl_u16(vget_low_u16(square_data_high), vget_high_u16(square_data_high));
            const uint32x4_t tmp1      = vaddq_u32(tmp0_low, tmp0_high);
            const uint32x2_t tmp2      = vadd_u32(vget_low_u32(tmp1), vget_high_u32(tmp1));

            // Update sum
            sum_squared = vpadal_u32(sum_squared, tmp2);
        }
    },
    iterator);

    return std::make_pair(sum, sum_squared);
}
} // namespace

NEMeanStdDevKernel::NEMeanStdDevKernel()
    : _input(nullptr), _mean(nullptr), _stddev(nullptr), _global_sum(nullptr), _global_sum_squared(nullptr), _mtx(), _border_size(0)
{
}

BorderSize NEMeanStdDevKernel::border_size() const
{
    return _border_size;
}

void NEMeanStdDevKernel::configure(const IImage *input, float *mean, uint64_t *global_sum, float *stddev, uint64_t *global_sum_squared)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON(nullptr == mean);
    ARM_COMPUTE_ERROR_ON(nullptr == global_sum);
    ARM_COMPUTE_ERROR_ON(stddev && nullptr == global_sum_squared);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);

    _input              = input;
    _mean               = mean;
    _stddev             = stddev;
    _global_sum         = global_sum;
    _global_sum_squared = global_sum_squared;

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    _border_size = BorderSize(ceil_to_multiple(input->info()->dimension(0), num_elems_processed_per_iteration) - input->info()->dimension(0));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration));

    INEKernel::configure(win);
}

void NEMeanStdDevKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    Iterator input(_input, window);

    uint64x1_t local_sum         = vdup_n_u64(0);
    uint64x1_t local_sum_squared = vdup_n_u64(0);

    if(_stddev != nullptr)
    {
        std::tie(local_sum, local_sum_squared) = accumulate<true>(window, input);
    }
    else
    {
        std::tie(local_sum, local_sum_squared) = accumulate<false>(window, input);
    }

    const float num_pixels = _input->info()->dimension(0) * _input->info()->dimension(1);

    // Merge sum and calculate mean and stddev
    std::unique_lock<arm_compute::Mutex> lock(_mtx);

    *_global_sum += vget_lane_u64(local_sum, 0);

    const float mean = *_global_sum / num_pixels;
    *_mean           = mean;

    if(_stddev != nullptr)
    {
        const uint64_t tmp_sum_squared = vget_lane_u64(local_sum_squared, 0);
        *_global_sum_squared += tmp_sum_squared;
        *_stddev = std::sqrt((*_global_sum_squared / num_pixels) - (mean * mean));
    }

    lock.unlock();
}
