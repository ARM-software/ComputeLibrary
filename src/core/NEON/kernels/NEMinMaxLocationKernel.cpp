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
#include "arm_compute/core/NEON/kernels/NEMinMaxLocationKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/utility.h"

#include <algorithm>
#include <arm_neon.h>
#include <climits>
#include <cstddef>

namespace arm_compute
{
NEMinMaxKernel::NEMinMaxKernel()
    : _func(), _input(nullptr), _min(), _max(), _mtx()
{
}

void NEMinMaxKernel::configure(const IImage *input, void *min, void *max)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16, DataType::F32);
    ARM_COMPUTE_ERROR_ON(nullptr == min);
    ARM_COMPUTE_ERROR_ON(nullptr == max);

    _input = input;
    _min   = min;
    _max   = max;

    switch(_input->info()->data_type())
    {
        case DataType::U8:
            _func = &NEMinMaxKernel::minmax_U8;
            break;
        case DataType::S16:
            _func = &NEMinMaxKernel::minmax_S16;
            break;
        case DataType::F32:
            _func = &NEMinMaxKernel::minmax_F32;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
            break;
    }

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 1;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    INEKernel::configure(win);
}

void NEMinMaxKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}

void NEMinMaxKernel::reset()
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    switch(_input->info()->data_type())
    {
        case DataType::U8:
            *static_cast<int32_t *>(_min) = UCHAR_MAX;
            *static_cast<int32_t *>(_max) = 0;
            break;
        case DataType::S16:
            *static_cast<int32_t *>(_min) = SHRT_MAX;
            *static_cast<int32_t *>(_max) = SHRT_MIN;
            break;
        case DataType::F32:
            *static_cast<float *>(_min) = std::numeric_limits<float>::max();
            *static_cast<float *>(_max) = std::numeric_limits<float>::lowest();
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
            break;
    }
}

template <typename T>
void NEMinMaxKernel::update_min_max(const T min, const T max)
{
    std::lock_guard<arm_compute::Mutex> lock(_mtx);

    using type = typename std::conditional<std::is_same<T, float>::value, float, int32_t>::type;

    auto min_ptr = static_cast<type *>(_min);
    auto max_ptr = static_cast<type *>(_max);

    if(min < *min_ptr)
    {
        *min_ptr = min;
    }

    if(max > *max_ptr)
    {
        *max_ptr = max;
    }
}

void NEMinMaxKernel::minmax_U8(Window win)
{
    uint8x8_t carry_min = vdup_n_u8(UCHAR_MAX);
    uint8x8_t carry_max = vdup_n_u8(0);

    uint8_t carry_max_scalar = 0;
    uint8_t carry_min_scalar = UCHAR_MAX;

    const int x_start = win.x().start();
    const int x_end   = win.x().end();

    // Handle X dimension manually to split into two loops
    // First one will use vector operations, second one processes the left over pixels
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        int x = x_start;

        // Vector loop
        for(; x <= x_end - 16; x += 16)
        {
            const uint8x16_t pixels  = vld1q_u8(input.ptr() + x);
            const uint8x8_t  tmp_min = vmin_u8(vget_high_u8(pixels), vget_low_u8(pixels));
            const uint8x8_t  tmp_max = vmax_u8(vget_high_u8(pixels), vget_low_u8(pixels));
            carry_min                = vmin_u8(tmp_min, carry_min);
            carry_max                = vmax_u8(tmp_max, carry_max);
        }

        // Process leftover pixels
        for(; x < x_end; ++x)
        {
            const uint8_t pixel = input.ptr()[x];
            carry_min_scalar    = std::min(pixel, carry_min_scalar);
            carry_max_scalar    = std::max(pixel, carry_max_scalar);
        }
    },
    input);

    // Reduce result
    carry_min = vpmin_u8(carry_min, carry_min);
    carry_max = vpmax_u8(carry_max, carry_max);
    carry_min = vpmin_u8(carry_min, carry_min);
    carry_max = vpmax_u8(carry_max, carry_max);
    carry_min = vpmin_u8(carry_min, carry_min);
    carry_max = vpmax_u8(carry_max, carry_max);

    // Extract max/min values
    const uint8_t min_i = std::min(vget_lane_u8(carry_min, 0), carry_min_scalar);
    const uint8_t max_i = std::max(vget_lane_u8(carry_max, 0), carry_max_scalar);

    // Perform reduction of local min/max values
    update_min_max(min_i, max_i);
}

void NEMinMaxKernel::minmax_S16(Window win)
{
    int16x4_t carry_min = vdup_n_s16(SHRT_MAX);
    int16x4_t carry_max = vdup_n_s16(SHRT_MIN);

    int16_t carry_max_scalar = SHRT_MIN;
    int16_t carry_min_scalar = SHRT_MAX;

    const int x_start = win.x().start();
    const int x_end   = win.x().end();

    // Handle X dimension manually to split into two loops
    // First one will use vector operations, second one processes the left over pixels
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        int        x      = x_start;
        const auto in_ptr = reinterpret_cast<const int16_t *const>(input.ptr());

        // Vector loop
        for(; x <= x_end - 16; x += 16)
        {
            const int16x8x2_t pixels   = vld2q_s16(in_ptr + x);
            const int16x8_t   tmp_min1 = vminq_s16(pixels.val[0], pixels.val[1]);
            const int16x8_t   tmp_max1 = vmaxq_s16(pixels.val[0], pixels.val[1]);
            const int16x4_t   tmp_min2 = vmin_s16(vget_high_s16(tmp_min1), vget_low_s16(tmp_min1));
            const int16x4_t   tmp_max2 = vmax_s16(vget_high_s16(tmp_max1), vget_low_s16(tmp_max1));
            carry_min                  = vmin_s16(tmp_min2, carry_min);
            carry_max                  = vmax_s16(tmp_max2, carry_max);
        }

        // Process leftover pixels
        for(; x < x_end; ++x)
        {
            const int16_t pixel = in_ptr[x];
            carry_min_scalar    = std::min(pixel, carry_min_scalar);
            carry_max_scalar    = std::max(pixel, carry_max_scalar);
        }

    },
    input);

    // Reduce result
    carry_min = vpmin_s16(carry_min, carry_min);
    carry_max = vpmax_s16(carry_max, carry_max);
    carry_min = vpmin_s16(carry_min, carry_min);
    carry_max = vpmax_s16(carry_max, carry_max);

    // Extract max/min values
    const int16_t min_i = std::min(vget_lane_s16(carry_min, 0), carry_min_scalar);
    const int16_t max_i = std::max(vget_lane_s16(carry_max, 0), carry_max_scalar);

    // Perform reduction of local min/max values
    update_min_max(min_i, max_i);
}

void NEMinMaxKernel::minmax_F32(Window win)
{
    float32x2_t carry_min = vdup_n_f32(std::numeric_limits<float>::max());
    float32x2_t carry_max = vdup_n_f32(std::numeric_limits<float>::lowest());

    float carry_min_scalar = std::numeric_limits<float>::max();
    float carry_max_scalar = std::numeric_limits<float>::lowest();

    const int x_start = win.x().start();
    const int x_end   = win.x().end();

    // Handle X dimension manually to split into two loops
    // First one will use vector operations, second one processes the left over pixels
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        int        x      = x_start;
        const auto in_ptr = reinterpret_cast<const float *const>(input.ptr());

        // Vector loop
        for(; x <= x_end - 8; x += 8)
        {
            const float32x4x2_t pixels   = vld2q_f32(in_ptr + x);
            const float32x4_t   tmp_min1 = vminq_f32(pixels.val[0], pixels.val[1]);
            const float32x4_t   tmp_max1 = vmaxq_f32(pixels.val[0], pixels.val[1]);
            const float32x2_t   tmp_min2 = vmin_f32(vget_high_f32(tmp_min1), vget_low_f32(tmp_min1));
            const float32x2_t   tmp_max2 = vmax_f32(vget_high_f32(tmp_max1), vget_low_f32(tmp_max1));
            carry_min                    = vmin_f32(tmp_min2, carry_min);
            carry_max                    = vmax_f32(tmp_max2, carry_max);
        }

        // Process leftover pixels
        for(; x < x_end; ++x)
        {
            const float pixel = in_ptr[x];
            carry_min_scalar  = std::min(pixel, carry_min_scalar);
            carry_max_scalar  = std::max(pixel, carry_max_scalar);
        }

    },
    input);

    // Reduce result
    carry_min = vpmin_f32(carry_min, carry_min);
    carry_max = vpmax_f32(carry_max, carry_max);
    carry_min = vpmin_f32(carry_min, carry_min);
    carry_max = vpmax_f32(carry_max, carry_max);

    // Extract max/min values
    const float min_i = std::min(vget_lane_f32(carry_min, 0), carry_min_scalar);
    const float max_i = std::max(vget_lane_f32(carry_max, 0), carry_max_scalar);

    // Perform reduction of local min/max values
    update_min_max(min_i, max_i);
}

NEMinMaxLocationKernel::NEMinMaxLocationKernel()
    : _func(nullptr), _input(nullptr), _min(nullptr), _max(nullptr), _min_count(nullptr), _max_count(nullptr), _min_loc(nullptr), _max_loc(nullptr)
{
}

bool NEMinMaxLocationKernel::is_parallelisable() const
{
    return false;
}

template <class T, std::size_t... N>
struct NEMinMaxLocationKernel::create_func_table<T, utility::index_sequence<N...>>
{
    static const NEMinMaxLocationKernel::MinMaxLocFunction func_table[sizeof...(N)];
};

template <class T, std::size_t... N>
const NEMinMaxLocationKernel::MinMaxLocFunction NEMinMaxLocationKernel::create_func_table<T, utility::index_sequence<N...>>::func_table[sizeof...(N)] =
{
    &NEMinMaxLocationKernel::minmax_loc<T, bool(N & 8), bool(N & 4), bool(N & 2), bool(N & 1)>...
};

void NEMinMaxLocationKernel::configure(const IImage *input, void *min, void *max,
                                       ICoordinates2DArray *min_loc, ICoordinates2DArray *max_loc,
                                       uint32_t *min_count, uint32_t *max_count)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16, DataType::F32);
    ARM_COMPUTE_ERROR_ON(nullptr == min);
    ARM_COMPUTE_ERROR_ON(nullptr == max);

    _input     = input;
    _min       = min;
    _max       = max;
    _min_count = min_count;
    _max_count = max_count;
    _min_loc   = min_loc;
    _max_loc   = max_loc;

    unsigned int count_min = (nullptr != min_count ? 1 : 0);
    unsigned int count_max = (nullptr != max_count ? 1 : 0);
    unsigned int loc_min   = (nullptr != min_loc ? 1 : 0);
    unsigned int loc_max   = (nullptr != max_loc ? 1 : 0);

    unsigned int table_idx = (count_min << 3) | (count_max << 2) | (loc_min << 1) | loc_max;

    switch(input->info()->data_type())
    {
        case DataType::U8:
            _func = create_func_table<uint8_t, utility::index_sequence_t<16>>::func_table[table_idx];
            break;
        case DataType::S16:
            _func = create_func_table<int16_t, utility::index_sequence_t<16>>::func_table[table_idx];
            break;
        case DataType::F32:
            _func = create_func_table<float, utility::index_sequence_t<16>>::func_table[table_idx];
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
            break;
    }

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration));

    INEKernel::configure(win);
}

void NEMinMaxLocationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}

template <class T, bool count_min, bool count_max, bool loc_min, bool loc_max>
void NEMinMaxLocationKernel::minmax_loc(const Window &win)
{
    if(count_min || count_max || loc_min || loc_max)
    {
        Iterator input(_input, win);

        size_t min_count = 0;
        size_t max_count = 0;

        // Clear min location array
        if(loc_min)
        {
            _min_loc->clear();
        }

        // Clear max location array
        if(loc_max)
        {
            _max_loc->clear();
        }

        using type = typename std::conditional<std::is_same<T, float>::value, float, int32_t>::type;

        auto min_ptr = static_cast<type *>(_min);
        auto max_ptr = static_cast<type *>(_max);

        execute_window_loop(win, [&](const Coordinates & id)
        {
            auto    in_ptr = reinterpret_cast<const T *>(input.ptr());
            int32_t idx    = id.x();
            int32_t idy    = id.y();

            const T       pixel = *in_ptr;
            Coordinates2D p{ idx, idy };

            if(count_min || loc_min)
            {
                if(*min_ptr == pixel)
                {
                    if(count_min)
                    {
                        ++min_count;
                    }

                    if(loc_min)
                    {
                        _min_loc->push_back(p);
                    }
                }
            }

            if(count_max || loc_max)
            {
                if(*max_ptr == pixel)
                {
                    if(count_max)
                    {
                        ++max_count;
                    }

                    if(loc_max)
                    {
                        _max_loc->push_back(p);
                    }
                }
            }
        },
        input);

        if(count_min)
        {
            *_min_count = min_count;
        }

        if(count_max)
        {
            *_max_count = max_count;
        }
    }
}
} // namespace arm_compute
