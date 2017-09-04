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

#include <arm_neon.h>
#include <climits>
#include <cstddef>

namespace arm_compute
{
NEMinMaxKernel::NEMinMaxKernel()
    : _func(), _input(nullptr), _min(), _max(), _min_init(), _max_init(), _mtx()
{
}

void NEMinMaxKernel::configure(const IImage *input, int32_t *min, int32_t *max)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON(nullptr == min);
    ARM_COMPUTE_ERROR_ON(nullptr == max);

    _input = input;
    _min   = min;
    _max   = max;

    switch(input->info()->format())
    {
        case Format::U8:
            _min_init = UCHAR_MAX;
            _max_init = 0;
            _func     = &NEMinMaxKernel::minmax_U8;
            break;
        case Format::S16:
            _min_init = SHRT_MAX;
            _max_init = SHRT_MIN;
            _func     = &NEMinMaxKernel::minmax_S16;
            break;
        default:
            ARM_COMPUTE_ERROR("You called with the wrong img formats");
            break;
    }

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration));

    INEKernel::configure(win);
}

void NEMinMaxKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}

void NEMinMaxKernel::reset()
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    *_min = _min_init;
    *_max = _max_init;
}

template <typename T>
void NEMinMaxKernel::update_min_max(const T min, const T max)
{
    std::lock_guard<std::mutex> lock(_mtx);

    if(min < *_min)
    {
        *_min = min;
    }

    if(max > *_max)
    {
        *_max = max;
    }
}

void NEMinMaxKernel::minmax_U8(const Window &win)
{
    uint8x8_t carry_min = vdup_n_u8(UCHAR_MAX);
    uint8x8_t carry_max = vdup_n_u8(0);

    Iterator input(_input, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const uint8x16_t pixels  = vld1q_u8(input.ptr());
        const uint8x8_t  tmp_min = vmin_u8(vget_high_u8(pixels), vget_low_u8(pixels));
        const uint8x8_t  tmp_max = vmax_u8(vget_high_u8(pixels), vget_low_u8(pixels));
        carry_min                = vmin_u8(tmp_min, carry_min);
        carry_max                = vmax_u8(tmp_max, carry_max);
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
    const uint8_t min_i = vget_lane_u8(carry_min, 0);
    const uint8_t max_i = vget_lane_u8(carry_max, 0);

    // Perform reduction of local min/max values
    update_min_max(min_i, max_i);
}

void NEMinMaxKernel::minmax_S16(const Window &win)
{
    int16x4_t carry_min = vdup_n_s16(SHRT_MAX);
    int16x4_t carry_max = vdup_n_s16(SHRT_MIN);

    Iterator input(_input, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const auto        in_ptr   = reinterpret_cast<const int16_t *>(input.ptr());
        const int16x8x2_t pixels   = vld2q_s16(in_ptr);
        const int16x8_t   tmp_min1 = vminq_s16(pixels.val[0], pixels.val[1]);
        const int16x8_t   tmp_max1 = vmaxq_s16(pixels.val[0], pixels.val[1]);
        const int16x4_t   tmp_min2 = vmin_s16(vget_high_s16(tmp_min1), vget_low_s16(tmp_min1));
        const int16x4_t   tmp_max2 = vmax_s16(vget_high_s16(tmp_max1), vget_low_s16(tmp_max1));
        carry_min                  = vmin_s16(tmp_min2, carry_min);
        carry_max                  = vmax_s16(tmp_max2, carry_max);
    },
    input);

    // Reduce result
    carry_min = vpmin_s16(carry_min, carry_min);
    carry_max = vpmax_s16(carry_max, carry_max);
    carry_min = vpmin_s16(carry_min, carry_min);
    carry_max = vpmax_s16(carry_max, carry_max);

    // Extract max/min values
    const int16_t min_i = vget_lane_s16(carry_min, 0);
    const int16_t max_i = vget_lane_s16(carry_max, 0);

    // Perform reduction of local min/max values
    update_min_max(min_i, max_i);
}

NEMinMaxLocationKernel::NEMinMaxLocationKernel()
    : _func(nullptr), _input(nullptr), _min(nullptr), _max(nullptr), _min_count(nullptr), _max_count(nullptr), _min_loc(nullptr), _max_loc(nullptr), _num_elems_processed_per_iteration(0)
{
}

bool NEMinMaxLocationKernel::is_parallelisable() const
{
    return false;
}

template <unsigned int...>
struct index_seq
{
    index_seq()                  = default;
    index_seq(const index_seq &) = default;
    index_seq &operator=(const index_seq &) = default;
    index_seq(index_seq &&) noexcept        = default;
    index_seq &operator=(index_seq &&) noexcept = default;
    virtual ~index_seq()                        = default;
};
template <unsigned int N, unsigned int... S>
struct gen_index_seq : gen_index_seq < N - 1, N - 1, S... >
{
};
template <unsigned int... S>
struct gen_index_seq<0u, S...> : index_seq<S...>
{
    using type = index_seq<S...>;
};

template <class T, unsigned int... N>
struct NEMinMaxLocationKernel::create_func_table<T, index_seq<N...>>
{
    static const NEMinMaxLocationKernel::MinMaxLocFunction func_table[sizeof...(N)];
};

template <class T, unsigned int... N>
const NEMinMaxLocationKernel::MinMaxLocFunction NEMinMaxLocationKernel::create_func_table<T, index_seq<N...>>::func_table[sizeof...(N)] =
{
    &NEMinMaxLocationKernel::minmax_loc<T, bool(N & 8), bool(N & 4), bool(N & 2), bool(N & 1)>...
};

void NEMinMaxLocationKernel::configure(const IImage *input, int32_t *min, int32_t *max,
                                       ICoordinates2DArray *min_loc, ICoordinates2DArray *max_loc,
                                       uint32_t *min_count, uint32_t *max_count)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(input, Format::U8, Format::S16);
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

    switch(input->info()->format())
    {
        case Format::U8:
            _func = create_func_table<uint8_t, gen_index_seq<16>::type>::func_table[table_idx];
            break;
        case Format::S16:
            _func = create_func_table<int16_t, gen_index_seq<16>::type>::func_table[table_idx];
            break;
        default:
            ARM_COMPUTE_ERROR("You called with the wrong img formats");
            break;
    }

    _num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(_num_elems_processed_per_iteration));

    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, _num_elems_processed_per_iteration));

    INEKernel::configure(win);
}

void NEMinMaxLocationKernel::run(const Window &window)
{
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

        size_t       min_count = 0;
        size_t       max_count = 0;
        unsigned int step      = _num_elems_processed_per_iteration;

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

        execute_window_loop(win, [&](const Coordinates & id)
        {
            auto    in_ptr = reinterpret_cast<const T *>(input.ptr());
            int32_t idx    = id.x();
            int32_t idy    = id.y();

            for(unsigned int i = 0; i < step; ++i)
            {
                const T       pixel = *in_ptr++;
                Coordinates2D p{ idx++, idy };

                if(count_min || loc_min)
                {
                    if(*_min == pixel)
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
                    if(*_max == pixel)
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
