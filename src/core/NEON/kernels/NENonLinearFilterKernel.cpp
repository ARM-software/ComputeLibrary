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
#include "arm_compute/core/NEON/kernels/NENonLinearFilterKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include <algorithm>
#include <arm_neon.h>
#include <array>
#include <tuple>
#include <utility>

namespace arm_compute
{
namespace
{
const uint8x16_t zero_u8 = vdupq_n_u8(0);

template <size_t columns>
inline uint8x8_t min_row(uint8x16_t row_data)
{
    uint8x8_t min = vget_low_u8(row_data);

    for(size_t c = 1; c < columns; ++c)
    {
        row_data = vextq_u8(row_data, zero_u8, 1);
        min      = vmin_u8(min, vget_low_u8(row_data));
    }

    return min;
}

template <size_t columns>
inline uint8x8_t max_row(uint8x16_t row_data)
{
    uint8x8_t max = vget_low_u8(row_data);

    for(size_t c = 1; c < columns; ++c)
    {
        row_data = vextq_u8(row_data, zero_u8, 1);
        max      = vmax_u8(max, vget_low_u8(row_data));
    }

    return max;
}

inline void sort(uint8x8_t &a, uint8x8_t &b)
{
    const uint8x8_t min = vmin_u8(a, b);
    const uint8x8_t max = vmax_u8(a, b);
    a                   = min;
    b                   = max;
}

// Sorting networks below were generated using http://pages.ripco.net/~jgamble/nw.html
// Calculations that do not affect the median were removed.
inline void sort5(uint8x8_t &p0, uint8x8_t &p1, uint8x8_t &p2, uint8x8_t &p3, uint8x8_t &p4)
{
    sort(p0, p1);
    sort(p2, p3);
    sort(p0, p2);
    sort(p1, p3);
    sort(p1, p2);
    sort(p0, p4);
    sort(p1, p4);
    sort(p2, p4);
}

inline void sort9(uint8x8_t &p0, uint8x8_t &p1, uint8x8_t &p2,
                  uint8x8_t &p3, uint8x8_t &p4, uint8x8_t &p5,
                  uint8x8_t &p6, uint8x8_t &p7, uint8x8_t &p8)
{
    sort(p1, p2);
    sort(p4, p5);
    sort(p7, p8);
    sort(p0, p1);
    sort(p3, p4);
    sort(p6, p7);
    sort(p1, p2);
    sort(p4, p5);
    sort(p7, p8);
    sort(p0, p3);
    sort(p5, p8);
    sort(p4, p7);
    sort(p3, p6);
    sort(p1, p4);
    sort(p2, p5);
    sort(p4, p7);
    sort(p4, p2);
    sort(p6, p4);
    sort(p4, p2);
}

inline void sort21(uint8x8_t p[21])
{
    sort(p[0], p[1]);
    sort(p[2], p[3]);
    sort(p[4], p[5]);
    sort(p[6], p[7]);
    sort(p[8], p[9]);
    sort(p[10], p[11]);
    sort(p[12], p[13]);
    sort(p[14], p[15]);
    sort(p[16], p[17]);
    sort(p[18], p[19]);
    sort(p[0], p[2]);
    sort(p[1], p[3]);
    sort(p[4], p[6]);
    sort(p[5], p[7]);
    sort(p[8], p[10]);
    sort(p[9], p[11]);
    sort(p[12], p[14]);
    sort(p[13], p[15]);
    sort(p[16], p[18]);
    sort(p[17], p[19]);
    sort(p[1], p[2]);
    sort(p[5], p[6]);
    sort(p[0], p[4]);
    sort(p[3], p[7]);
    sort(p[9], p[10]);
    sort(p[13], p[14]);
    sort(p[8], p[12]);
    sort(p[11], p[15]);
    sort(p[17], p[18]);
    sort(p[16], p[20]);
    sort(p[1], p[5]);
    sort(p[2], p[6]);
    sort(p[9], p[13]);
    sort(p[10], p[14]);
    sort(p[0], p[8]);
    sort(p[7], p[15]);
    sort(p[17], p[20]);
    sort(p[1], p[4]);
    sort(p[3], p[6]);
    sort(p[9], p[12]);
    sort(p[11], p[14]);
    sort(p[18], p[20]);
    sort(p[0], p[16]);
    sort(p[2], p[4]);
    sort(p[3], p[5]);
    sort(p[10], p[12]);
    sort(p[11], p[13]);
    sort(p[1], p[9]);
    sort(p[6], p[14]);
    sort(p[19], p[20]);
    sort(p[3], p[4]);
    sort(p[11], p[12]);
    sort(p[1], p[8]);
    sort(p[2], p[10]);
    sort(p[5], p[13]);
    sort(p[7], p[14]);
    sort(p[3], p[11]);
    sort(p[2], p[8]);
    sort(p[4], p[12]);
    sort(p[7], p[13]);
    sort(p[1], p[17]);
    sort(p[3], p[10]);
    sort(p[5], p[12]);
    sort(p[1], p[16]);
    sort(p[2], p[18]);
    sort(p[3], p[9]);
    sort(p[6], p[12]);
    sort(p[2], p[16]);
    sort(p[3], p[8]);
    sort(p[7], p[12]);
    sort(p[5], p[9]);
    sort(p[6], p[10]);
    sort(p[4], p[8]);
    sort(p[7], p[11]);
    sort(p[3], p[19]);
    sort(p[5], p[8]);
    sort(p[7], p[10]);
    sort(p[3], p[18]);
    sort(p[4], p[20]);
    sort(p[6], p[8]);
    sort(p[7], p[9]);
    sort(p[3], p[17]);
    sort(p[5], p[20]);
    sort(p[7], p[8]);
    sort(p[3], p[16]);
    sort(p[6], p[20]);
    sort(p[5], p[17]);
    sort(p[7], p[20]);
    sort(p[4], p[16]);
    sort(p[6], p[18]);
    sort(p[5], p[16]);
    sort(p[7], p[19]);
    sort(p[7], p[18]);
    sort(p[6], p[16]);
    sort(p[7], p[17]);
    sort(p[10], p[18]);
    sort(p[7], p[16]);
    sort(p[9], p[17]);
    sort(p[8], p[16]);
    sort(p[9], p[16]);
    sort(p[10], p[16]);
}

inline void sort25(uint8x8_t p[25])
{
    sort(p[1], p[2]);
    sort(p[0], p[1]);
    sort(p[1], p[2]);
    sort(p[4], p[5]);
    sort(p[3], p[4]);
    sort(p[4], p[5]);
    sort(p[0], p[3]);
    sort(p[2], p[5]);
    sort(p[2], p[3]);
    sort(p[1], p[4]);
    sort(p[1], p[2]);
    sort(p[3], p[4]);
    sort(p[7], p[8]);
    sort(p[6], p[7]);
    sort(p[7], p[8]);
    sort(p[10], p[11]);
    sort(p[9], p[10]);
    sort(p[10], p[11]);
    sort(p[6], p[9]);
    sort(p[8], p[11]);
    sort(p[8], p[9]);
    sort(p[7], p[10]);
    sort(p[7], p[8]);
    sort(p[9], p[10]);
    sort(p[0], p[6]);
    sort(p[4], p[10]);
    sort(p[4], p[6]);
    sort(p[2], p[8]);
    sort(p[2], p[4]);
    sort(p[6], p[8]);
    sort(p[1], p[7]);
    sort(p[5], p[11]);
    sort(p[5], p[7]);
    sort(p[3], p[9]);
    sort(p[3], p[5]);
    sort(p[7], p[9]);
    sort(p[1], p[2]);
    sort(p[3], p[4]);
    sort(p[5], p[6]);
    sort(p[7], p[8]);
    sort(p[9], p[10]);
    sort(p[13], p[14]);
    sort(p[12], p[13]);
    sort(p[13], p[14]);
    sort(p[16], p[17]);
    sort(p[15], p[16]);
    sort(p[16], p[17]);
    sort(p[12], p[15]);
    sort(p[14], p[17]);
    sort(p[14], p[15]);
    sort(p[13], p[16]);
    sort(p[13], p[14]);
    sort(p[15], p[16]);
    sort(p[19], p[20]);
    sort(p[18], p[19]);
    sort(p[19], p[20]);
    sort(p[21], p[22]);
    sort(p[23], p[24]);
    sort(p[21], p[23]);
    sort(p[22], p[24]);
    sort(p[22], p[23]);
    sort(p[18], p[21]);
    sort(p[20], p[23]);
    sort(p[20], p[21]);
    sort(p[19], p[22]);
    sort(p[22], p[24]);
    sort(p[19], p[20]);
    sort(p[21], p[22]);
    sort(p[23], p[24]);
    sort(p[12], p[18]);
    sort(p[16], p[22]);
    sort(p[16], p[18]);
    sort(p[14], p[20]);
    sort(p[20], p[24]);
    sort(p[14], p[16]);
    sort(p[18], p[20]);
    sort(p[22], p[24]);
    sort(p[13], p[19]);
    sort(p[17], p[23]);
    sort(p[17], p[19]);
    sort(p[15], p[21]);
    sort(p[15], p[17]);
    sort(p[19], p[21]);
    sort(p[13], p[14]);
    sort(p[15], p[16]);
    sort(p[17], p[18]);
    sort(p[19], p[20]);
    sort(p[21], p[22]);
    sort(p[23], p[24]);
    sort(p[0], p[12]);
    sort(p[8], p[20]);
    sort(p[8], p[12]);
    sort(p[4], p[16]);
    sort(p[16], p[24]);
    sort(p[12], p[16]);
    sort(p[2], p[14]);
    sort(p[10], p[22]);
    sort(p[10], p[14]);
    sort(p[6], p[18]);
    sort(p[6], p[10]);
    sort(p[10], p[12]);
    sort(p[1], p[13]);
    sort(p[9], p[21]);
    sort(p[9], p[13]);
    sort(p[5], p[17]);
    sort(p[13], p[17]);
    sort(p[3], p[15]);
    sort(p[11], p[23]);
    sort(p[11], p[15]);
    sort(p[7], p[19]);
    sort(p[7], p[11]);
    sort(p[11], p[13]);
    sort(p[11], p[12]);
}
} // namespace

NENonLinearFilterKernel::NENonLinearFilterKernel()
    : _border_width(0), _input(nullptr), _output(nullptr), _mask(nullptr), _pattern(MatrixPattern::BOX), _function(NonLinearFilterFunction::MIN), _func_idx(0), _border_size()
{
}

BorderSize NENonLinearFilterKernel::border_size() const
{
    return _border_size;
}

void NENonLinearFilterKernel::configure(const ITensor *input, ITensor *output, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask,
                                        bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(3 != mask_size && 5 != mask_size);
    ARM_COMPUTE_ERROR_ON(MatrixPattern::OTHER == pattern && nullptr == mask);

    // Set class variables
    _border_size = BorderSize(mask_size / 2);
    _input       = input;
    _output      = output;
    _mask        = mask;
    _pattern     = pattern;
    _function    = function;

    // Configure kernel window
    const unsigned int     num_elems_processed_per_iteration = (MatrixPattern::OTHER == pattern) ? 1 : 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;

    Window                 win = calculate_max_window(*input->info(), num_elems_processed_per_iteration, border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, mask_size),
                              output_access);
    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);

    // Define function index
    _func_idx = (3 == mask_size) ? 0 : 1;

    if(MatrixPattern::OTHER != pattern)
    {
        _func_idx = (_func_idx) * 3 + static_cast<unsigned int>(function);
    }
}

void NENonLinearFilterKernel::fill_mask(uint8_t *mask, int cols, int rows, MatrixPattern pattern)
{
    unsigned int v = 0;

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < cols; ++c, ++v)
        {
            uint8_t val = 0;

            switch(pattern)
            {
                case MatrixPattern::BOX:
                    val = 255;
                    break;
                case MatrixPattern::CROSS:
                    val = ((r == (rows / 2)) || (c == (cols / 2))) ? 255 : 0;
                    break;
                case MatrixPattern::DISK:
                    val = (((r - rows / 2.0f + 0.5f) * (r - rows / 2.0f + 0.5f)) / ((rows / 2.0f) * (rows / 2.0f)) + ((c - cols / 2.0f + 0.5f) * (c - cols / 2.0f + 0.5f)) / ((cols / 2.0f) *
                            (cols / 2.0f))) <= 1.0f ? 255 : 0;
                    break;
                default:
                    return;
            }

            mask[v] = val;
        }
    }
}

template <>
void NENonLinearFilterKernel::median_filter_box<3, 3>(const Window &win)
{
    Iterator input(_input, win);
    Iterator output(_output, win);

    const auto input_top_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-1, -1)));
    const auto input_mid_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-1, 0)));
    const auto input_bot_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-1, 1)));

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const uint8x16_t top_data = vld1q_u8(input_top_ptr + input.offset());
        const uint8x16_t mid_data = vld1q_u8(input_mid_ptr + input.offset());
        const uint8x16_t bot_data = vld1q_u8(input_bot_ptr + input.offset());

        uint8x8_t p0 = vget_low_u8(top_data);
        uint8x8_t p1 = vext_u8(vget_low_u8(top_data), vget_high_u8(top_data), 1);
        uint8x8_t p2 = vext_u8(vget_low_u8(top_data), vget_high_u8(top_data), 2);
        uint8x8_t p3 = vget_low_u8(mid_data);
        uint8x8_t p4 = vext_u8(vget_low_u8(mid_data), vget_high_u8(mid_data), 1);
        uint8x8_t p5 = vext_u8(vget_low_u8(mid_data), vget_high_u8(mid_data), 2);
        uint8x8_t p6 = vget_low_u8(bot_data);
        uint8x8_t p7 = vext_u8(vget_low_u8(bot_data), vget_high_u8(bot_data), 1);
        uint8x8_t p8 = vext_u8(vget_low_u8(bot_data), vget_high_u8(bot_data), 2);

        sort9(p0, p1, p2, p3, p4, p5, p6, p7, p8);

        vst1_u8(output.ptr(), p4);
    },
    input, output);
}
template <>
void NENonLinearFilterKernel::median_filter_box<5, 5>(const Window &win)
{
    Iterator input(_input, win);
    Iterator output(_output, win);

    const auto input_top2_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, -2)));
    const auto input_top_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, -1)));
    const auto input_mid_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 0)));
    const auto input_bot_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 1)));
    const auto input_bot2_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 2)));

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const uint8x16_t top2_data = vld1q_u8(input_top2_ptr + input.offset());
        const uint8x16_t top_data  = vld1q_u8(input_top_ptr + input.offset());
        const uint8x16_t mid_data  = vld1q_u8(input_mid_ptr + input.offset());
        const uint8x16_t bot_data  = vld1q_u8(input_bot_ptr + input.offset());
        const uint8x16_t bot2_data = vld1q_u8(input_bot2_ptr + input.offset());

        const uint8x8_t d[] =
        {
            vget_low_u8(top2_data),
            vget_high_u8(top2_data),
            vget_low_u8(top_data),
            vget_high_u8(top_data),
            vget_low_u8(mid_data),
            vget_high_u8(mid_data),
            vget_low_u8(bot_data),
            vget_high_u8(bot_data),
            vget_low_u8(bot2_data),
            vget_high_u8(bot2_data)
        };

        uint8x8_t p[25];
        for(unsigned int i = 0; i < 5; ++i)
        {
            const unsigned int idx_d = i * 2;
            const unsigned int idx_p = i * 5;

            p[idx_p]     = d[idx_d];
            p[idx_p + 1] = vext_u8(d[idx_d], d[idx_d + 1], 1);
            p[idx_p + 2] = vext_u8(d[idx_d], d[idx_d + 1], 2);
            p[idx_p + 3] = vext_u8(d[idx_d], d[idx_d + 1], 3);
            p[idx_p + 4] = vext_u8(d[idx_d], d[idx_d + 1], 4);
        }

        sort25(p);

        vst1_u8(output.ptr(), p[12]);
    },
    input, output);
}

template <int mask_w, int mask_h>
void NENonLinearFilterKernel::min_filter_box(const Window &win)
{
    static_assert(mask_w > 0, "Mask size must not be 0");
    static_assert(mask_h > 0, "Mask size must not be 0");

    Iterator input(_input, win);
    Iterator output(_output, win);

    const int k_row_half = mask_h / 2;
    const int k_col_half = mask_w / 2;

    // Set row pointers
    std::array<const unsigned char *, mask_h> input_ptrs{ {} };
    for(int i = -k_row_half; i <= k_row_half; ++i)
    {
        input_ptrs[k_row_half + i] = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-k_col_half, i));
    }

    execute_window_loop(win, [&](const Coordinates & id)
    {
        // Get min of rows
        uint8x16_t rows_min = vld1q_u8(input_ptrs[0] + input.offset());

        for(unsigned int r = 1; r < mask_h; ++r)
        {
            const uint8x16_t data = vld1q_u8(input_ptrs[r] + input.offset());
            rows_min              = vminq_u8(rows_min, data);
        }

        const uint8x8_t out = min_row<mask_w>(rows_min);

        // Store result as U8
        vst1_u8(output.ptr(), out);
    },
    input, output);
}

template <int mask_w, int mask_h>
void NENonLinearFilterKernel::max_filter_box(const Window &win)
{
    static_assert(mask_w > 0, "Mask size must not be 0");
    static_assert(mask_h > 0, "Mask size must not be 0");
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    Iterator input(_input, win);
    Iterator output(_output, win);

    const int k_row_half = mask_h / 2;
    const int k_col_half = mask_w / 2;

    // Set row pointers
    std::array<const unsigned char *, mask_h> input_ptrs{ {} };
    for(int i = -k_row_half; i <= k_row_half; ++i)
    {
        input_ptrs[k_row_half + i] = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-k_col_half, i));
    }

    execute_window_loop(win, [&](const Coordinates & id)
    {
        uint8x16_t rows_max = vld1q_u8(input_ptrs[0] + input.offset());

        // Get max of rows
        for(unsigned int r = 1; r < mask_h; ++r)
        {
            const uint8x16_t data = vld1q_u8(input_ptrs[r] + input.offset());
            rows_max              = vmaxq_u8(rows_max, data);
        }

        // Get max of columns
        const uint8x8_t out = max_row<mask_w>(rows_max);

        // Store result as U8
        vst1_u8(output.ptr(), out);
    },
    input, output);
}

template <>
void NENonLinearFilterKernel::median_filter_cross<3, 3>(const Window &win)
{
    Iterator input(_input, win);
    Iterator output(_output, win);

    const auto input_top_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(0, -1)));
    const auto input_mid_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-1, 0)));
    const auto input_bot_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(0, 1)));

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const uint8x8_t  top_data = vld1_u8(input_top_ptr + input.offset());
        const uint8x16_t mid_data = vld1q_u8(input_mid_ptr + input.offset());
        const uint8x8_t  bot_data = vld1_u8(input_bot_ptr + input.offset());

        uint8x8_t p0 = top_data;
        uint8x8_t p1 = vget_low_u8(mid_data);
        uint8x8_t p2 = vext_u8(vget_low_u8(mid_data), vget_high_u8(mid_data), 1);
        uint8x8_t p3 = vext_u8(vget_low_u8(mid_data), vget_high_u8(mid_data), 2);
        uint8x8_t p4 = bot_data;

        sort5(p0, p1, p2, p3, p4);

        vst1_u8(output.ptr(), p2);
    },
    input, output);
}

template <>
void NENonLinearFilterKernel::median_filter_cross<5, 5>(const Window &win)
{
    Iterator input(_input, win);
    Iterator output(_output, win);

    const auto input_top2_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(0, -2)));
    const auto input_top_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(0, -1)));
    const auto input_mid_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 0)));
    const auto input_bot_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(0, 1)));
    const auto input_bot2_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(0, 2)));

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const uint8x8_t  top2_data = vld1_u8(input_top2_ptr + input.offset());
        const uint8x8_t  top_data  = vld1_u8(input_top_ptr + input.offset());
        const uint8x16_t mid_data  = vld1q_u8(input_mid_ptr + input.offset());
        const uint8x8_t  bot_data  = vld1_u8(input_bot_ptr + input.offset());
        const uint8x8_t  bot2_data = vld1_u8(input_bot2_ptr + input.offset());

        uint8x8_t p0 = top2_data;
        uint8x8_t p1 = top_data;
        uint8x8_t p2 = vget_low_u8(mid_data);
        uint8x8_t p3 = vext_u8(vget_low_u8(mid_data), vget_high_u8(mid_data), 1);
        uint8x8_t p4 = vext_u8(vget_low_u8(mid_data), vget_high_u8(mid_data), 2);
        uint8x8_t p5 = vext_u8(vget_low_u8(mid_data), vget_high_u8(mid_data), 3);
        uint8x8_t p6 = vext_u8(vget_low_u8(mid_data), vget_high_u8(mid_data), 4);
        uint8x8_t p7 = bot_data;
        uint8x8_t p8 = bot2_data;

        sort9(p0, p1, p2, p3, p4, p5, p6, p7, p8);

        vst1_u8(output.ptr(), p4);
    },
    input, output);
}

template <int mask_w, int mask_h>
void NENonLinearFilterKernel::min_filter_cross(const Window &win)
{
    static_assert(mask_w > 0, "Mask size must not be 0");
    static_assert(mask_h > 0, "Mask size must not be 0");
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    Iterator input(_input, win);
    Iterator output(_output, win);

    const int k_row_half = mask_h / 2;
    const int k_col_half = mask_w / 2;

    const unsigned char *mid_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-k_col_half, 0));

    // Set row pointers
    std::array<const unsigned char *, mask_h> input_ptrs{ {} };
    for(int i = -k_row_half; i <= k_row_half; ++i)
    {
        input_ptrs[k_row_half + i] = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(0, i));
    }

    execute_window_loop(win, [&](const Coordinates & id)
    {
        uint8x8_t rows_min = vld1_u8(input_ptrs[0] + input.offset());

        // Get min of rows
        for(unsigned int r = 1; r < mask_h; ++r)
        {
            const uint8x8_t data = vld1_u8(input_ptrs[r] + input.offset());
            rows_min             = vmin_u8(rows_min, data);
        }

        // Get min of middle row
        const uint8x16_t data = vld1q_u8(mid_ptr + input.offset());
        uint8x8_t        out  = min_row<mask_w>(data);

        // Get final min
        out = vmin_u8(out, rows_min);

        // Store result as U8
        vst1_u8(output.ptr(), out);
    },
    input, output);
}

template <int mask_w, int mask_h>
void NENonLinearFilterKernel::max_filter_cross(const Window &win)
{
    static_assert(mask_w > 0, "Mask size must not be 0");
    static_assert(mask_h > 0, "Mask size must not be 0");
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    Iterator input(_input, win);
    Iterator output(_output, win);

    const int k_row_half = mask_h / 2;
    const int k_col_half = mask_w / 2;

    const unsigned char *mid_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-k_col_half, 0));

    // Set row pointers
    std::array<unsigned char *, mask_h> input_ptrs{ {} };
    for(int i = -k_row_half; i <= k_row_half; ++i)
    {
        input_ptrs[k_row_half + i] = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(0, i));
    }

    execute_window_loop(win, [&](const Coordinates & id)
    {
        uint8x8_t rows_max = vld1_u8(input_ptrs[0] + input.offset());

        // Get max of rows
        for(unsigned int r = 1; r < mask_h; ++r)
        {
            const uint8x8_t data = vld1_u8(input_ptrs[r] + input.offset());
            rows_max             = vmax_u8(rows_max, data);
        }

        // Get max of middle row
        const uint8x16_t data = vld1q_u8(mid_ptr + input.offset());
        uint8x8_t        out  = max_row<mask_w>(data);

        // Get final max
        out = vmax_u8(out, rows_max);

        // Store result as U8
        vst1_u8(output.ptr(), out);
    },
    input, output);
}

template <>
void NENonLinearFilterKernel::median_filter_disk<5, 5>(const Window &win)
{
    Iterator input(_input, win);
    Iterator output(_output, win);

    static const uint8x16_t zero           = vdupq_n_u8(0);
    const auto              input_top2_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, -2)));
    const auto              input_top_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, -1)));
    const auto              input_mid_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 0)));
    const auto              input_bot_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 1)));
    const auto              input_bot2_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 2)));

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const uint8x16_t top2_data = vextq_u8(vld1q_u8(input_top2_ptr + input.offset()), zero, 1);
        const uint8x16_t top_data  = vld1q_u8(input_top_ptr + input.offset());
        const uint8x16_t mid_data  = vld1q_u8(input_mid_ptr + input.offset());
        const uint8x16_t bot_data  = vld1q_u8(input_bot_ptr + input.offset());
        const uint8x16_t bot2_data = vextq_u8(vld1q_u8(input_bot2_ptr + input.offset()), zero, 1);

        uint8x8_t d[] =
        {
            vget_low_u8(top2_data),
            vget_high_u8(top2_data),
            vget_low_u8(top_data),
            vget_high_u8(top_data),
            vget_low_u8(mid_data),
            vget_high_u8(mid_data),
            vget_low_u8(bot_data),
            vget_high_u8(bot_data),
            vget_low_u8(bot2_data),
            vget_high_u8(bot2_data)
        };

        uint8x8_t p[21];
        p[0]  = d[0];
        p[1]  = vext_u8(d[0], d[1], 1);
        p[2]  = vext_u8(d[0], d[1], 2);
        p[18] = d[8];
        p[19] = vext_u8(d[8], d[9], 1);
        p[20] = vext_u8(d[8], d[9], 2);

        for(unsigned int i = 0; i < 3; ++i)
        {
            const unsigned int idx_d = 2 + i * 2;
            const unsigned int idx_p = 3 + i * 5;

            p[idx_p]     = d[idx_d];
            p[idx_p + 1] = vext_u8(d[idx_d], d[idx_d + 1], 1);
            p[idx_p + 2] = vext_u8(d[idx_d], d[idx_d + 1], 2);
            p[idx_p + 3] = vext_u8(d[idx_d], d[idx_d + 1], 3);
            p[idx_p + 4] = vext_u8(d[idx_d], d[idx_d + 1], 4);
        }

        sort21(p);

        vst1_u8(output.ptr(), p[10]);
    },
    input, output);
}

template <>
void NENonLinearFilterKernel::min_filter_disk<5, 5>(const Window &win)
{
    Iterator input(_input, win);
    Iterator output(_output, win);

    static const uint8x16_t zero           = vdupq_n_u8(0);
    const auto              input_top2_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, -2)));
    const auto              input_top_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, -1)));
    const auto              input_mid_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 0)));
    const auto              input_bot_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 1)));
    const auto              input_bot2_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 2)));

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const uint8x16_t top2_data = vextq_u8(vld1q_u8(input_top2_ptr + input.offset()), zero, 1);
        const uint8x16_t top_data  = vld1q_u8(input_top_ptr + input.offset());
        const uint8x16_t mid_data  = vld1q_u8(input_mid_ptr + input.offset());
        const uint8x16_t bot_data  = vld1q_u8(input_bot_ptr + input.offset());
        const uint8x16_t bot2_data = vextq_u8(vld1q_u8(input_bot2_ptr + input.offset()), zero, 1);

        const uint8x16_t rows_min_3 = vminq_u8(top2_data, bot2_data);
        uint8x16_t       rows_min_5 = vminq_u8(top_data, bot_data);
        rows_min_5                  = vminq_u8(rows_min_5, mid_data);

        const uint8x8_t out_3 = min_row<3>(rows_min_3);
        const uint8x8_t out_5 = min_row<5>(rows_min_5);

        vst1_u8(output.ptr(), vmin_u8(out_3, out_5));
    },
    input, output);
}

template <>
void NENonLinearFilterKernel::max_filter_disk<5, 5>(const Window &win)
{
    Iterator input(_input, win);
    Iterator output(_output, win);

    static const uint8x16_t zero           = vdupq_n_u8(0);
    const auto              input_top2_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, -2)));
    const auto              input_top_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, -1)));
    const auto              input_mid_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 0)));
    const auto              input_bot_ptr  = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 1)));
    const auto              input_bot2_ptr = static_cast<const unsigned char *>(_input->ptr_to_element(Coordinates(-2, 2)));

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const uint8x16_t top2_data = vextq_u8(vld1q_u8(input_top2_ptr + input.offset()), zero, 1);
        const uint8x16_t top_data  = vld1q_u8(input_top_ptr + input.offset());
        const uint8x16_t mid_data  = vld1q_u8(input_mid_ptr + input.offset());
        const uint8x16_t bot_data  = vld1q_u8(input_bot_ptr + input.offset());
        const uint8x16_t bot2_data = vextq_u8(vld1q_u8(input_bot2_ptr + input.offset()), zero, 1);

        const uint8x16_t rows_max_3 = vmaxq_u8(top2_data, bot2_data);
        uint8x16_t       rows_max_5 = vmaxq_u8(top_data, bot_data);
        rows_max_5                  = vmaxq_u8(rows_max_5, mid_data);

        const uint8x8_t out_3 = max_row<3>(rows_max_3);
        const uint8x8_t out_5 = max_row<5>(rows_max_5);

        vst1_u8(output.ptr(), vmax_u8(out_3, out_5));
    },
    input, output);
}

template <int mask_w, int mask_h>
void NENonLinearFilterKernel::non_linear_filter_generic(const Window &win)
{
    Iterator input(_input, win);
    Iterator output(_output, win);
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    const int     k_row_half = mask_h / 2;
    const int     k_col_half = mask_w / 2;
    constexpr int mask_size  = mask_w * mask_h;

    // Set row pointers
    std::array<unsigned char *, mask_h> input_ptrs{ {} };
    for(int i = -k_row_half; i <= k_row_half; ++i)
    {
        input_ptrs[k_row_half + i] = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-k_col_half, i));
    }

    execute_window_loop(win, [&](const Coordinates & id)
    {
        std::array<uint8_t, mask_size> vals{ {} };

        size_t v = 0;
        size_t m = 0;

        for(unsigned int r = 0; r < mask_h; ++r)
        {
            const auto in_ptr = static_cast<const uint8_t *>(input_ptrs[r] + input.offset());

            for(unsigned int c = 0; c < mask_w; ++c, ++m)
            {
                if(_mask[m] == 255)
                {
                    vals[v] = in_ptr[c];
                    ++v;
                }
            }
        }

        // Only do something if there is at least one non-zero element in the
        // mask
        if(v > 0)
        {
            std::sort(vals.begin(), vals.begin() + v);

            switch(_function)
            {
                case NonLinearFilterFunction::MIN:
                    *output.ptr() = vals[0];
                    break;
                case NonLinearFilterFunction::MAX:
                    *output.ptr() = vals[v - 1];
                    break;
                case NonLinearFilterFunction::MEDIAN:
                    *output.ptr() = vals[v / 2];
                    break;
                default:
                    break;
            }
        }
    },
    input, output);
}

void NENonLinearFilterKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    using NonLinearFilterFunction = void (NENonLinearFilterKernel::*)(const Window & window);

    // Function table for BOX pattern
    static const std::array<NonLinearFilterFunction, 6> func_table_box =
    {
        {
            &NENonLinearFilterKernel::median_filter_box<3, 3>,
            &NENonLinearFilterKernel::min_filter_box<3, 3>,
            &NENonLinearFilterKernel::max_filter_box<3, 3>,
            &NENonLinearFilterKernel::median_filter_box<5, 5>,
            &NENonLinearFilterKernel::min_filter_box<5, 5>,
            &NENonLinearFilterKernel::max_filter_box<5, 5>,
        }
    };

    // Function table for CROSS pattern
    static const std::array<NonLinearFilterFunction, 6> func_table_cross =
    {
        {
            &NENonLinearFilterKernel::median_filter_cross<3, 3>,
            &NENonLinearFilterKernel::min_filter_cross<3, 3>,
            &NENonLinearFilterKernel::max_filter_cross<3, 3>,
            &NENonLinearFilterKernel::median_filter_cross<5, 5>,
            &NENonLinearFilterKernel::min_filter_cross<5, 5>,
            &NENonLinearFilterKernel::max_filter_cross<5, 5>,
        }
    };

    // Function table for DISK pattern
    static const std::array<NonLinearFilterFunction, 6> func_table_disk =
    {
        {
            &NENonLinearFilterKernel::median_filter_box<3, 3>,
            &NENonLinearFilterKernel::min_filter_box<3, 3>,
            &NENonLinearFilterKernel::max_filter_box<3, 3>,
            &NENonLinearFilterKernel::median_filter_disk<5, 5>,
            &NENonLinearFilterKernel::min_filter_disk<5, 5>,
            &NENonLinearFilterKernel::max_filter_disk<5, 5>,
        }
    };

    // Function table for OTHER pattern
    static const std::array<NonLinearFilterFunction, 2> func_table_generic =
    {
        {
            &NENonLinearFilterKernel::non_linear_filter_generic<3, 3>,
            &NENonLinearFilterKernel::non_linear_filter_generic<5, 5>,
        }
    };

    switch(_pattern)
    {
        case MatrixPattern::BOX:
            ARM_COMPUTE_ERROR_ON(_func_idx >= func_table_box.size());
            (this->*func_table_box[_func_idx])(window);
            break;
        case MatrixPattern::CROSS:
            ARM_COMPUTE_ERROR_ON(_func_idx >= func_table_cross.size());
            (this->*func_table_cross[_func_idx])(window);
            break;
        case MatrixPattern::DISK:
            ARM_COMPUTE_ERROR_ON(_func_idx >= func_table_disk.size());
            (this->*func_table_disk[_func_idx])(window);
            break;
        case MatrixPattern::OTHER:
        default:
            ARM_COMPUTE_ERROR_ON(_func_idx >= func_table_generic.size());
            (this->*func_table_generic[_func_idx])(window);
            break;
    }
}
} // namespace arm_compute
