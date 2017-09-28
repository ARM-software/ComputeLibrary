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
#include "arm_compute/core/NEON/kernels/NEFastCornersKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"

#include <algorithm>
#include <arm_neon.h>
#include <cstddef>
#include <limits>

using namespace arm_compute;

NEFastCornersKernel::NEFastCornersKernel()
    : INEKernel(), _input(nullptr), _output(nullptr), _threshold(0), _non_max_suppression(false)
{
}

namespace
{
constexpr size_t PERMUTATIONS = 16;
constexpr size_t PERM_SIZE    = 16;

inline uint8x8x2_t create_permutation_index(size_t k)
{
    ARM_COMPUTE_ERROR_ON(k >= PERMUTATIONS);

    static const uint8_t permutations_table[PERMUTATIONS][PERM_SIZE]
    {
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 255, 255, 255, 255, 255, 255, 255 },
        { 15, 0, 1, 2, 3, 4, 5, 6, 7, 255, 255, 255, 255, 255, 255, 255 },
        { 14, 15, 0, 1, 2, 3, 4, 5, 6, 255, 255, 255, 255, 255, 255, 255 },
        { 13, 14, 15, 0, 1, 2, 3, 4, 5, 255, 255, 255, 255, 255, 255, 255 },
        { 12, 13, 14, 15, 0, 1, 2, 3, 4, 255, 255, 255, 255, 255, 255, 255 },
        { 11, 12, 13, 14, 15, 0, 1, 2, 3, 255, 255, 255, 255, 255, 255, 255 },
        { 10, 11, 12, 13, 14, 15, 0, 1, 2, 255, 255, 255, 255, 255, 255, 255 },
        { 9, 10, 11, 12, 13, 14, 15, 0, 1, 255, 255, 255, 255, 255, 255, 255 },
        { 8, 9, 10, 11, 12, 13, 14, 15, 0, 255, 255, 255, 255, 255, 255, 255 },
        { 7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 255, 255, 255, 255, 255 },
        { 6, 7, 8, 9, 10, 11, 12, 13, 14, 255, 255, 255, 255, 255, 255, 255 },
        { 5, 6, 7, 8, 9, 10, 11, 12, 13, 255, 255, 255, 255, 255, 255, 255 },
        { 4, 5, 6, 7, 8, 9, 10, 11, 12, 255, 255, 255, 255, 255, 255, 255 },
        { 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 255, 255, 255, 255, 255, 255 },
        { 2, 3, 4, 5, 6, 7, 8, 9, 10, 255, 255, 255, 255, 255, 255, 255 },
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 255, 255, 255, 255, 255, 255, 255 }

    };

    const uint8x8x2_t index =
    {
        {
            vld1_u8(permutations_table[k]),
            vld1_u8(permutations_table[k] + 8)
        }
    };

    return index;
}

inline uint8x8x4_t create_circle_index_register()
{
    /*
        This function creates the index registers to retrieve the 16 texels in the Bresenham circle of radius 3 with center in P.

        . . F 0 1 . . .
        . E . . . 2 . .
        D . . . . . 3 .
        C . . P . . 4 .
        B . . . . . 5 .
        . A . . . 6 . .
        . . 9 8 7 . . .

        Where . is an irrelevant texel value

        We want to retrieve all texels [0,F]

        The 4 registers in r will then be used to get these texels out of two tables in the function get_circle_texels()

        The first table holds the top 4 rows of texels
        . . F 0 1 . . .
        . E . . . 2 . .
        D . . . . . 3 .
        C . . P . . 4 .

        The second table the bottom 3 rows of texels
        B . . . . . 5 .
        . A . . . 6 . .
        . . 9 8 7 . . .

    */
    static const uint8_t top_right[8] =
    {
        /* The register r.val[0] will be used to retrieve these texels:
        . . . 0 1 . . .
        . . . . . 2 . .
        . . . . . . 3 .
        . . . . . . 4 .
        */
        3 /* top table, first row, elem 4, value 0 in the diagram above */,
        4 /* top table, first row, elem 5, value 1 in the diagram above */,
        13 /* top table, second row, elem 6, value 2 in the diagram above */,
        22 /* top table, third row, elem 7, value 3 in the diagram above*/,
        30 /* top table, fourth row, elem 7, value 4 in the diagram above*/,
        255,
        255,
        255
    };

    static const uint8_t bottom_right[8] =
    {
        /* The register r.val[1] will be used to retrieve these texels:
        . . . . . . 5 .
        . . . . . 6 . .
        . . . . 7 . . .
        */
        255,
        255,
        255,
        255,
        255,
        6 /* low table, first row, elem 7, value 5 in the diagram above*/,
        13 /* low table, second row, elem 6, value 6 in the diagram above*/,
        20 /* low table, third row, elem 5, value 7 in the diagram above*/
    };

    static const uint8_t top_left[8] =
    {
        /* The register r.val[2] will be used to retrieve these texels:
        . . F . . . . .
        . E . . . . . .
        D . . . . . . .
        C . . . . . . .
        */
        255,
        255,
        255,
        255,
        24 /* top table, fourth row, elem 1, value C in the diagram above */,
        16 /* top table, third row, elem 1, value D in the diagram above*/,
        9 /* top table, second row, elem 2, value E in the diagram above*/,
        2 /* top table, first row, elem 3, value F in the diagram above*/
    };

    static const uint8_t bottom_left[8] =
    {
        /* The register r.val[3] will be used to retrieve these texels:
        B . . . . . . .
        . A . . . . . .
        . . 9 8 . . . .
        */
        19 /* low table, third row, elem 4, value 8 in the diagram above */,
        18 /* low table, third row, elem 3, value 9 in the diagram above */,
        9 /* low table, second row, elem 2, value A in the diagram above */,
        0 /* low table, first row, elem 1, value B in the diagram above */,
        255,
        255,
        255,
        255
    };

    const uint8x8x4_t reg =
    {
        {
            vld1_u8(top_right),
            vld1_u8(bottom_right),
            vld1_u8(top_left),
            vld1_u8(bottom_left)
        }
    };

    return reg;
}

inline uint8x16_t get_circle_texels(const uint8x8x4_t &index, const uint8x8x4_t &tbl_hi, const uint8x8x3_t &tbl_lo)
{
    /*
        This function loads the 16 texels in the Bresenham circle of radius 3 into the register 'texels'.
        The parameter 'index' is an array of indices which was previously setup in setup_circle_index_register().
        tbl_hi and tbl_lo are the two tables holding the texels in the window [(-3,-3),(+3,+3)] for a given texel P
    */
    return vcombine_u8(vtbx3_u8(vtbl4_u8(tbl_hi, index.val[0]), tbl_lo, index.val[1]),
                       vtbx3_u8(vtbl4_u8(tbl_hi, index.val[2]), tbl_lo, index.val[3]));
}

inline uint8x16_t get_permutation_texels(const uint8x8x2_t &permutation_index, const uint8x8x2_t &tbl_circle)
{
    /*
        This function stores the 9 texels of a give permutation X in the neon register 'texels'

        'tbl_circle' is a LUT with the texels 0 to F

        . . F 0 1 . . .
        . E . . . 2 . .
        D . . . . . 3 .
        C . . P . . 4 .
        B . . . . . 5 .
        . A . . . 6 . .
        . . 9 8 7 . . .

        'permutation_index' is one of the permutations below:

        { 0, 1, 2, 3, 4, 5, 6, 7, 8},
        { F, 0, 1, 2, 3, 4, 5, 6, 7},
        { E, F, 0, 1, 2, 3, 4, 5, 6},
        { D, E, F, 0, 1, 2, 3, 4, 5},
        { C, D, E, F, 0, 1, 2, 3, 4},
        { B, C, D, E, F, 0, 1, 2, 3},
        { A, B, C, D, E, F, 0, 1, 2},
        { 9, A, B, C, D, E, F, 0, 1},
        { 8, 9, A, B, C, D, E, F, 0},
        { 7, 8, 9, A, B, C, D, E, F},
        { 6, 7, 8, 9, A, B, C, D, E},
        { 5, 6, 7, 8, 9, A, B, C, D},
        { 4, 5, 6, 7, 8, 9, A, B, C},
        { 3, 4, 5, 6, 7, 8, 9, A, B},
        { 2, 3, 4, 5, 6, 7, 8, 9, A},
        { 1, 2, 3, 4, 5, 6, 7, 8, 9},
    */
    static const uint8x8_t perm_right = vdup_n_u8(255); // init to 255 so that vtbx preserves the original values of the lanes

    return vcombine_u8(vtbl2_u8(tbl_circle, permutation_index.val[0]),
                       vtbx2_u8(perm_right, tbl_circle, permutation_index.val[1]));
}

inline bool is_permutation_brighter(const uint8x16_t &permutation, const uint8x16_t &pg)
{
    const uint8x16_t res_gt = vcgtq_u8(permutation, pg);

    return vget_lane_u64(vreinterpret_u64_u8(vand_u8(vget_high_u8(res_gt), vget_low_u8(res_gt))), 0) == std::numeric_limits<uint64_t>::max();
}

inline bool is_permutation_darker(const uint8x16_t &permutation, const uint8x16_t &pl)
{
    const uint8x16_t res_lt    = vcltq_u8(permutation, pl);
    const uint64x2_t u64res_lt = vreinterpretq_u64_u8(res_lt);
    const uint64_t   t3        = vgetq_lane_u64(u64res_lt, 0);
    const uint64_t   t4        = vgetq_lane_u64(u64res_lt, 1);

    return std::numeric_limits<uint64_t>::max() == t3 && 255 == t4;
}

inline bool is_permutation_corner(const uint8x16_t &permutation, const uint8x16_t &pg, const uint8x16_t &pl)
{
    return is_permutation_brighter(permutation, pg) || is_permutation_darker(permutation, pl);
}

inline bool point_is_fast_corner(uint8_t p, uint8_t threshold, const uint8x8x2_t &tbl_circle_texels, uint8x8x2_t perm_indices[PERMUTATIONS])
{
    /*
        This function determines whether the point 'p' is a corner.
    */
    uint8x16_t pg = vqaddq_u8(vdupq_n_u8(p), vdupq_n_u8(threshold));
    uint8x16_t pl = vqsubq_u8(vdupq_n_u8(p), vdupq_n_u8(threshold));

    bool corner_detected = false;

    for(size_t j = 0; !corner_detected && j < PERMUTATIONS; ++j)
    {
        const uint8x16_t pe_texels = get_permutation_texels(perm_indices[j], tbl_circle_texels);
        corner_detected            = is_permutation_corner(pe_texels, pg, pl);
    }

    return corner_detected;
}

inline uint8x8x2_t create_circle_tbl(const uint8_t *const __restrict buffer[7], size_t in_offset, const uint8x8x4_t &circle_index_r)
{
    /*
        This function builds a LUT holding the 16 texels in the Brensenham circle radius 3.
        circle_index_r is a vector of 4 registers to retrieve the texels from the two tables mentioned above.
    */

    //Load the texels in the window [(x-3,y-3),(x+3,y+3)].
    //The top 4 rows are loaded in tbl_hi and the low 3 rows in tbl_lo.
    //These two tables are then used to retrieve the texels in the Bresenham circle of radius 3.
    const uint8x8x4_t tbl_window_hi =
    {
        {
            vld1_u8(buffer[0] + in_offset),
            vld1_u8(buffer[1] + in_offset),
            vld1_u8(buffer[2] + in_offset),
            vld1_u8(buffer[3] + in_offset)
        }
    };

    const uint8x8x3_t tbl_window_lo =
    {
        {
            vld1_u8(buffer[4] + in_offset),
            vld1_u8(buffer[5] + in_offset),
            vld1_u8(buffer[6] + in_offset)
        }
    };

    const uint8x16_t circle_texels = get_circle_texels(circle_index_r, tbl_window_hi, tbl_window_lo);

    const uint8x8x2_t tbl_circle_texels =
    {
        {
            vget_low_u8(circle_texels),
            vget_high_u8(circle_texels)
        }
    };

    return tbl_circle_texels;
}

inline uint8_t get_point_score(uint8_t p, uint8_t tolerance, const uint8x8x2_t &tbl_circle, uint8x8x2_t perm_indices[PERMUTATIONS])
{
    uint8_t b = 255;
    uint8_t a = tolerance;

    while(b - a > 1)
    {
        const uint16_t ab = a + b;
        const uint8_t  c  = ab >> 1;

        if(point_is_fast_corner(p, c, tbl_circle, perm_indices))
        {
            a = c;
        }
        else
        {
            b = c;
        }
    }

    return a;
}
} // namespace

BorderSize NEFastCornersKernel::border_size() const
{
    return BorderSize(3);
}

void NEFastCornersKernel::configure(const IImage *input, IImage *output, uint8_t threshold, bool non_max_suppression, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MSG(border_undefined == false, "Not implemented");

    _input               = input;
    _output              = output;
    _threshold           = threshold;
    _non_max_suppression = non_max_suppression;

    constexpr unsigned int num_elems_processed_per_iteration = 1;
    constexpr unsigned int num_elems_read_per_iteration      = 8;
    constexpr unsigned int num_elems_written_per_iteration   = 1;
    constexpr unsigned int num_rows_read_per_iteration       = 7;

    // Configure kernel window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);
    AccessWindowRectangle  input_access(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

void NEFastCornersKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    std::array<uint8x8x2_t, PERMUTATIONS> perm_index{ {} };
    /*
        We use a LUT loaded with 7 rows of uint8_t from the input image [-3,-3]...[+3,+3] to retrieve the texels in the Brensenham circle radius 3 and put them in one neon register uint8x16_t.
        The three lines below setup the neon index registers to get these texels out from the table
    */
    const uint8x8x4_t circle_index_r = create_circle_index_register();
    /*
        We put the 16 texels (circle) in a LUT to easily generate all the permutations. The for block below setups the indices for each permutation.
    */
    for(size_t k = 0; k < PERMUTATIONS; ++k)
    {
        perm_index[k] = create_permutation_index(k);
    }

    Iterator in(_input, window);
    Iterator out(_output, window);

    const uint8_t *const __restrict in_row[7] =
    {
        _input->ptr_to_element(Coordinates(-3, -3)),
        _input->ptr_to_element(Coordinates(-3, -2)),
        _input->ptr_to_element(Coordinates(-3, -1)),
        _input->ptr_to_element(Coordinates(-3, 0)),
        _input->ptr_to_element(Coordinates(-3, 1)),
        _input->ptr_to_element(Coordinates(-3, 2)),
        _input->ptr_to_element(Coordinates(-3, 3))
    };

    auto is_rejected = [](uint8_t p, uint8_t q, uint8_t a, uint8_t b)
    {
        const bool p_is_in_ab = (a <= p) && (p <= b);
        const bool q_is_in_ab = (a <= q) && (q <= b);
        return p_is_in_ab && q_is_in_ab;
    };

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const size_t  in_offset = in.offset();
        const uint8_t p0        = *in.ptr();
        const uint8_t b         = std::min(p0 + _threshold, 255);
        const uint8_t a         = std::max(p0 - _threshold, 0);
        uint8_t       score     = 0;
        /*
            Fast check to discard points which cannot be corners and avoid the expensive computation of the potential 16 permutations

            pixels 1 and 9 are examined, if both I1 and I9 are within [Ip - t, Ip + t], then candidate p is not a corner.
        */
        const uint8_t p1 = (in_offset + in_row[0])[3];
        const uint8_t p9 = (in_offset + in_row[6])[3];

        if(!is_rejected(p1, p9, a, b))
        {
            /* pixels 5 and 13 are further examined to check whether three of them are brighter than Ip + t or darker than Ip - t */
            const uint8_t p5  = (in_offset + in_row[3])[6];
            const uint8_t p13 = (in_offset + in_row[3])[0];

            if(!is_rejected(p5, p13, a, b))
            {
                /* at this stage we use the full test with the 16 permutations to classify the point as corner or not */
                const uint8x8x2_t tbl_circle_texel = create_circle_tbl(in_row, in_offset, circle_index_r);

                if(point_is_fast_corner(p0, _threshold, tbl_circle_texel, perm_index.data()))
                {
                    if(_non_max_suppression)
                    {
                        score = get_point_score(p0, _threshold, tbl_circle_texel, perm_index.data());
                    }
                    else
                    {
                        score = 1;
                    }
                }
            }
        }

        *out.ptr() = score;
    },
    in, out);
}
