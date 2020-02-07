/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpOffsetContributionOutputStageKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <map>

namespace arm_compute
{
class Coordinates;

namespace
{
inline int32x4x4_t load_results_input(const Iterator &mm_result_it, int32_t x)
{
    return
    {
        {
            vld1q_s32(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x + 0),
            vld1q_s32(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x + 4),
            vld1q_s32(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x + 8),
            vld1q_s32(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x + 12)
        }
    };
}

inline int32x4x4_t load(const int32_t *ptr, int32_t x)
{
    return
    {
        {
            vld1q_s32(ptr + x + 0),
            vld1q_s32(ptr + x + 4),
            vld1q_s32(ptr + x + 8),
            vld1q_s32(ptr + x + 12)
        }
    };
}

inline int32x4x4_t add_s32(int32x4x4_t a, int32x4_t b)
{
    return
    {
        {
            vaddq_s32(a.val[0], b),
            vaddq_s32(a.val[1], b),
            vaddq_s32(a.val[2], b),
            vaddq_s32(a.val[3], b)
        }
    };
}

inline int32x4x4_t add_s32(int32x4x4_t a, int32x4x4_t b)
{
    return
    {
        {
            vaddq_s32(a.val[0], b.val[0]),
            vaddq_s32(a.val[1], b.val[1]),
            vaddq_s32(a.val[2], b.val[2]),
            vaddq_s32(a.val[3], b.val[3])
        }
    };
}

inline int32x4x4_t mul_s32(int32x4x4_t &a, int32_t mul_scalar)
{
    return
    {
        {
            vmulq_n_s32(a.val[0], mul_scalar),
            vmulq_n_s32(a.val[1], mul_scalar),
            vmulq_n_s32(a.val[2], mul_scalar),
            vmulq_n_s32(a.val[3], mul_scalar)
        }
    };
}

inline int32x4x4_t mul_s32(int32x4x4_t &a, const int32_t *multilpier)
{
    return
    {
        {
            vmulq_s32(a.val[0], vld1q_s32(multilpier)),
            vmulq_s32(a.val[1], vld1q_s32(multilpier + 4)),
            vmulq_s32(a.val[2], vld1q_s32(multilpier + 8)),
            vmulq_s32(a.val[3], vld1q_s32(multilpier + 12))
        }
    };
}

inline int32x4x4_t get_a_offset(const int32_t *vector_sum_col_ptr, int32_t a_offset, int32_t x)
{
    int32x4x4_t a_offset_term_s32 = load(vector_sum_col_ptr, x);

    a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], a_offset);
    a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], a_offset);
    a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], a_offset);
    a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], a_offset);
    return a_offset_term_s32;
}

inline int32x4_t get_b_offset(const int32_t *vector_sum_row_ptr, int32_t b_offset)
{
    int32x4_t b_offset_term_s32 = vld1q_dup_s32(vector_sum_row_ptr);
    b_offset_term_s32           = vmulq_n_s32(b_offset_term_s32, b_offset);
    return b_offset_term_s32;
}

inline int32x4x4_t get_k_offset(int32_t k_offset)
{
    return
    {
        {
            vdupq_n_s32(k_offset),
            vdupq_n_s32(k_offset),
            vdupq_n_s32(k_offset),
            vdupq_n_s32(k_offset)
        }
    };
}

template <bool    is_bounded_relu>
inline uint8x16_t finalize_quantization_floating_point(int32x4x4_t &in_s32, int32x4_t result_shift_s32, uint8x16_t min_u8, uint8x16_t max_u8)
{
    const static int32x4_t zero_s32 = vdupq_n_s32(0);

    // Shift final result (negative value shift right)
    in_s32.val[0] = vshlq_s32(in_s32.val[0], result_shift_s32);
    in_s32.val[1] = vshlq_s32(in_s32.val[1], result_shift_s32);
    in_s32.val[2] = vshlq_s32(in_s32.val[2], result_shift_s32);
    in_s32.val[3] = vshlq_s32(in_s32.val[3], result_shift_s32);

    // Saturate negative values
    in_s32.val[0] = vmaxq_s32(in_s32.val[0], zero_s32);
    in_s32.val[1] = vmaxq_s32(in_s32.val[1], zero_s32);
    in_s32.val[2] = vmaxq_s32(in_s32.val[2], zero_s32);
    in_s32.val[3] = vmaxq_s32(in_s32.val[3], zero_s32);

    // Convert S32 to S16
    const int16x8x2_t in_s16 =
    {
        {
            vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
            vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
        }
    };

    // Convert S16 to U8
    uint8x16_t out_u8 = vcombine_u8(vqmovun_s16(in_s16.val[0]), vqmovun_s16(in_s16.val[1]));

    if(is_bounded_relu)
    {
        out_u8 = vmaxq_u8(out_u8, min_u8);
        out_u8 = vminq_u8(out_u8, max_u8);
    }

    return out_u8;
}

template <bool   is_bounded_relu>
inline int8x16_t finalize_quantization_floating_point(int32x4x4_t &in_s32, int32x4_t result_shift_s32, int8x16_t min_s8, int8x16_t max_s8)
{
    const static int32x4_t zero_s32 = vdupq_n_s32(0);

    // Shift final result (negative value shift right)
    in_s32.val[0] = vshlq_s32(in_s32.val[0], result_shift_s32);
    in_s32.val[1] = vshlq_s32(in_s32.val[1], result_shift_s32);
    in_s32.val[2] = vshlq_s32(in_s32.val[2], result_shift_s32);
    in_s32.val[3] = vshlq_s32(in_s32.val[3], result_shift_s32);

    // Saturate negative values
    in_s32.val[0] = vmaxq_s32(in_s32.val[0], zero_s32);
    in_s32.val[1] = vmaxq_s32(in_s32.val[1], zero_s32);
    in_s32.val[2] = vmaxq_s32(in_s32.val[2], zero_s32);
    in_s32.val[3] = vmaxq_s32(in_s32.val[3], zero_s32);

    // Convert S32 to S16
    const int16x8x2_t in_s16 =
    {
        {
            vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
            vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
        }
    };

    // Convert S16 to S8
    int8x16_t out_s8 = vcombine_s8(vqmovn_s16(in_s16.val[0]), vqmovn_s16(in_s16.val[1]));

    if(is_bounded_relu)
    {
        out_s8 = vmaxq_s8(out_s8, min_s8);
        out_s8 = vminq_s8(out_s8, max_s8);
    }

    return out_s8;
}

template <bool   is_bounded_relu>
inline int8x16_t finalize_quantization_floating_point(int32x4x4_t &in_s32, int32x4x4_t result_shift_s32, int8x16_t min_s8, int8x16_t max_s8)
{
    const static int32x4_t zero_s32 = vdupq_n_s32(0);

    // Shift final result (negative value shift right)
    in_s32.val[0] = vshlq_s32(in_s32.val[0], vnegq_s32(result_shift_s32.val[0]));
    in_s32.val[1] = vshlq_s32(in_s32.val[1], vnegq_s32(result_shift_s32.val[1]));
    in_s32.val[2] = vshlq_s32(in_s32.val[2], vnegq_s32(result_shift_s32.val[2]));
    in_s32.val[3] = vshlq_s32(in_s32.val[3], vnegq_s32(result_shift_s32.val[3]));

    // Saturate negative values
    in_s32.val[0] = vmaxq_s32(in_s32.val[0], zero_s32);
    in_s32.val[1] = vmaxq_s32(in_s32.val[1], zero_s32);
    in_s32.val[2] = vmaxq_s32(in_s32.val[2], zero_s32);
    in_s32.val[3] = vmaxq_s32(in_s32.val[3], zero_s32);

    // Convert S32 to S16
    const int16x8x2_t in_s16 =
    {
        {
            vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
            vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
        }
    };

    // Convert S16 to S8
    int8x16_t out_s8 = vcombine_s8(vqmovn_s16(in_s16.val[0]), vqmovn_s16(in_s16.val[1]));

    if(is_bounded_relu)
    {
        out_s8 = vmaxq_s8(out_s8, min_s8);
        out_s8 = vminq_s8(out_s8, max_s8);
    }

    return out_s8;
}

template <typename T>
struct VectorTyper
{
    using stype = T;
    using vtype = typename wrapper::traits::neon_bitvector_t<T, wrapper::traits::BitWidth::W128>;
};

inline Window get_win_vector_sum(const Window &window)
{
    Window win_vector_sum(window);
    win_vector_sum.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_vector_sum.set(Window::DimZ, Window::Dimension(0, 0, 0));
    return win_vector_sum;
}

inline Iterator get_vector_sum_col_it(const Window &window, const ITensor *vector_sum_col)
{
    Iterator vector_sum_col_it(vector_sum_col, get_win_vector_sum(window));
    return vector_sum_col_it;
}

inline Iterator get_vector_sum_row_it(const Window &window, const ITensor *vector_sum_row)
{
    Window win_vector_sum_row = get_win_vector_sum(window);
    win_vector_sum_row.set(Window::DimX, Window::Dimension(0, 0, 0));
    Iterator vector_sum_row_it(vector_sum_row, win_vector_sum_row);
    return vector_sum_row_it;
}

inline Iterator get_bias_it(const Window &window, const ITensor *bias)
{
    Window win_bias(window);
    win_bias.set(Window::DimY, Window::Dimension(0, 1, 1));
    win_bias.set(Window::DimZ, Window::Dimension(0, 1, 1));
    Iterator bias_it(bias, win_bias);
    return bias_it;
}

template <typename VT, bool has_a_offset, bool has_b_offset, bool has_bias, bool is_bounded_relu, bool is_fixed_point>
inline void run_offset_contribution_output_stage_window(const int32_t *vector_sum_col_ptr, const int32_t *vector_sum_row_ptr, const int32_t *bias_ptr, Iterator mm_result_it, Iterator out_it,
                                                        const int32x4_t result_offset_s32, const int32x4_t result_shift_s32,
                                                        typename VT::vtype min_vec, typename VT::vtype max_vec,
                                                        int32_t a_offset, int32_t b_offset, int32_t k_offset,
                                                        int32_t multiplier, int32_t shift, int32_t offset, int32_t min_bound, int32_t max_bound,
                                                        int window_step_x, int window_start_x, int window_end_x)
{
    int32x4x4_t offset_term_s32 = { 0, 0, 0, 0 };
    if(!is_fixed_point)
    {
        // Combine quantization offset with other offsets.
        offset_term_s32 = add_s32(offset_term_s32, result_offset_s32);
    }
    if(has_a_offset && has_b_offset)
    {
        offset_term_s32 = add_s32(offset_term_s32, get_k_offset(k_offset));
    }
    if(has_b_offset)
    {
        offset_term_s32 = add_s32(offset_term_s32, get_b_offset(vector_sum_row_ptr, b_offset));
    }

    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        int32x4x4_t in_s32 = load_results_input(mm_result_it, x);

        if(has_a_offset)
        {
            in_s32 = add_s32(in_s32, get_a_offset(vector_sum_col_ptr, a_offset, x));
        }
        if(has_bias)
        {
            in_s32 = add_s32(in_s32, load(bias_ptr, x));
        }
        if(!is_fixed_point || has_b_offset)
        {
            in_s32 = add_s32(in_s32, offset_term_s32);
        }
        if(!is_fixed_point)
        {
            in_s32 = mul_s32(in_s32, multiplier);
        }

        if(is_fixed_point)
        {
            wrapper::vstore(reinterpret_cast<typename VT::stype *>(out_it.ptr() + x),
                            finalize_quantization<is_bounded_relu>(in_s32, multiplier, shift, result_offset_s32, min_vec, max_vec));
        }
        else
        {
            wrapper::vstore(reinterpret_cast<typename VT::stype *>(out_it.ptr() + x),
                            finalize_quantization_floating_point<is_bounded_relu>(in_s32, result_shift_s32, min_vec, max_vec));
        }
    }
    // Compute left-over elements
    for(; x < window_end_x; ++x)
    {
        int32_t in_value = *(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x) + wrapper::vgetlane(offset_term_s32.val[0], 0);

        if(has_a_offset)
        {
            in_value += (*(vector_sum_col_ptr + x) * a_offset);
        }
        if(has_bias)
        {
            in_value += *(bias_ptr + x);
        }

        if(is_fixed_point)
        {
            // Finalize and store the result
            *reinterpret_cast<typename VT::stype *>(out_it.ptr() + x) = finalize_quantization<is_bounded_relu>(in_value, multiplier, shift, offset,
                                                                                                               static_cast<typename VT::stype>(min_bound),
                                                                                                               static_cast<typename VT::stype>(max_bound));
        }
        else
        {
            // Finalize quantization
            in_value = (in_value * multiplier) >> shift;

            // Bound and store the result
            if(is_bounded_relu)
            {
                in_value = static_cast<typename VT::stype>(std::max<int32_t>(min_bound, std::min<int32_t>(max_bound, in_value)));
            }
            *reinterpret_cast<typename VT::stype *>(out_it.ptr() + x) = static_cast<typename VT::stype>(std::max<int32_t>(static_cast<int32_t>(std::numeric_limits<typename VT::stype>::lowest()),
                                                                                                                          std::min<int32_t>(static_cast<int32_t>(std::numeric_limits<typename VT::stype>::max()), in_value)));
        }
    }
}

template <bool has_a_offset, bool has_bias, bool is_bounded_relu, bool is_fixed_point>
inline void run_offset_contribution_output_stage_window_symm(const int32_t *vector_sum_col_ptr, const int32_t *bias_ptr, Iterator mm_result_it, Iterator out_it,
                                                             const int32_t *result_multipliers, const int32_t *result_shifts,
                                                             const int32x4_t result_offset, int8x16_t min_s8, int8x16_t max_s8,
                                                             int32_t a_offset, int32_t offset, int32_t min_bound, int32_t max_bound,
                                                             int window_step_x, int window_start_x, int window_end_x)
{
    int32x4x4_t offset_term_s32 = { 0, 0, 0, 0 };
    if(!is_fixed_point)
    {
        // Combine quantization offset with other offsets.
        offset_term_s32 = add_s32(offset_term_s32, result_offset);
    }

    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        int32x4x4_t in_s32 = load_results_input(mm_result_it, x);

        if(has_a_offset)
        {
            in_s32 = add_s32(in_s32, get_a_offset(vector_sum_col_ptr, a_offset, x));
        }
        if(has_bias)
        {
            in_s32 = add_s32(in_s32, load(bias_ptr, x));
        }
        if(!is_fixed_point)
        {
            in_s32 = add_s32(in_s32, offset_term_s32);
            in_s32 = mul_s32(in_s32, result_multipliers + x);
        }

        if(is_fixed_point)
        {
            vst1q_s8(reinterpret_cast<int8_t *>(out_it.ptr() + x), finalize_quantization_symm<is_bounded_relu>(in_s32, load(result_multipliers, x), load(result_shifts, x), result_offset, min_s8, max_s8));
        }
        else
        {
            vst1q_s8(reinterpret_cast<int8_t *>(out_it.ptr() + x), finalize_quantization_floating_point<is_bounded_relu>(in_s32, load(result_shifts, x), min_s8, max_s8));
        }
    }
    // Compute left-over elements
    for(; x < window_end_x; ++x)
    {
        int32_t in_value = *(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x) + wrapper::vgetlane(offset_term_s32.val[0], 0);

        if(has_a_offset)
        {
            in_value += (*(vector_sum_col_ptr + x) * a_offset);
        }
        if(has_bias)
        {
            in_value += *(bias_ptr + x);
        }

        if(is_fixed_point)
        {
            // Finalize and store the result
            *(out_it.ptr() + x) = finalize_quantization<is_bounded_relu>(in_value, result_multipliers[x], result_shifts[x], offset, static_cast<int8_t>(min_bound), static_cast<int8_t>(max_bound));
        }
        else
        {
            // Finalize quantization
            in_value = (in_value * result_multipliers[x]) >> (-result_shifts[x]);

            // Bound and store the result
            if(is_bounded_relu)
            {
                in_value = static_cast<int8_t>(std::max<int32_t>(min_bound, std::min<int32_t>(max_bound, in_value)));
            }
            *(out_it.ptr() + x) = static_cast<int8_t>(std::max<int32_t>(-128, std::min<int32_t>(127, in_value)));
        }
    }
}

template <typename T, bool is_gemm3d, bool is_bounded_relu, bool is_fixed_point>
void run_offset_contribution_output_stage(const Window &window,
                                          const ITensor *mm_result, const ITensor *vector_sum_col, const ITensor *vector_sum_row, const ITensor *bias, ITensor *output,
                                          int32_t a_offset, int32_t b_offset, int32_t k_offset, bool slide_vector_sum_col,
                                          GEMMLowpOutputStageInfo output_stage)
{
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;
    using Typer        = VectorTyper<T>;

    const int height_input = is_gemm3d ? mm_result->info()->dimension(1) : 0;
    const int depth_input  = is_gemm3d ? mm_result->info()->dimension(2) : 1;

    const int32_t multiplier = output_stage.gemmlowp_multiplier;
    const int32_t shift      = output_stage.gemmlowp_shift;
    const int32_t offset     = output_stage.gemmlowp_offset;
    const int32_t min_bound  = output_stage.gemmlowp_min_bound;
    const int32_t max_bound  = output_stage.gemmlowp_max_bound;

    const int32x4_t result_offset_s32 = vdupq_n_s32(offset);
    const int32x4_t result_shift_s32  = vdupq_n_s32(is_fixed_point ? shift : -shift);
    const auto      min_vec           = wrapper::vdup_n(static_cast<T>(min_bound), ExactTagType{});
    const auto      max_vec           = wrapper::vdup_n(static_cast<T>(max_bound), ExactTagType{});

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Window collapsed_window = win.collapse_if_possible(win, Window::DimZ);

    Iterator mm_result_it(mm_result, win);
    Iterator out_it(output, win);

    if((a_offset != 0) && (b_offset != 0))
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(vector_sum_col);
        ARM_COMPUTE_ERROR_ON_NULLPTR(vector_sum_row);

        Iterator vector_sum_col_it = get_vector_sum_col_it(collapsed_window, vector_sum_col);
        Iterator vector_sum_row_it = get_vector_sum_row_it(collapsed_window, vector_sum_row);

        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

        // Offset in case vector_sum_col is batched
        const int vector_sum_col_batch_offset = slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

        if(bias != nullptr)
        {
            Iterator bias_it = get_bias_it(collapsed_window, bias);
            execute_window_loop(collapsed_window, [&](const Coordinates & id)
            {
                const int  batch_id           = id.z() / depth_input;
                const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(vector_sum_col_it.ptr() + batch_id * vector_sum_col_batch_offset);
                const auto vector_sum_row_ptr = reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() + batch_id * sum_row_stride_y)
                                                + id.y() + (id.z() % depth_input) * height_input;
                run_offset_contribution_output_stage_window<Typer, true, true, true, is_bounded_relu, is_fixed_point>(vector_sum_col_ptr, vector_sum_row_ptr, reinterpret_cast<const int32_t *>(bias_it.ptr()),
                                                                                                                      mm_result_it,
                                                                                                                      out_it,
                                                                                                                      result_offset_s32, result_shift_s32,
                                                                                                                      min_vec, max_vec, a_offset, b_offset, k_offset,
                                                                                                                      multiplier, shift, offset, min_bound, max_bound,
                                                                                                                      window_step_x, window_start_x, window_end_x);
            },
            vector_sum_col_it, vector_sum_row_it, bias_it, mm_result_it, out_it);
        }
        else
        {
            execute_window_loop(collapsed_window, [&](const Coordinates & id)
            {
                const int  batch_id           = id.z() / depth_input;
                const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(vector_sum_col_it.ptr() + batch_id * vector_sum_col_batch_offset);
                const auto vector_sum_row_ptr = reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() + batch_id * sum_row_stride_y)
                                                + id.y() + (id.z() % depth_input) * height_input;
                run_offset_contribution_output_stage_window<Typer, true, true, false, is_bounded_relu, is_fixed_point>(vector_sum_col_ptr, vector_sum_row_ptr, nullptr, mm_result_it, out_it,
                                                                                                                       result_offset_s32, result_shift_s32,
                                                                                                                       min_vec, max_vec, a_offset, b_offset, k_offset,
                                                                                                                       multiplier, shift, offset, min_bound, max_bound,
                                                                                                                       window_step_x, window_start_x, window_end_x);
            },
            vector_sum_col_it, vector_sum_row_it, mm_result_it, out_it);
        }
    }
    else if((a_offset == 0) && (b_offset != 0))
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(vector_sum_row);

        Iterator vector_sum_row_it = get_vector_sum_row_it(collapsed_window, vector_sum_row);

        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

        if(bias != nullptr)
        {
            Iterator bias_it = get_bias_it(collapsed_window, bias);
            execute_window_loop(collapsed_window, [&](const Coordinates & id)
            {
                const int  batch_id           = id.z() / depth_input;
                const auto vector_sum_row_ptr = reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() + batch_id * sum_row_stride_y)
                                                + id.y() + (id.z() % depth_input) * height_input;
                run_offset_contribution_output_stage_window<Typer, false, true, true, is_bounded_relu, is_fixed_point>(nullptr, vector_sum_row_ptr, reinterpret_cast<const int32_t *>(bias_it.ptr()), mm_result_it,
                                                                                                                       out_it,
                                                                                                                       result_offset_s32, result_shift_s32,
                                                                                                                       min_vec, max_vec, a_offset, b_offset, k_offset,
                                                                                                                       multiplier, shift, offset, min_bound, max_bound,
                                                                                                                       window_step_x, window_start_x, window_end_x);
            },
            vector_sum_row_it, bias_it, mm_result_it, out_it);
        }
        else
        {
            execute_window_loop(collapsed_window, [&](const Coordinates & id)
            {
                const int  batch_id           = id.z() / depth_input;
                const auto vector_sum_row_ptr = reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() + batch_id * sum_row_stride_y)
                                                + id.y() + (id.z() % depth_input) * height_input;
                run_offset_contribution_output_stage_window<Typer, false, true, false, is_bounded_relu, is_fixed_point>(nullptr, vector_sum_row_ptr, nullptr, mm_result_it, out_it,
                                                                                                                        result_offset_s32, result_shift_s32,
                                                                                                                        min_vec, max_vec, a_offset, b_offset, k_offset,
                                                                                                                        multiplier, shift, offset, min_bound, max_bound,
                                                                                                                        window_step_x, window_start_x, window_end_x);
            },
            vector_sum_row_it, mm_result_it, out_it);
        }
    }
    else if((a_offset != 0) && (b_offset == 0))
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(vector_sum_col);

        Iterator vector_sum_col_it = get_vector_sum_col_it(collapsed_window, vector_sum_col);

        // Offset in case vector_sum_col is batched
        const int vector_sum_col_batch_offset = slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

        if(bias != nullptr)
        {
            Iterator bias_it = get_bias_it(collapsed_window, bias);
            execute_window_loop(collapsed_window, [&](const Coordinates & id)
            {
                const int  batch_id           = id.z() / depth_input;
                const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(vector_sum_col_it.ptr() + batch_id * vector_sum_col_batch_offset);
                run_offset_contribution_output_stage_window<Typer, true, false, true, is_bounded_relu, is_fixed_point>(vector_sum_col_ptr, nullptr, reinterpret_cast<const int32_t *>(bias_it.ptr()), mm_result_it,
                                                                                                                       out_it,
                                                                                                                       result_offset_s32, result_shift_s32,
                                                                                                                       min_vec, max_vec, a_offset, b_offset, k_offset,
                                                                                                                       multiplier, shift, offset, min_bound, max_bound,
                                                                                                                       window_step_x, window_start_x, window_end_x);
            },
            vector_sum_col_it, bias_it, mm_result_it, out_it);
        }
        else
        {
            execute_window_loop(collapsed_window, [&](const Coordinates & id)
            {
                const int  batch_id           = id.z() / depth_input;
                const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(vector_sum_col_it.ptr() + batch_id * vector_sum_col_batch_offset);
                run_offset_contribution_output_stage_window<Typer, true, false, false, is_bounded_relu, is_fixed_point>(vector_sum_col_ptr, nullptr, nullptr, mm_result_it, out_it,
                                                                                                                        result_offset_s32, result_shift_s32,
                                                                                                                        min_vec, max_vec, a_offset, b_offset, k_offset,
                                                                                                                        multiplier, shift, offset, min_bound, max_bound,
                                                                                                                        window_step_x, window_start_x, window_end_x);
            },
            vector_sum_col_it, mm_result_it, out_it);
        }
    }
    else
    {
        if(bias != nullptr)
        {
            Iterator bias_it = get_bias_it(collapsed_window, bias);
            execute_window_loop(collapsed_window, [&](const Coordinates &)
            {
                run_offset_contribution_output_stage_window<Typer, false, false, true, is_bounded_relu, is_fixed_point>(nullptr, nullptr, reinterpret_cast<const int32_t *>(bias_it.ptr()), mm_result_it, out_it,
                                                                                                                        result_offset_s32, result_shift_s32,
                                                                                                                        min_vec, max_vec, a_offset, b_offset, k_offset,
                                                                                                                        multiplier, shift, offset, min_bound, max_bound,
                                                                                                                        window_step_x, window_start_x, window_end_x);
            },
            bias_it, mm_result_it, out_it);
        }
        else
        {
            execute_window_loop(collapsed_window, [&](const Coordinates &)
            {
                run_offset_contribution_output_stage_window<Typer, false, false, false, is_bounded_relu, is_fixed_point>(nullptr, nullptr, nullptr, mm_result_it, out_it,
                                                                                                                         result_offset_s32, result_shift_s32,
                                                                                                                         min_vec, max_vec, a_offset, b_offset, k_offset,
                                                                                                                         multiplier, shift, offset, min_bound, max_bound,
                                                                                                                         window_step_x, window_start_x, window_end_x);
            },
            mm_result_it, out_it);
        }
        return;
    }
}

template <bool is_gemm3d, bool is_bounded_relu, bool is_fixed_point>
void run_offset_contribution_output_stage_symm(const Window &window,
                                               const ITensor *mm_result, const ITensor *vector_sum_col, const ITensor *vector_sum_row, const ITensor *bias, ITensor *output,
                                               int32_t a_offset, int32_t b_offset, int32_t k_offset, bool slide_vector_sum_col,
                                               GEMMLowpOutputStageInfo output_stage)
{
    ARM_COMPUTE_UNUSED(vector_sum_row, b_offset, k_offset);

    const int depth_input = is_gemm3d ? mm_result->info()->dimension(2) : 1;

    const int32_t offset    = output_stage.gemmlowp_offset;
    const int32_t min_bound = output_stage.gemmlowp_min_bound;
    const int32_t max_bound = output_stage.gemmlowp_max_bound;

    const int32_t *result_multipliers = output_stage.gemmlowp_multipliers.data();
    const int32_t *result_shifts      = output_stage.gemmlowp_shifts.data();
    const int32x4_t result_offset_s32  = vdupq_n_s32(offset);
    const int8x16_t min_s8             = vdupq_n_s8(static_cast<int8_t>(min_bound));
    const int8x16_t max_s8             = vdupq_n_s8(static_cast<int8_t>(max_bound));

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Window collapsed_window = win.collapse_if_possible(win, Window::DimZ);

    Iterator mm_result_it(mm_result, win);
    Iterator out_it(output, win);

    if(a_offset != 0)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(vector_sum_col);

        Iterator vector_sum_col_it = get_vector_sum_col_it(collapsed_window, vector_sum_col);

        // Offset in case vector_sum_col is batched
        const int vector_sum_col_batch_offset = slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

        if(bias != nullptr)
        {
            Iterator bias_it = get_bias_it(collapsed_window, bias);
            execute_window_loop(collapsed_window, [&](const Coordinates & id)
            {
                const int  batch_id           = id.z() / depth_input;
                const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(vector_sum_col_it.ptr() + batch_id * vector_sum_col_batch_offset);
                run_offset_contribution_output_stage_window_symm<true, true, is_bounded_relu, is_fixed_point>(vector_sum_col_ptr, reinterpret_cast<const int32_t *>(bias_it.ptr()), mm_result_it, out_it,
                                                                                                              result_multipliers, result_shifts,
                                                                                                              result_offset_s32, min_s8, max_s8,
                                                                                                              a_offset, offset, min_bound, max_bound,
                                                                                                              window_step_x, window_start_x, window_end_x);
            },
            vector_sum_col_it, bias_it, mm_result_it, out_it);
        }
        else
        {
            execute_window_loop(collapsed_window, [&](const Coordinates & id)
            {
                const int  batch_id           = id.z() / depth_input;
                const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(vector_sum_col_it.ptr() + batch_id * vector_sum_col_batch_offset);
                run_offset_contribution_output_stage_window_symm<true, false, is_bounded_relu, is_fixed_point>(vector_sum_col_ptr, nullptr, mm_result_it, out_it,
                                                                                                               result_multipliers, result_shifts,
                                                                                                               result_offset_s32, min_s8, max_s8,
                                                                                                               a_offset, offset, min_bound, max_bound,
                                                                                                               window_step_x, window_start_x, window_end_x);
            },
            vector_sum_col_it, mm_result_it, out_it);
        }
    }
    else
    {
        if(bias != nullptr)
        {
            Iterator bias_it = get_bias_it(collapsed_window, bias);
            execute_window_loop(collapsed_window, [&](const Coordinates &)
            {
                run_offset_contribution_output_stage_window_symm<false, true, is_bounded_relu, is_fixed_point>(nullptr, reinterpret_cast<const int32_t *>(bias_it.ptr()), mm_result_it, out_it,
                                                                                                               result_multipliers, result_shifts,
                                                                                                               result_offset_s32, min_s8, max_s8,
                                                                                                               a_offset, offset, min_bound, max_bound,
                                                                                                               window_step_x, window_start_x, window_end_x);
            },
            bias_it, mm_result_it, out_it);
        }
        else
        {
            execute_window_loop(collapsed_window, [&](const Coordinates &)
            {
                run_offset_contribution_output_stage_window_symm<false, false, is_bounded_relu, is_fixed_point>(nullptr, nullptr, mm_result_it, out_it,
                                                                                                                result_multipliers, result_shifts,
                                                                                                                result_offset_s32, min_s8, max_s8,
                                                                                                                a_offset, offset, min_bound, max_bound,
                                                                                                                window_step_x, window_start_x, window_end_x);
            },
            mm_result_it, out_it);
        }
        return;
    }
}

Status validate_arguments(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias, const ITensorInfo *output,
                          int32_t a_offset, int32_t b_offset, GEMMLowpOutputStageInfo output_stage)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mm_result, 1, DataType::S32);
    if(output->data_type() != DataType::QASYMM8)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(mm_result->dimension(0) > 1 && output_stage.gemmlowp_multipliers.size() > 1 && b_offset != 0);
    }
    ARM_COMPUTE_RETURN_ERROR_ON(output_stage.gemmlowp_min_bound > output_stage.gemmlowp_max_bound);
    ARM_COMPUTE_RETURN_ERROR_ON(output_stage.type != GEMMLowpOutputStageType::QUANTIZE_DOWN && output_stage.type != GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT);

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(mm_result->dimension(0) != bias->dimension(0));
    }

    // If a_offset == 0, vector_sum_col can be a nullptr
    if(a_offset != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_col, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(vector_sum_col->dimension(0) != mm_result->dimension(0));
    }

    // If b_offset == 0, vector_sum_row can be a nullptr
    if(b_offset != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_row, 1, DataType::S32);

        // Check if input is a 3D reinterpretation
        const bool reinterpret_as_3d = mm_result->num_dimensions() > 1 && mm_result->tensor_shape().y() != vector_sum_row->tensor_shape().x();

        // Validate input
        ARM_COMPUTE_RETURN_ERROR_ON(reinterpret_as_3d && vector_sum_row->dimension(0) != (mm_result->dimension(1) * mm_result->dimension(2)));
        ARM_COMPUTE_RETURN_ERROR_ON(!reinterpret_as_3d && vector_sum_row->dimension(0) != mm_result->dimension(1));

        TensorShape output_shape = output->tensor_shape();
        if(output_shape.num_dimensions() > 1)
        {
            const unsigned int output_batch_idx = reinterpret_as_3d ? 3 : 2;

            TensorShape vector_sum_row_shape = vector_sum_row->tensor_shape();
            vector_sum_row_shape.collapse_from(1);
            output_shape.collapse_from(output_batch_idx);

            ARM_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_row_shape[1] != output_shape[output_batch_idx],
                                            "mm_result tensor must have the same number of batches of output tensor");

            if(a_offset != 0)
            {
                TensorShape vector_sum_col_shape = vector_sum_col->tensor_shape();
                vector_sum_col_shape.collapse_from(1);

                ARM_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_col_shape[1] != 1 && vector_sum_col_shape[1] != vector_sum_row_shape[1],
                                                "vector_sum_col tensor must have the same number of batches of vector_sum_row_shape or the number of batches must be set to 1");
            }
        }
    }

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mm_result, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *mm_result, ITensorInfo *output)
{
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, mm_result->clone()->set_data_type(DataType::QASYMM8));

    // Configure kernel window
    Window win = calculate_max_window(*mm_result, Steps());

    // Note: This kernel performs 16 elements per iteration.
    // However, since we use a left-over for loop, we cannot have any read or write out of memory
    // For this reason num_elems_processed_per_iteration is 1 and so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));

    return std::make_pair(Status{}, win);
}

NEGEMMLowpOffsetContributionOutputStageKernel::NEGEMMLowpOffsetContributionOutputStageFunction
get_configured_function(const ITensor *mm_result, const ITensor *vector_sum_row, const ITensor *output, GEMMLowpOutputStageInfo output_stage)
{
    static std::map<uint8_t, NEGEMMLowpOffsetContributionOutputStageKernel::NEGEMMLowpOffsetContributionOutputStageFunction> map_function_qasymm =
    {
        { 0, &run_offset_contribution_output_stage<uint8_t, false, false, false> },
        { 1, &run_offset_contribution_output_stage<uint8_t, true, false, false> },
        { 2, &run_offset_contribution_output_stage<uint8_t, false, true, false> },
        { 3, &run_offset_contribution_output_stage<uint8_t, true, true, false> },
        { 4, &run_offset_contribution_output_stage<uint8_t, false, false, true> },
        { 5, &run_offset_contribution_output_stage<uint8_t, true, false, true> },
        { 6, &run_offset_contribution_output_stage<uint8_t, false, true, true> },
        { 7, &run_offset_contribution_output_stage<uint8_t, true, true, true> },
        { 8, &run_offset_contribution_output_stage<int8_t, false, false, false> },
        { 9, &run_offset_contribution_output_stage<int8_t, true, false, false> },
        { 10, &run_offset_contribution_output_stage<int8_t, false, true, false> },
        { 11, &run_offset_contribution_output_stage<int8_t, true, true, false> },
        { 12, &run_offset_contribution_output_stage<int8_t, false, false, true> },
        { 13, &run_offset_contribution_output_stage<int8_t, true, false, true> },
        { 14, &run_offset_contribution_output_stage<int8_t, false, true, true> },
        { 15, &run_offset_contribution_output_stage<int8_t, true, true, true> },
    };

    static std::map<uint8_t, NEGEMMLowpOffsetContributionOutputStageKernel::NEGEMMLowpOffsetContributionOutputStageFunction> map_function_qsymm =
    {
        { 0, &run_offset_contribution_output_stage_symm<false, false, false> },
        { 1, &run_offset_contribution_output_stage_symm<true, false, false> },
        { 2, &run_offset_contribution_output_stage_symm<false, true, false> },
        { 3, &run_offset_contribution_output_stage_symm<true, true, false> },
        { 4, &run_offset_contribution_output_stage_symm<false, false, true> },
        { 5, &run_offset_contribution_output_stage_symm<true, false, true> },
        { 6, &run_offset_contribution_output_stage_symm<false, true, true> },
        { 7, &run_offset_contribution_output_stage_symm<true, true, true> }
    };

    // Check if input is a 3D reinterpretation
    const bool reinterpret_as_3d = vector_sum_row != nullptr
                                   && mm_result->info()->num_dimensions() > 1
                                   && mm_result->info()->tensor_shape().y() != vector_sum_row->info()->tensor_shape().x();

    // Check if we need to clamp the result using min and max
    PixelValue type_min{};
    PixelValue type_max{};
    std::tie(type_min, type_max) = get_min_max(output->info()->data_type());
    int32_t    type_min_int    = type_min.get<int32_t>();
    int32_t    type_max_int    = type_max.get<int32_t>();
    const bool is_bounded_relu = !(output_stage.gemmlowp_min_bound <= type_min_int && output_stage.gemmlowp_max_bound >= type_max_int);

    // Check if we need to perform fixed point requantization
    const bool is_fixed_point = output_stage.type != GEMMLowpOutputStageType::QUANTIZE_DOWN;

    // Check if symmetric per-channel execution
    const bool is_signed = output->info()->data_type() == DataType::QASYMM8_SIGNED;

    // Check if symmetric per-channel execution
    const bool is_symm = output_stage.is_quantized_per_channel;

    // key acts as a bitset, setting the first bit on reinterpret_as_3d,
    // the second on is_bounded_relu, and the third on is_fixed_point.
    uint8_t key = (reinterpret_as_3d ? 1UL : 0UL) | ((is_bounded_relu ? 1UL : 0UL) << 1) | ((is_fixed_point ? 1UL : 0UL) << 2);
    if(is_symm)
    {
        return map_function_qsymm.find(key)->second;
    }
    else
    {
        key |= ((is_signed ? 1UL : 0UL) << 3);
        return map_function_qasymm.find(key)->second;
    }
}
} // namespace

NEGEMMLowpOffsetContributionOutputStageKernel::NEGEMMLowpOffsetContributionOutputStageKernel()
    : _function(nullptr), _vector_sum_col(nullptr), _vector_sum_row(nullptr), _bias(nullptr), _mm_result(nullptr), _output(nullptr), _a_offset(0), _b_offset(0), _k_offset(0), _slide_vector_sum_col(true),
      _output_stage(GEMMLowpOutputStageInfo())

{
}

void NEGEMMLowpOffsetContributionOutputStageKernel::configure(const ITensor *mm_result, const ITensor *vector_sum_col,
                                                              const ITensor *vector_sum_row, const ITensor *bias, ITensor *output,
                                                              int32_t k, int32_t a_offset, int32_t b_offset,
                                                              GEMMLowpOutputStageInfo output_stage)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(mm_result, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(mm_result->info(),
                                                  vector_sum_col != nullptr ? vector_sum_col->info() : nullptr, // NOLINT
                                                  vector_sum_row != nullptr ? vector_sum_row->info() : nullptr, // NOLINT
                                                  bias != nullptr ? bias->info() : nullptr,                     // NOLINT
                                                  output->info(), a_offset, b_offset, output_stage));           // NOLINT

    _vector_sum_col = vector_sum_col;
    _vector_sum_row = vector_sum_row;
    _bias           = bias;
    _mm_result      = mm_result;
    _output         = output;
    _a_offset       = a_offset;
    _b_offset       = b_offset;
    _k_offset       = a_offset * b_offset * k;
    _output_stage   = output_stage;

    // If a_offset == 0, vector_sum_col can be a nullptr
    if(a_offset != 0)
    {
        // Check if vector_sum_col_shape should be slidden or not
        // Don't slide vector_sum_col_shape along the y dimension if vector_sum_col_shape has just 1 dimension and vector_sum_row_shape more than 1
        // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
        _slide_vector_sum_col = vector_sum_col->info()->tensor_shape().num_dimensions() > 1;
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(mm_result->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);

    _function = get_configured_function(mm_result, vector_sum_row, output, output_stage);
}

Status NEGEMMLowpOffsetContributionOutputStageKernel::validate(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col,
                                                               const ITensorInfo *vector_sum_row, const ITensorInfo *bias, const ITensorInfo *output,
                                                               int32_t a_offset, int32_t b_offset, GEMMLowpOutputStageInfo output_stage)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(mm_result, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(mm_result, vector_sum_col, vector_sum_row, bias, output, a_offset, b_offset, output_stage));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(mm_result->clone().get(), output->clone().get()).first);
    return Status{};
}

void NEGEMMLowpOffsetContributionOutputStageKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    _function(window, _mm_result, _vector_sum_col, _vector_sum_row, _bias, _output, _a_offset, _b_offset, _k_offset, _slide_vector_sum_col, _output_stage);
}

} // namespace arm_compute