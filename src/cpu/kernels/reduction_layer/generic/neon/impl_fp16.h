/*
 * Copyright (c) 2024 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_REDUCTION_LAYER_GENERIC_NEON_IMPL_FP16_H
#define ACL_SRC_CPU_KERNELS_REDUCTION_LAYER_GENERIC_NEON_IMPL_FP16_H

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"

#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "support/SaturateCast.h"

#include <arm_neon.h>

namespace arm_compute
{
// Helper function that calls vqmovun/vqmvn, vcombine and vstore, allows templating of RedOpYZW_quantized
void combine_and_store(int16x8_t t1, int16x8_t t2, Iterator &output, int offset = 0)
{
    auto res = wrapper::vcombine(wrapper::vqmovn(t1), wrapper::vqmovn(t2));
    wrapper::vstore(reinterpret_cast<int8_t *>(output.ptr() + offset), res);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
uint32x4x4_t
calculate_index(uint32_t idx, float16x8_t a, float16x8_t b, uint32x4x4_t c, ReductionOperation op, int axis)
{
    uint32x4x2_t mask{0};
    uint16x8_t   mask_u16{0};
    if (op == ReductionOperation::ARG_IDX_MIN)
    {
        mask_u16 = wrapper::vcgt(b, a);
    }
    else
    {
        mask_u16 = wrapper::vclt(b, a);
    }
    mask.val[0]          = wrapper::vmovl(wrapper::vgetlow(mask_u16));
    mask.val[1]          = wrapper::vmovl(wrapper::vgethigh(mask_u16));
    uint32x4x2_t vec_idx = {{{idx + 0, idx + 1, idx + 2, idx + 3}, {idx + 4, idx + 5, idx + 6, idx + 7}}};
    if (axis != 0)
    {
        vec_idx.val[0] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
        vec_idx.val[1] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
    }
    uint32x4x4_t res = {wrapper::vbsl(mask.val[0], vec_idx.val[0], c.val[0]),
                        wrapper::vbsl(mask.val[1], vec_idx.val[1], c.val[1]), 0, 0};

    return res;
}

// Helper function to calculate the minimum value of the input vector. All the elements in the output vector contain the min value.
inline float16x4_t calculate_min(float16x8_t in)
{
    auto pmin = wrapper::vpmin(wrapper::vgethigh(in), wrapper::vgetlow(in));
    pmin      = wrapper::vpmin(pmin, pmin);
    return wrapper::vpmin(pmin, pmin);
}
// Helper function to calculate the maximum value of the input vector. All the elements in the output vector contain the max value.
inline float16x4_t calculate_max(float16x8_t in)
{
    auto pmax = wrapper::vpmax(wrapper::vgethigh(in), wrapper::vgetlow(in));
    pmax      = wrapper::vpmax(pmax, pmax);
    return wrapper::vpmax(pmax, pmax);
}

uint32_t calculate_vector_index(uint32x4x4_t vec_res_idx, float16x8_t vec_res_value, ReductionOperation op)
{
    uint32x4x2_t res_idx_mask{0};
    uint32x4_t   mask_ones = vdupq_n_u32(0xFFFFFFFF);
    uint16x8_t   mask_u16;
    if (op == ReductionOperation::ARG_IDX_MIN)
    {
        auto pmin = calculate_min(vec_res_value);
        mask_u16  = wrapper::vceq(vec_res_value, wrapper::vcombine(pmin, pmin));
    }
    else
    {
        auto pmax = calculate_max(vec_res_value);
        mask_u16  = wrapper::vceq(vec_res_value, wrapper::vcombine(pmax, pmax));
    }

    // Widen vectors
    auto wide_u32_1 =
        wrapper::vorr(vshll_n_u16(wrapper::vgetlow(mask_u16), 8), wrapper::vmovl(wrapper::vgetlow(mask_u16)));
    auto wide_u32_2 =
        wrapper::vorr(vshll_n_u16(wrapper::vgethigh(mask_u16), 8), wrapper::vmovl(wrapper::vgethigh(mask_u16)));
    res_idx_mask.val[0] = wrapper::vand(vec_res_idx.val[0], wide_u32_1);
    res_idx_mask.val[1] = wrapper::vand(vec_res_idx.val[1], wide_u32_2);
    res_idx_mask.val[0] = wrapper::vadd(res_idx_mask.val[0], mask_ones);
    res_idx_mask.val[1] = wrapper::vadd(res_idx_mask.val[1], mask_ones);

    uint32_t res  = 0xFFFFFFFF;
    uint32_t iter = 0;
    do
    {
        auto pmin = wrapper::vpmin(wrapper::vgethigh(res_idx_mask.val[iter]), wrapper::vgetlow(res_idx_mask.val[iter]));
        pmin      = wrapper::vpmin(pmin, pmin);
        res       = std::min(wrapper::vgetlane(pmin, 0), res);
        iter++;
    } while (iter < 2);

    return (res - 0xFFFFFFFF);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <class F>
class Reducer
{
public:
    static void reduceX(const Window &window, const ITensor *input, ITensor *output, F f, const ReductionOperation op)
    {
        // Set out window
        Window out_window(window);
        out_window.set(Window::DimX, Window::Dimension(0, 1, 1));

        f(window, out_window, input, output, op);
    }
    static void reduceY(const Window &window, const ITensor *input, ITensor *output, F f, const ReductionOperation op)
    {
        // Set in window
        Window in_window(window);
        Window out_window(window);

        in_window.set(Window::DimY, Window::Dimension(0, 1, 1));
        out_window.set(Window::DimY, Window::Dimension(0, output->info()->dimension(1), output->info()->dimension(1)));

        f(in_window, out_window, input, output, 1, op);
    }
    static void reduceZ(const Window &window, const ITensor *input, ITensor *output, F f, const ReductionOperation op)
    {
        // Set in window
        Window in_window(window);
        Window out_window(window);

        in_window.set(Window::DimZ, Window::Dimension(0, 1, 1));
        out_window.set(Window::DimZ, Window::Dimension(0, output->info()->dimension(2), output->info()->dimension(2)));

        f(in_window, out_window, input, output, 2, op);
    }
    static void reduceW(const Window &window, const ITensor *input, ITensor *output, F f, const ReductionOperation op)
    {
        // Set in/out window
        Window in_window(window);
        Window out_window(window);

        in_window.set(3, Window::Dimension(0, 1, 1));
        out_window.set(3, Window::Dimension(0, 1, 1));

        f(in_window, out_window, input, output, 3, op);
    }
};

template <typename T, int S>
struct RedOpX
{
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    inline void operator()(
        const Window &in_window, Window &out_window, const ITensor *in, ITensor *out, const ReductionOperation op)
    {
        const size_t input_dim_0    = in->info()->dimension(0);
        const int    window_step_x  = 16 / sizeof(T);
        const auto   window_start_x = static_cast<int>(in_window.x().start());
        const auto   window_end_x   = static_cast<int>(in_window.x().end());

        Window in_win_no_pad = in_window;
        in_win_no_pad.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input(in, in_win_no_pad);
        Iterator output(out, out_window);

        execute_window_loop(
            in_win_no_pad,
            [&](const Coordinates &)
            {
                const auto input_ptr = reinterpret_cast<const T *>(input.ptr());

                auto init_res_value = static_cast<T>(0.f);
                switch (op)
                {
                    case ReductionOperation::ARG_IDX_MAX:
                    case ReductionOperation::ARG_IDX_MIN:
                    case ReductionOperation::MIN:
                    case ReductionOperation::MAX:
                    {
                        init_res_value = static_cast<T>(*input_ptr);
                        break;
                    }
                    case ReductionOperation::PROD:
                    {
                        init_res_value = static_cast<T>(1.f);
                        break;
                    }
                    default:
                        break;
                }
                auto         vec_res_value = wrapper::vdup_n(init_res_value, ExactTagType{});
                uint32x4x4_t vec_res_idx{{0}};

                // Compute window_step_x elements per iteration
                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const auto vec_elements = wrapper::vloadq(input_ptr + x);
                    switch (op)
                    {
                        case ReductionOperation::SUM_SQUARE:
                            vec_res_value = wrapper::vadd(wrapper::vmul(vec_elements, vec_elements), vec_res_value);
                            break;
                        case ReductionOperation::MEAN_SUM:
                        case ReductionOperation::SUM:
                            vec_res_value = wrapper::vadd(vec_elements, vec_res_value);
                            break;
                        case ReductionOperation::PROD:
                            vec_res_value = wrapper::vmul(vec_elements, vec_res_value);
                            break;
                        case ReductionOperation::ARG_IDX_MIN:
                        {
                            auto temp_vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                            vec_res_idx   = calculate_index(x, temp_vec_res_value, vec_res_value, vec_res_idx, op, 0);
                            vec_res_value = temp_vec_res_value;
                            break;
                        }
                        case ReductionOperation::ARG_IDX_MAX:
                        {
                            auto temp_vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                            vec_res_idx   = calculate_index(x, temp_vec_res_value, vec_res_value, vec_res_idx, op, 0);
                            vec_res_value = temp_vec_res_value;
                            break;
                        }
                        case ReductionOperation::MIN:
                        {
                            vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                            break;
                        }
                        case ReductionOperation::MAX:
                        {
                            vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                            break;
                        }
                        default:
                            ARM_COMPUTE_ERROR("Not supported");
                    }
                }

                switch (op)
                {
                    case ReductionOperation::SUM:
                    case ReductionOperation::MEAN_SUM:
                    case ReductionOperation::SUM_SQUARE:
                    {
#ifdef ARM_COMPUTE_DEBUG_ENABLED
                        auto res = static_cast<T>(0.f);
                        for (int i = 0; i < S; ++i)
                        {
                            res += wrapper::vgetlane(vec_res_value, i);
                        }
#else  // ARM_COMPUTE_DEBUG_ENABLED
                        auto carry_res =
                            wrapper::vpadd(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
                        for (int i = 0; i < S / 4; ++i)
                        {
                            carry_res = wrapper::vpadd(carry_res, carry_res);
                        }
                        auto res = wrapper::vgetlane(carry_res, 0);
#endif // ARM_COMPUTE_DEBUG_ENABLED
                        if (op == ReductionOperation::SUM_SQUARE)
                        {
                            // Compute left-over elements
                            for (; x < window_end_x; ++x)
                            {
                                res += (*(input_ptr + x)) * (*(input_ptr + x));
                            }
                        }
                        else
                        {
                            // Compute left-over elements
                            for (; x < window_end_x; ++x)
                            {
                                res += *(input_ptr + x);
                            }
                        }

                        if (op == ReductionOperation::MEAN_SUM)
                        {
                            res /= input_dim_0;
                        }

                        *(reinterpret_cast<T *>(output.ptr())) = res;
                        break;
                    }
                    case ReductionOperation::PROD:
                    {
                        auto carry_res =
                            wrapper::vmul(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
                        T res = 1;
                        for (int i = 0; i < S / 2; ++i)
                        {
                            res *= wrapper::vgetlane(carry_res, i);
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x)
                        {
                            res *= *(input_ptr + x);
                        }

                        *(reinterpret_cast<T *>(output.ptr())) = res;
                        break;
                    }
                    case ReductionOperation::ARG_IDX_MIN:
                    {
                        auto idx = calculate_vector_index(vec_res_idx, vec_res_value, op);
                        auto res = static_cast<T>(wrapper::vgetlane(calculate_min(vec_res_value), 0));

                        // Compute left-over elements
                        for (; x < window_end_x; ++x)
                        {
                            if (*(input_ptr + x) < res)
                            {
                                idx = x;
                                res = *(input_ptr + x);
                            }
                        }
                        *(reinterpret_cast<uint32_t *>(output.ptr())) = idx;
                        break;
                    }
                    case ReductionOperation::ARG_IDX_MAX:
                    {
                        auto idx = calculate_vector_index(vec_res_idx, vec_res_value, op);
                        auto res = static_cast<T>(wrapper::vgetlane(calculate_max(vec_res_value), 0));

                        // Compute left-over elements
                        for (; x < window_end_x; ++x)
                        {
                            if (*(input_ptr + x) > res)
                            {
                                idx = x;
                                res = *(input_ptr + x);
                            }
                        }
                        *(reinterpret_cast<uint32_t *>(output.ptr())) = idx;
                        break;
                    }
                    case ReductionOperation::MIN:
                    {
                        auto res = static_cast<T>(wrapper::vgetlane(calculate_min(vec_res_value), 0));

                        // Compute left-over elements
                        for (; x < window_end_x; ++x)
                        {
                            res = *(input_ptr + x) < res ? *(input_ptr + x) : res;
                        }
                        *(reinterpret_cast<T *>(output.ptr())) = res;
                        break;
                    }
                    case ReductionOperation::MAX:
                    {
                        auto res = static_cast<T>(wrapper::vgetlane(calculate_max(vec_res_value), 0));

                        // Compute left-over elements
                        for (; x < window_end_x; ++x)
                        {
                            res = *(input_ptr + x) > res ? *(input_ptr + x) : res;
                        }
                        *(reinterpret_cast<T *>(output.ptr())) = res;
                        break;
                    }
                    default:
                        ARM_COMPUTE_ERROR("Not supported");
                }
            },
            input, output);
    }
};

template <typename T, int S>
struct RedOpYZW
{
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;
    using neon_vector  = typename wrapper::traits::neon_vector<T, S>::type;

    inline void operator()(const Window            &in_window,
                           Window                  &out_window,
                           const ITensor           *in,
                           ITensor                 *out,
                           int                      axis,
                           const ReductionOperation op)
    {
        const TensorInfo in_info            = *(in->info());
        const int        window_step_x      = 16 / sizeof(T);
        const auto       window_start_x_tmp = static_cast<int>(in_window.x().start());
        const auto       window_end_x_tmp   = static_cast<int>(in_window.x().end());
        // As it split over x-axis, need to set the correct spiltted window start and end.
        const auto window_start_x = static_cast<int>(0);
        const auto window_end_x   = static_cast<int>(in_window.shape().x());

        Window in_win_no_pad = in_window;
        in_win_no_pad.set(Window::DimX, Window::Dimension(window_start_x_tmp, window_end_x_tmp, in_window.shape().x()));
        Window out_win_no_pad = out_window;
        out_win_no_pad.set(Window::DimX,
                           Window::Dimension(window_start_x_tmp, window_end_x_tmp, out_window.shape().x()));

        Iterator input(in, in_win_no_pad);
        Iterator output(out, out_win_no_pad);

        execute_window_loop(
            in_win_no_pad,
            [&](const Coordinates &)
            {
                const auto input_ptr = reinterpret_cast<T *>(input.ptr());

                // Compute window_step_x elements per iteration
                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    neon_vector vec_res_value = {0};
                    switch (op)
                    {
                        case ReductionOperation::ARG_IDX_MAX:
                        case ReductionOperation::ARG_IDX_MIN:
                        case ReductionOperation::MIN:
                        case ReductionOperation::MAX:
                        {
                            vec_res_value = wrapper::vloadq(input_ptr + x);
                            break;
                        }
                        case ReductionOperation::PROD:
                        {
                            vec_res_value = wrapper::vdup_n(static_cast<T>(1.f), ExactTagType{});
                            break;
                        }
                        default:
                        {
                            vec_res_value = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
                            break;
                        }
                    }
                    uint32x4x4_t vec_res_idx{{0}};

                    for (unsigned int dim = 0; dim < in_info.dimension(axis); ++dim)
                    {
                        const T *in_ptr =
                            reinterpret_cast<T *>(input.ptr() + x * sizeof(T) + in_info.strides_in_bytes()[axis] * dim);
                        const auto vec_elements = wrapper::vloadq(in_ptr);
                        switch (op)
                        {
                            case ReductionOperation::SUM:
                            case ReductionOperation::MEAN_SUM:
                                vec_res_value = wrapper::vadd(vec_elements, vec_res_value);
                                break;
                            case ReductionOperation::SUM_SQUARE:
                                vec_res_value = wrapper::vadd(wrapper::vmul(vec_elements, vec_elements), vec_res_value);
                                break;
                            case ReductionOperation::PROD:
                                vec_res_value = wrapper::vmul(vec_elements, vec_res_value);
                                break;
                            case ReductionOperation::ARG_IDX_MIN:
                            {
                                auto temp_vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                                vec_res_idx =
                                    calculate_index(dim, temp_vec_res_value, vec_res_value, vec_res_idx, op, axis);
                                vec_res_value = temp_vec_res_value;
                                break;
                            }
                            case ReductionOperation::ARG_IDX_MAX:
                            {
                                auto temp_vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                                vec_res_idx =
                                    calculate_index(dim, temp_vec_res_value, vec_res_value, vec_res_idx, op, axis);
                                vec_res_value = temp_vec_res_value;
                                break;
                            }
                            case ReductionOperation::MIN:
                            {
                                vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                                break;
                            }
                            case ReductionOperation::MAX:
                            {
                                vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                                break;
                            }
                            default:
                                ARM_COMPUTE_ERROR("Not supported");
                        }
                    }

                    if (op == ReductionOperation::MEAN_SUM)
                    {
                        auto vec_width_inv =
                            wrapper::vinv(wrapper::vdup_n(static_cast<T>(in_info.dimension(axis)), ExactTagType{}));
                        vec_res_value = wrapper::vmul(vec_res_value, vec_width_inv);
                    }

                    if (op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::ARG_IDX_MAX)
                    {
                        wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr()) + x, vec_res_idx.val[0]);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                        if (std::is_same<T, float16_t>::value)
                        {
                            wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr()) + x + 4, vec_res_idx.val[1]);
                        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    }
                    else
                    {
                        wrapper::vstore(reinterpret_cast<T *>(output.ptr() + x * sizeof(T)), vec_res_value);
                    }
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    auto res_value = 0.f;
                    switch (op)
                    {
                        case ReductionOperation::ARG_IDX_MAX:
                        case ReductionOperation::ARG_IDX_MIN:
                        case ReductionOperation::MIN:
                        case ReductionOperation::MAX:
                        {
                            res_value = *(input_ptr + x);
                            break;
                        }
                        case ReductionOperation::PROD:
                        {
                            res_value = static_cast<T>(1.f);
                            break;
                        }
                        default:
                        {
                            res_value = static_cast<T>(0.f);
                            break;
                        }
                    }

                    uint32_t res_idx = 0;
                    for (unsigned int dim = 0; dim < in_info.dimension(axis); ++dim)
                    {
                        const T *in_ptr =
                            reinterpret_cast<T *>(input.ptr() + x * sizeof(T) + in_info.strides_in_bytes()[axis] * dim);

                        switch (op)
                        {
                            case ReductionOperation::SUM:
                            case ReductionOperation::MEAN_SUM:
                                res_value += *in_ptr;
                                break;
                            case ReductionOperation::SUM_SQUARE:
                                res_value += *in_ptr * *in_ptr;
                                break;
                            case ReductionOperation::PROD:
                                res_value *= *in_ptr;
                                break;
                            case ReductionOperation::ARG_IDX_MIN:
                            {
                                if (*in_ptr < res_value)
                                {
                                    res_value = *in_ptr;
                                    res_idx   = dim;
                                }
                                break;
                            }
                            case ReductionOperation::ARG_IDX_MAX:
                            {
                                if (*in_ptr > res_value)
                                {
                                    res_value = *in_ptr;
                                    res_idx   = dim;
                                }
                                break;
                            }
                            case ReductionOperation::MIN:
                            {
                                res_value = *in_ptr < res_value ? *in_ptr : res_value;
                                break;
                            }
                            case ReductionOperation::MAX:
                            {
                                res_value = *in_ptr > res_value ? *in_ptr : res_value;
                                break;
                            }
                            default:
                                ARM_COMPUTE_ERROR("Not supported");
                        }
                    }

                    if (op == ReductionOperation::MEAN_SUM)
                    {
                        res_value /= in_info.dimension(axis);
                    }

                    if (op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::ARG_IDX_MAX)
                    {
                        *(reinterpret_cast<uint32_t *>(output.ptr()) + x) = res_idx;
                    }
                    else
                    {
                        *(reinterpret_cast<T *>(output.ptr() + x * sizeof(T))) = res_value;
                    }
                }
            },
            input, output);
    }
};

template <typename T, int S, int axis, ReductionOperation op>
struct RedOpYZW_complex
{
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;
    using neon_vector  = typename wrapper::traits::neon_vector<T, S>::type;

    inline void operator()(
        const Window &in_window, Window &out_window, const ITensor *in, ITensor *out, int, const ReductionOperation)
    {
        ARM_COMPUTE_ERROR_ON(axis != 2);
        ARM_COMPUTE_ERROR_ON(op != ReductionOperation::SUM);

        const TensorInfo in_info            = *(in->info());
        const size_t     stride_z           = in_info.strides_in_bytes()[axis];
        const int        window_step_x      = 16 / sizeof(T);
        const auto       window_start_x_tmp = static_cast<int>(in_window.x().start());
        const auto       window_end_x_tmp   = static_cast<int>(in_window.x().end());
        // As it split over x-axis, need to set the correct spiltted window start and end.
        const auto window_start_x = static_cast<int>(0);
        const auto window_end_x   = static_cast<int>(in_window.shape().x());

        Window in_win_no_pad = in_window;
        in_win_no_pad.set(Window::DimX, Window::Dimension(window_start_x_tmp, window_end_x_tmp, in_window.shape().x()));
        Window out_win_no_pad = out_window;
        out_win_no_pad.set(Window::DimX,
                           Window::Dimension(window_start_x_tmp, window_end_x_tmp, out_window.shape().x()));

        Iterator input(in, in_win_no_pad);
        Iterator output(out, out_win_no_pad);

        execute_window_loop(
            in_win_no_pad,
            [&](const Coordinates &)
            {
                // Compute window_step_x elements per iteration
                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    neon_vector vec_res_value_0 = {0};
                    neon_vector vec_res_value_1 = {0};

                    vec_res_value_0 = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
                    vec_res_value_1 = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});

                    T *out_ptr = reinterpret_cast<T *>(output.ptr() + 2 * x * sizeof(T));
                    for (unsigned int dim = 0; dim < in_info.dimension(axis); ++dim)
                    {
                        T *in_ptr_0 = reinterpret_cast<T *>(input.ptr() + 2 * x * sizeof(T) + stride_z * dim);
                        T *in_ptr_1 = reinterpret_cast<T *>(input.ptr() + 2 * x * sizeof(T) + 16 + stride_z * dim);

                        const auto vec_elements_0 = wrapper::vloadq(in_ptr_0);
                        const auto vec_elements_1 = wrapper::vloadq(in_ptr_1);

                        vec_res_value_0 = wrapper::vadd(vec_elements_0, vec_res_value_0);
                        vec_res_value_1 = wrapper::vadd(vec_elements_1, vec_res_value_1);
                    }

                    wrapper::vstore(out_ptr, vec_res_value_0);
                    wrapper::vstore(out_ptr + 4, vec_res_value_1);
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    auto res_value_0 = 0.f;
                    auto res_value_1 = 0.f;

                    T *out_ptr = reinterpret_cast<T *>(output.ptr() + 2 * x * sizeof(T));
                    for (unsigned int dim = 0; dim < in_info.dimension(axis); ++dim)
                    {
                        T *in_ptr = reinterpret_cast<T *>(input.ptr() + 2 * x * sizeof(T) + stride_z * dim);
                        res_value_0 += *in_ptr;
                        res_value_1 += *(in_ptr + 1);
                    }
                    *out_ptr       = res_value_0;
                    *(out_ptr + 1) = res_value_1;
                }
            },
            input, output);
    }
};

} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_REDUCTION_LAYER_GENERIC_NEON_IMPL_FP16_H
