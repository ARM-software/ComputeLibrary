/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_SOFTMAX_IMPL_H
#define SRC_CORE_NEON_KERNELS_SOFTMAX_IMPL_H

#include "arm_compute/core/Helpers.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{
template <typename T>
void neon_logits_1d_max(const ITensor *in, ITensor *out, const Window &window)
{
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    constexpr int window_step_x  = 16 / sizeof(T);
    const auto    window_start_x = static_cast<int>(window.x().start());
    const auto    window_end_x   = static_cast<int>(window.x().end());

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator input(in, win);
    Iterator output(out, win);

    const int sum_stages = log2(window_step_x / 2);
    execute_window_loop(win, [&](const Coordinates &)
    {
        // Get pointers
        const auto in_ptr  = reinterpret_cast<const T *>(input.ptr());
        const auto out_ptr = reinterpret_cast<T *>(output.ptr());

        // Init max value
        auto vec_max = wrapper::vdup_n(support::cpp11::lowest<T>(), ExactTagType{});
        int  x       = window_start_x;

        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto current_value = wrapper::vloadq(in_ptr + x);
            vec_max                  = wrapper::vmax(vec_max, current_value);
        }
        auto carry_max = wrapper::vpmax(wrapper::vgethigh(vec_max), wrapper::vgetlow(vec_max));

        for(int i = 0; i < sum_stages; ++i)
        {
            carry_max = wrapper::vpmax(carry_max, carry_max);
        }
        T max_val = wrapper::vgetlane(carry_max, 0);

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            max_val = *(in_ptr + x) > max_val ? *(in_ptr + x) : max_val;
        }

        *out_ptr = max_val;
    },
    input, output);
}

template <typename T>
void neon_softmax_logits_1d_quantized(const ITensor *in, const ITensor *max, void *const tmp,
                                      ITensor *out, float beta, bool is_log, const Window &window);

template <typename T>
void neon_softmax_logits_1d_float(const ITensor *in, const ITensor *max, void *const tmp,
                                  ITensor *out, const float beta, bool is_log, const Window &window)
{
    const int start_x     = in->info()->valid_region().anchor.x();
    const int input_width = in->info()->valid_region().shape.x();

    Iterator in_it(in, window);
    Iterator max_it(max, window);
    Iterator out_it(out, window);

    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    constexpr int vec_size   = 16 / sizeof(T);
    const int     sum_stages = log2(vec_size / 2);

    execute_window_loop(window, [&](const Coordinates &)
    {
        /* Get pointers */
        const auto in_ptr  = reinterpret_cast<const T *>(in_it.ptr()) + start_x;
        const auto out_ptr = reinterpret_cast<T *>(out_it.ptr()) + start_x;
        const auto tmp_ptr = reinterpret_cast<T *>(tmp);

        T sum{};
        T sum_inversed{};

        /* Compute exponentials and sum */
        {
            /* Get max value */
            const auto max_val = *reinterpret_cast<const T *>(max_it.ptr());
            const auto vec_max = wrapper::vdup_n(max_val, ExactTagType{});

            /* Init sum to zero */
            auto vec_sum = wrapper::vdup_n(static_cast<T>(0), ExactTagType{});

            /* Loop over row and compute exponentials and sum */
            int x = 0;
            for(; x <= (input_width - vec_size); x += vec_size)
            {
                auto vec_elements = wrapper::vloadq(in_ptr + x);
                vec_elements      = wrapper::vsub(vec_elements, vec_max);
                if(is_log)
                {
                    vec_elements = wrapper::vmul(vec_elements, wrapper::vdup_n(static_cast<T>(beta), ExactTagType{}));
                    vec_sum      = wrapper::vadd(vec_sum, wrapper::vexpq(vec_elements));
                }
                else
                {
                    vec_elements = wrapper::vexpq(wrapper::vmul(vec_elements, wrapper::vdup_n(static_cast<T>(beta), ExactTagType{})));
                    vec_sum      = wrapper::vadd(vec_sum, vec_elements);
                }
                wrapper::vstore(tmp_ptr + x, vec_elements);
            }

            /* Reduce sum */
            auto sum_res = wrapper::vpadd(wrapper::vgethigh(vec_sum), wrapper::vgetlow(vec_sum));
            for(int i = 0; i < sum_stages; ++i)
            {
                sum_res = wrapper::vpadd(sum_res, sum_res);
            }
            sum = wrapper::vgetlane(sum_res, 0);

            /* Run remaining elements */
            for(; x < input_width; ++x)
            {
                T element{};

                if(is_log)
                {
                    element = (in_ptr[x] - max_val) * beta;
                    sum += std::exp(element);
                }
                else
                {
                    element = std::exp((in_ptr[x] - max_val) * beta);
                    sum += element;
                }
                tmp_ptr[x] = element;
            }

            if(!is_log)
            {
                sum_inversed = T(1) / sum;
            }
            else
            {
                sum = static_cast<T>(std::log(sum));
            }
        }

        /* Normalize exponentials */
        {
            /* Loop over row and compute softmax */
            int x = 0;
            for(; x <= (input_width - vec_size); x += vec_size)
            {
                auto vec_in           = wrapper::vloadq(tmp_ptr + x);
                auto normalized_value = wrapper::vdup_n(static_cast<T>(0), ExactTagType{});
                if(is_log)
                {
                    normalized_value = wrapper::vsub(vec_in, wrapper::vdup_n(static_cast<T>(sum), ExactTagType{}));
                }
                else
                {
                    normalized_value = wrapper::vmul(vec_in, wrapper::vdup_n(static_cast<T>(sum_inversed), ExactTagType{}));
                }
                wrapper::vstore(out_ptr + x, normalized_value);
            }
            /* Run remaining elements */
            for(; x < input_width; ++x)
            {
                if(is_log)
                {
                    out_ptr[x] = tmp_ptr[x] - sum;
                }
                else
                {
                    out_ptr[x] = tmp_ptr[x] * sum_inversed;
                }
            }
        }
    },
    in_it, max_it, out_it);
}
} // namespace cpu
} // namespace arm_compute

#endif /* SRC_CORE_NEON_KERNELS_SOFTMAX_IMPL_H */
