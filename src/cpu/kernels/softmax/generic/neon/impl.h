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
#ifndef ACL_SRC_CPU_KERNELS_SOFTMAX_GENERIC_NEON_IMPL_H
#define ACL_SRC_CPU_KERNELS_SOFTMAX_GENERIC_NEON_IMPL_H

#include "arm_compute/core/Helpers.h"

#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{

#ifdef __aarch64__
namespace
{
// These helper functions are added because vaddv does not exist for fp16,
// and, therefore, is not part of the wrapper::vaddv interface.
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float16_t wrapper_vaddv(const float16x8_t &a, int sum_stages)
{
    auto sum_res = wrapper::vpadd(wrapper::vgethigh(a), wrapper::vgetlow(a));
    for (int i = 0; i < sum_stages; ++i)
    {
        sum_res = wrapper::vpadd(sum_res, sum_res);
    }
    return wrapper::vgetlane(sum_res, 0);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

inline float wrapper_vaddv(const float32x4_t &a, int sum_stages)
{
    ARM_COMPUTE_UNUSED(sum_stages);
    return wrapper::vaddv(a);
}
} // namespace
#endif // __aarch64__

// The template implementation for float data types is stored in the header file because
// we need all fp16 instantiated code to live in fp16.cpp files.
template <typename T, bool IS_LOG>
void neon_softmax_float(const ITensor *in, void *const tmp, ITensor *out, float beta, const Window &window)
{
    ARM_COMPUTE_UNUSED(tmp);

    const int input_width = in->info()->valid_region().shape.x();

    Iterator in_it(in, window);
    Iterator out_it(out, window);

    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    constexpr int vec_size = 16 / sizeof(T);

    const int sum_stages = log2(vec_size >> 1);

    const auto beta_vec = wrapper::vdup_n(static_cast<T>(beta), ExactTagType{});

    execute_window_loop(
        window,
        [&](const Coordinates &)
        {
            /* Get pointers */
            const T *in_ptr  = reinterpret_cast<const T *>(in_it.ptr());
            T       *out_ptr = reinterpret_cast<T *>(out_it.ptr());

            T max_val;

            /* Compute Max */
            {
                // Init max value
                auto vec_max = wrapper::vdup_n(support::cpp11::lowest<T>(), ExactTagType{});
                int  x       = 0;

                for (; x <= (input_width - vec_size); x += vec_size)
                {
                    const auto current_value = wrapper::vloadq(in_ptr + x);
                    vec_max                  = wrapper::vmax(vec_max, current_value);
                }

#ifdef __aarch64__
                max_val = wrapper::vmaxv(vec_max);
#else  // __aarch64__
                auto carry_max = wrapper::vpmax(wrapper::vgethigh(vec_max), wrapper::vgetlow(vec_max));

                for (int i = 0; i < sum_stages; ++i)
                {
                    carry_max = wrapper::vpmax(carry_max, carry_max);
                }

                max_val      = wrapper::vgetlane(carry_max, 0);
#endif // __aarch64__

                // Compute left-over elements
                for (; x < input_width; ++x)
                {
                    max_val = std::max(*(in_ptr + x), max_val);
                }
            } // compute max

            T sum_transformed{};

            /* Compute exponentials and sum */
            {
                /* Get max value */
                const auto vec_max = wrapper::vdup_n(max_val, ExactTagType{});

                /* Init sum to zero */
                auto vec_sum = wrapper::vdup_n(static_cast<T>(0), ExactTagType{});

                /* Loop over row and compute exponentials and sum */
                int x = 0;
                for (; x <= (input_width - vec_size); x += vec_size)
                {
                    auto vec_elements = wrapper::vloadq(in_ptr + x);
                    vec_elements      = wrapper::vsub(vec_elements, vec_max);
                    if (IS_LOG)
                    {
                        vec_elements = wrapper::vmul(vec_elements, beta_vec);
                        vec_sum      = wrapper::vadd(vec_sum, wrapper::vexpq(vec_elements));
                    }
                    else
                    {
                        vec_elements = wrapper::vexpq(wrapper::vmul(vec_elements, beta_vec));
                        vec_sum      = wrapper::vadd(vec_sum, vec_elements);
                    }
                    wrapper::vstore(out_ptr + x, vec_elements);
                }

                /* Reduce sum */
                T sum{};
#ifdef __aarch64__
                sum = wrapper_vaddv(vec_sum, sum_stages);
#else  // __aarch64__
                auto sum_res = wrapper::vpadd(wrapper::vgethigh(vec_sum), wrapper::vgetlow(vec_sum));
                for (int i = 0; i < sum_stages; ++i)
                {
                    sum_res = wrapper::vpadd(sum_res, sum_res);
                }
                sum = wrapper::vgetlane(sum_res, 0);
#endif // __aarch64__

                /* Run remaining elements */
                for (; x < input_width; ++x)
                {
                    T element{};

                    if (IS_LOG)
                    {
                        element = (in_ptr[x] - max_val) * beta;
                        sum += std::exp(element);
                    }
                    else
                    {
                        element = std::exp((in_ptr[x] - max_val) * beta);
                        sum += element;
                    }

                    out_ptr[x] = element;
                }

                if (!IS_LOG)
                {
                    sum_transformed = T(1) / sum;
                }
                else
                {
                    sum_transformed = static_cast<T>(std::log(sum));
                }
            } // Compute exponentials and sum

            /* Normalize exponentials */
            {
                const auto sum_vec = wrapper::vdup_n(static_cast<T>(sum_transformed), ExactTagType{});

                /* Loop over row and compute softmax */
                int x = 0;
                for (; x <= (input_width - vec_size); x += vec_size)
                {
                    const auto vec_in = wrapper::vloadq(out_ptr + x);
                    if (IS_LOG)
                    {
                        wrapper::vstore(out_ptr + x, wrapper::vsub(vec_in, sum_vec));
                    }
                    else
                    {
                        wrapper::vstore(out_ptr + x, wrapper::vmul(vec_in, sum_vec));
                    }
                }

                /* Run remaining elements */
                for (; x < input_width; ++x)
                {
                    if (IS_LOG)
                    {
                        out_ptr[x] = out_ptr[x] - sum_transformed;
                    }
                    else
                    {
                        out_ptr[x] = out_ptr[x] * sum_transformed;
                    }
                }
            } // Normalize exponentials
        },
        in_it, out_it);
}

template <typename T, bool IS_LOG>
void neon_softmax_quantized(const ITensor *in, void *const tmp, ITensor *out, float beta, const Window &window);
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_SOFTMAX_GENERIC_NEON_IMPL_H
