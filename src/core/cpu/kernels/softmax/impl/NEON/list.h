/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_SOFTMAX_LIST_H
#define SRC_CORE_NEON_KERNELS_SOFTMAX_LIST_H

#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "support/SaturateCast.h"

namespace arm_compute
{
namespace cpu
{
namespace
{
template <typename float_vec_type, typename int_vec_type>
int_vec_type convert_float_to_int(const float_vec_type &in);

template <typename float_vec_type, typename int_vec_type>
float_vec_type convert_int_to_float(const int_vec_type &in);

template <>
uint8x16_t convert_float_to_int<float32x4x4_t, uint8x16_t>(const float32x4x4_t &in)
{
    uint8x16_t out;
    convert_float32x4x4_to_uint8x16(in, out);
    return out;
}

template <>
int8x16_t convert_float_to_int<float32x4x4_t, int8x16_t>(const float32x4x4_t &in)
{
    int8x16_t out;
    convert_float32x4x4_to_int8x16(in, out);
    return out;
}

template <>
float32x4x4_t convert_int_to_float<float32x4x4_t, uint8x16_t>(const uint8x16_t &in)
{
    return convert_uint8x16_to_float32x4x4(in);
}

template <>
float32x4x4_t convert_int_to_float<float32x4x4_t, int8x16_t>(const int8x16_t &in)
{
    return convert_int8x16_to_float32x4x4(in);
}
} // namespace

template <typename T>
void neon_logits_1d_max(const ITensor *in, ITensor *out, const Window &window)
{
    /** Neon vector tag type. */
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
                                      ITensor *out, float beta, bool is_log, const Window &window)
{
    static_assert(std::is_same<T, qasymm8_t>::value
                  || std::is_same<T, qasymm8_signed_t>::value,
                  "quantized type should be either qasymm8_t or qasymm8_signed_t.");

    const int start_x     = in->info()->valid_region().anchor.x();
    const int input_width = in->info()->valid_region().shape.x();

    const float scale_beta     = -beta * in->info()->quantization_info().uniform().scale;
    const auto  scale_beta_vec = vdupq_n_f32(scale_beta);

    Iterator      in_it(in, window);
    Iterator      max_it(max, window);
    Iterator      out_it(out, window);
    constexpr int vec_size = 16;

    execute_window_loop(window, [&](const Coordinates &)
    {
        /* Get pointers */
        const auto in_ptr  = reinterpret_cast<const T *>(in_it.ptr()) + start_x;
        const auto out_ptr = reinterpret_cast<T *>(out_it.ptr()) + start_x;
        const auto tmp_ptr = reinterpret_cast<float *>(tmp);

        float sum{};
        float sum_inversed{};

        /* Compute exponentials and sum */
        {
            /* Get max value */
            const auto max_val = *reinterpret_cast<const T *>(max_it.ptr());
            const auto vec_max = wrapper::vdup_n(max_val, wrapper::traits::vector_128_tag{});

            /* Init sum to zero */
            float32x4x4_t vec_sum =
            {
                vdupq_n_f32(0.f),
                vdupq_n_f32(0.f),
                vdupq_n_f32(0.f),
                vdupq_n_f32(0.f),
            };

            /* Loop over row and compute exponentials and sum */
            int x = 0;
            for(; x <= (input_width - vec_size); x += vec_size)
            {
                auto vec_elements     = wrapper::vloadq(in_ptr + x);
                vec_elements          = wrapper::vqsub(vec_max, vec_elements);
                auto vec_elements_flt = convert_int_to_float<float32x4x4_t>(vec_elements);

                if(is_log)
                {
                    vec_elements_flt.val[0] = vmulq_f32(vec_elements_flt.val[0], scale_beta_vec);
                    vec_elements_flt.val[1] = vmulq_f32(vec_elements_flt.val[1], scale_beta_vec);
                    vec_elements_flt.val[2] = vmulq_f32(vec_elements_flt.val[2], scale_beta_vec);
                    vec_elements_flt.val[3] = vmulq_f32(vec_elements_flt.val[3], scale_beta_vec);
                    vec_sum.val[0]          = vaddq_f32(vec_sum.val[0], vexpq_f32(vec_elements_flt.val[0]));
                    vec_sum.val[1]          = vaddq_f32(vec_sum.val[1], vexpq_f32(vec_elements_flt.val[1]));
                    vec_sum.val[2]          = vaddq_f32(vec_sum.val[2], vexpq_f32(vec_elements_flt.val[2]));
                    vec_sum.val[3]          = vaddq_f32(vec_sum.val[3], vexpq_f32(vec_elements_flt.val[3]));
                }
                else
                {
                    vec_elements_flt.val[0] = vexpq_f32(vmulq_f32(vec_elements_flt.val[0], scale_beta_vec));
                    vec_elements_flt.val[1] = vexpq_f32(vmulq_f32(vec_elements_flt.val[1], scale_beta_vec));
                    vec_elements_flt.val[2] = vexpq_f32(vmulq_f32(vec_elements_flt.val[2], scale_beta_vec));
                    vec_elements_flt.val[3] = vexpq_f32(vmulq_f32(vec_elements_flt.val[3], scale_beta_vec));
                    vec_sum.val[0]          = vaddq_f32(vec_sum.val[0], vec_elements_flt.val[0]);
                    vec_sum.val[1]          = vaddq_f32(vec_sum.val[1], vec_elements_flt.val[1]);
                    vec_sum.val[2]          = vaddq_f32(vec_sum.val[2], vec_elements_flt.val[2]);
                    vec_sum.val[3]          = vaddq_f32(vec_sum.val[3], vec_elements_flt.val[3]);
                }

                vst4q_f32(tmp_ptr + x, vec_elements_flt);
            }

            /* Reduce sum */
            const auto sum_16_byte = vaddq_f32(vaddq_f32(vec_sum.val[0], vec_sum.val[1]), vaddq_f32(vec_sum.val[2], vec_sum.val[3]));
            auto       sum_res     = vpadd_f32(vget_high_f32(sum_16_byte), vget_low_f32(sum_16_byte));
            sum_res                = vpadd_f32(sum_res, sum_res);
            sum                    = wrapper::vgetlane(sum_res, 0);

            /* Run remaining elements */
            for(; x < input_width; ++x)
            {
                float element{};
                if(is_log)
                {
                    element = (max_val - in_ptr[x]) * scale_beta;
                    sum += std::exp(element);
                }
                else
                {
                    element = std::exp((max_val - in_ptr[x]) * scale_beta);
                    sum += element;
                }

                tmp_ptr[x] = element;
            }

            if(!is_log)
            {
                sum_inversed = 256.f / sum;
            }
            else
            {
                sum = std::log(sum);
            }
        }

        /* Normalize exponentials */
        {
            constexpr bool is_qasymm8_signed = std::is_same<T, qasymm8_signed_t>::value;
            /* Loop over row and compute softmax */
            int x = 0;
            for(; x <= (input_width - vec_size); x += vec_size)
            {
                using int_vec_type   = wrapper::traits::neon_vector_t<T, 16>;
                float32x4x4_t vec_in = vld4q_f32(tmp_ptr + x);
                int_vec_type  normalized_value{};
                if(is_log)
                {
                    const float32x4x4_t sub =
                    {
                        vsubq_f32(vec_in.val[0], vdupq_n_f32(sum)),
                        vsubq_f32(vec_in.val[1], vdupq_n_f32(sum)),
                        vsubq_f32(vec_in.val[2], vdupq_n_f32(sum)),
                        vsubq_f32(vec_in.val[3], vdupq_n_f32(sum)),
                    };
                    normalized_value = convert_float_to_int<float32x4x4_t, int_vec_type>(sub);
                }
                else
                {
                    float32x4x4_t mul =
                    {
                        vmulq_f32(vec_in.val[0], vdupq_n_f32(sum_inversed)),
                        vmulq_f32(vec_in.val[1], vdupq_n_f32(sum_inversed)),
                        vmulq_f32(vec_in.val[2], vdupq_n_f32(sum_inversed)),
                        vmulq_f32(vec_in.val[3], vdupq_n_f32(sum_inversed)),
                    };

                    if(is_qasymm8_signed)
                    {
                        const auto offset_vec = wrapper::vdup_n(128.f, wrapper::traits::vector_128_tag{});
                        mul.val[0]            = wrapper::vsub(mul.val[0], offset_vec);
                        mul.val[1]            = wrapper::vsub(mul.val[1], offset_vec);
                        mul.val[2]            = wrapper::vsub(mul.val[2], offset_vec);
                        mul.val[3]            = wrapper::vsub(mul.val[3], offset_vec);
                    }

                    normalized_value = convert_float_to_int<float32x4x4_t, int_vec_type>(mul);
                }
                wrapper::vstore(out_ptr + x, normalized_value);
            }
            /* Run remaining elements */
            for(; x < input_width; ++x)
            {
                if(is_log)
                {
                    out_ptr[x] = utils::cast::saturate_cast<T>(tmp_ptr[x] - sum);
                }
                else
                {
                    out_ptr[x] = utils::cast::saturate_cast<T>((tmp_ptr[x] * sum_inversed) - (is_qasymm8_signed ? 128.f : 0));
                }
            }
        }
    },
    in_it, max_it, out_it);
}

template <typename T>
void neon_softmax_logits_1d_float(const ITensor *in, const ITensor *max, void *const tmp,
                                  ITensor *out, const float beta, bool is_log, const Window &window)
{
    const int start_x     = in->info()->valid_region().anchor.x();
    const int input_width = in->info()->valid_region().shape.x();

    Iterator in_it(in, window);
    Iterator max_it(max, window);
    Iterator out_it(out, window);

    /** Neon vector tag type. */
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

#endif /* SRC_CORE_NEON_KERNELS_SOFTMAX_LIST_H */
