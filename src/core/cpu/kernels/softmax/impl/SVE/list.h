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
#ifndef SRC_CORE_SVE_KERNELS_SOFTMAX_LIST_H
#define SRC_CORE_SVE_KERNELS_SOFTMAX_LIST_H

#if defined(__ARM_FEATURE_SVE)
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/SVEMath.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
namespace
{
#if defined(__ARM_FEATURE_SVE2)
template <typename int_vec_type>
int_vec_type convert_float_to_int(const svfloat32_t &in_0, const svfloat32_t &in_1, const svfloat32_t &in_2, const svfloat32_t &in_3);

template <>
svuint8_t convert_float_to_int<svuint8_t>(const svfloat32_t &in_0, const svfloat32_t &in_1, const svfloat32_t &in_2, const svfloat32_t &in_3)
{
    svuint8_t  out;
    const auto all_true_pg = svptrue_b32();
    auto       tmp_0       = svcvt_u32_f32_z(all_true_pg, in_0);
    auto       tmp_1       = svcvt_u32_f32_z(all_true_pg, in_1);
    auto       tmp_2       = svcvt_u32_f32_z(all_true_pg, in_2);
    auto       tmp_3       = svcvt_u32_f32_z(all_true_pg, in_3);

    auto tmp_16_0 = svqxtnt_u32(svqxtnb_u32(tmp_0), tmp_1);
    auto tmp_16_1 = svqxtnt_u32(svqxtnb_u32(tmp_2), tmp_3);

    auto tmp_16_uzp_0 = svuzp1(tmp_16_0, tmp_16_0);
    auto tmp_16_uzp_1 = svuzp2(tmp_16_0, tmp_16_0);
    auto tmp_16_uzp_2 = svuzp1(tmp_16_1, tmp_16_1);
    auto tmp_16_uzp_3 = svuzp2(tmp_16_1, tmp_16_1);

    auto pg = svwhilelt_b16_s32(0, svcnth() / 2);

    tmp_16_0 = svsplice(pg, tmp_16_uzp_0, tmp_16_uzp_1);
    tmp_16_1 = svsplice(pg, tmp_16_uzp_2, tmp_16_uzp_3);

    out = svqxtnt_u16(svqxtnb_u16(tmp_16_0), tmp_16_1);

    auto out_uzp_0 = svuzp1(out, out);
    auto out_uzp_1 = svuzp2(out, out);

    pg  = svwhilelt_b8_s32(0, svcntb() / 2);
    out = svsplice(pg, out_uzp_0, out_uzp_1);

    return out;
}

template <>
svint8_t convert_float_to_int<svint8_t>(const svfloat32_t &in_0, const svfloat32_t &in_1, const svfloat32_t &in_2, const svfloat32_t &in_3)
{
    svint8_t   out;
    const auto all_true_pg = svptrue_b32();
    auto       tmp_0       = svcvt_s32_f32_z(all_true_pg, in_0);
    auto       tmp_1       = svcvt_s32_f32_z(all_true_pg, in_1);
    auto       tmp_2       = svcvt_s32_f32_z(all_true_pg, in_2);
    auto       tmp_3       = svcvt_s32_f32_z(all_true_pg, in_3);

    auto tmp_16_0 = svqxtnt_s32(svqxtnb_s32(tmp_0), tmp_1);
    auto tmp_16_1 = svqxtnt_s32(svqxtnb_s32(tmp_2), tmp_3);

    auto tmp_16_uzp_0 = svuzp1(tmp_16_0, tmp_16_0);
    auto tmp_16_uzp_1 = svuzp2(tmp_16_0, tmp_16_0);
    auto tmp_16_uzp_2 = svuzp1(tmp_16_1, tmp_16_1);
    auto tmp_16_uzp_3 = svuzp2(tmp_16_1, tmp_16_1);

    auto pg = svwhilelt_b16_s32(0, svcnth() / 2);

    tmp_16_0 = svsplice(pg, tmp_16_uzp_0, tmp_16_uzp_1);
    tmp_16_1 = svsplice(pg, tmp_16_uzp_2, tmp_16_uzp_3);

    out = svqxtnt_s16(svqxtnb_s16(tmp_16_0), tmp_16_1);

    auto out_uzp_0 = svuzp1(out, out);
    auto out_uzp_1 = svuzp2(out, out);

    pg  = svwhilelt_b8_s32(0, svcntb() / 2);
    out = svsplice(pg, out_uzp_0, out_uzp_1);

    return out;
}
#endif /* defined(__ARM_FEATURE_SVE2) */
} // namespace

template <typename ScalarType>
void sve_logits_1d_max(const ITensor *in, ITensor *out, const Window &window)
{
    const auto all_true_pg    = wrapper::svptrue<ScalarType>();
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator input(in, win);
    Iterator output(out, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        // Get pointers
        const auto in_ptr  = reinterpret_cast<const ScalarType *>(input.ptr());
        const auto out_ptr = reinterpret_cast<ScalarType *>(output.ptr());

        // Init max value
        auto vec_max = wrapper::svdup_n(support::cpp11::lowest<ScalarType>());

        int      x  = window_start_x;
        svbool_t pg = wrapper::svwhilelt<ScalarType>(x, window_end_x);
        do
        {
            const auto current_value = svld1(pg, in_ptr + x);
            vec_max                  = svmax_m(pg, vec_max, current_value);

            x += wrapper::svcnt<ScalarType>();
            pg = wrapper::svwhilelt<ScalarType>(x, window_end_x);
        }
        while(svptest_any(all_true_pg, pg));

        auto max_val = svmaxv(all_true_pg, vec_max);

        *out_ptr = max_val;
    },
    input, output);
}

#if defined(__ARM_FEATURE_SVE2)
template <typename ScalarType>
void sve_softmax_logits_1d_quantized(const ITensor *in, const ITensor *max, void *const tmp,
                                     ITensor *out, float beta, bool is_log, const Window &window)
{
    const int start_x     = in->info()->valid_region().anchor.x();
    const int input_width = in->info()->valid_region().shape.x();

    const float scale_beta     = -beta * in->info()->quantization_info().uniform().scale;
    const auto  scale_beta_vec = svdup_n_f32(scale_beta);

    Iterator   in_it(in, window);
    Iterator   max_it(max, window);
    Iterator   out_it(out, window);
    const auto all_true_pg = wrapper::svptrue<ScalarType>();
    using SVEType          = typename wrapper::traits::sve_vector<ScalarType>::type;

    const int inc_1 = static_cast<int>(svcntw());
    const int inc_2 = static_cast<int>(2 * svcntw());
    const int inc_3 = static_cast<int>(3 * svcntw());

    execute_window_loop(window, [&](const Coordinates &)
    {
        /* Get pointers */
        const auto in_ptr  = reinterpret_cast<const ScalarType *>(in_it.ptr()) + start_x;
        const auto out_ptr = reinterpret_cast<ScalarType *>(out_it.ptr()) + start_x;
        const auto tmp_ptr = reinterpret_cast<float *>(tmp);

        float sum{};

        /* Compute exponentials and sum */
        {
            /* Get max value */
            const auto max_val = *reinterpret_cast<const ScalarType *>(max_it.ptr());
            const auto vec_max = wrapper::svdup_n(max_val);

            /* Init sum to zero */
            auto vec_sum_0 = svdup_n_f32(0.f);
            auto vec_sum_1 = svdup_n_f32(0.f);
            auto vec_sum_2 = svdup_n_f32(0.f);
            auto vec_sum_3 = svdup_n_f32(0.f);

            /* Loop over row and compute exponentials and sum */
            int      x    = 0;
            svbool_t pg   = wrapper::svwhilelt<ScalarType>(x, input_width);
            svbool_t pg_0 = svunpklo(svunpklo(pg));
            svbool_t pg_1 = svunpkhi(svunpklo(pg));
            svbool_t pg_2 = svunpklo(svunpkhi(pg));
            svbool_t pg_3 = svunpkhi(svunpkhi(pg));
            do
            {
                auto vec_elements = svld1(pg, in_ptr + x);
                vec_elements      = svsub_z(pg, vec_max, vec_elements);

                auto vec_elements_flt_0 = svcvt_f32_z(pg_0, svunpklo(svunpklo(vec_elements)));
                auto vec_elements_flt_1 = svcvt_f32_z(pg_1, svunpkhi(svunpklo(vec_elements)));
                auto vec_elements_flt_2 = svcvt_f32_z(pg_2, svunpklo(svunpkhi(vec_elements)));
                auto vec_elements_flt_3 = svcvt_f32_z(pg_3, svunpkhi(svunpkhi(vec_elements)));

                if(is_log)
                {
                    vec_elements_flt_0 = svmul_f32_z(pg_0, vec_elements_flt_0, scale_beta_vec);
                    vec_elements_flt_1 = svmul_f32_z(pg_1, vec_elements_flt_1, scale_beta_vec);
                    vec_elements_flt_2 = svmul_f32_z(pg_2, vec_elements_flt_2, scale_beta_vec);
                    vec_elements_flt_3 = svmul_f32_z(pg_3, vec_elements_flt_3, scale_beta_vec);
                    vec_sum_0          = svadd_f32_m(pg_0, vec_sum_0, svexp_f32_z(pg_0, vec_elements_flt_0));
                    vec_sum_1          = svadd_f32_m(pg_1, vec_sum_1, svexp_f32_z(pg_1, vec_elements_flt_1));
                    vec_sum_2          = svadd_f32_m(pg_2, vec_sum_2, svexp_f32_z(pg_2, vec_elements_flt_2));
                    vec_sum_3          = svadd_f32_m(pg_3, vec_sum_3, svexp_f32_z(pg_3, vec_elements_flt_3));
                }
                else
                {
                    vec_elements_flt_0 = svexp_f32_z(pg_0, svmul_f32_z(pg_0, vec_elements_flt_0, scale_beta_vec));
                    vec_elements_flt_1 = svexp_f32_z(pg_1, svmul_f32_z(pg_1, vec_elements_flt_1, scale_beta_vec));
                    vec_elements_flt_2 = svexp_f32_z(pg_2, svmul_f32_z(pg_2, vec_elements_flt_2, scale_beta_vec));
                    vec_elements_flt_3 = svexp_f32_z(pg_3, svmul_f32_z(pg_3, vec_elements_flt_3, scale_beta_vec));
                    vec_sum_0          = svadd_f32_m(pg_0, vec_sum_0, vec_elements_flt_0);
                    vec_sum_1          = svadd_f32_m(pg_1, vec_sum_1, vec_elements_flt_1);
                    vec_sum_2          = svadd_f32_m(pg_2, vec_sum_2, vec_elements_flt_2);
                    vec_sum_3          = svadd_f32_m(pg_3, vec_sum_3, vec_elements_flt_3);
                }

                svst1_f32(pg_0, tmp_ptr + x, vec_elements_flt_0);
                svst1_f32(pg_1, tmp_ptr + x + inc_1, vec_elements_flt_1);
                svst1_f32(pg_2, tmp_ptr + x + inc_2, vec_elements_flt_2);
                svst1_f32(pg_3, tmp_ptr + x + inc_3, vec_elements_flt_3);

                x += wrapper::svcnt<ScalarType>();
                pg   = wrapper::svwhilelt<ScalarType>(x, input_width);
                pg_0 = svunpklo(svunpklo(pg));
                pg_1 = svunpkhi(svunpklo(pg));
                pg_2 = svunpklo(svunpkhi(pg));
                pg_3 = svunpkhi(svunpkhi(pg));
            }
            while(svptest_any(all_true_pg, pg));

            /* Reduce sum */
            const auto vec_sum = svadd_f32_z(all_true_pg, svadd_f32_z(all_true_pg, vec_sum_0, vec_sum_1), svadd_f32_z(all_true_pg, vec_sum_2, vec_sum_3));
            sum                = svaddv_f32(all_true_pg, vec_sum);

            /* Run remaining elements */
            x = 0;
            if(is_log)
            {
                sum = std::log(sum);
            }
            else
            {
                sum = 256.f / sum;
            }
        }

        /* Normalize exponentials */
        {
            constexpr bool is_qasymm8_signed = std::is_same<ScalarType, qasymm8_signed_t>::value;
            /* Loop over row and compute softmax */
            int      x    = 0;
            svbool_t pg   = wrapper::svwhilelt<ScalarType>(x, input_width);
            svbool_t pg_0 = svunpklo(svunpklo(pg));
            svbool_t pg_1 = svunpkhi(svunpklo(pg));
            svbool_t pg_2 = svunpklo(svunpkhi(pg));
            svbool_t pg_3 = svunpkhi(svunpkhi(pg));
            do
            {
                auto vec_in_0 = svld1_f32(pg_0, tmp_ptr + x);
                auto vec_in_1 = svld1_f32(pg_1, tmp_ptr + x + inc_1);
                auto vec_in_2 = svld1_f32(pg_2, tmp_ptr + x + inc_2);
                auto vec_in_3 = svld1_f32(pg_3, tmp_ptr + x + inc_3);

                svfloat32_t res_0{};
                svfloat32_t res_1{};
                svfloat32_t res_2{};
                svfloat32_t res_3{};

                if(is_log)
                {
                    res_0 = svsub_f32_z(pg_0, vec_in_0, svdup_n_f32(sum));
                    res_1 = svsub_f32_z(pg_1, vec_in_1, svdup_n_f32(sum));
                    res_2 = svsub_f32_z(pg_2, vec_in_2, svdup_n_f32(sum));
                    res_3 = svsub_f32_z(pg_3, vec_in_3, svdup_n_f32(sum));
                }
                else
                {
                    res_0 = svmul_f32_z(pg_0, vec_in_0, svdup_n_f32(sum));
                    res_1 = svmul_f32_z(pg_1, vec_in_1, svdup_n_f32(sum));
                    res_2 = svmul_f32_z(pg_2, vec_in_2, svdup_n_f32(sum));
                    res_3 = svmul_f32_z(pg_3, vec_in_3, svdup_n_f32(sum));

                    if(is_qasymm8_signed)
                    {
                        const auto offset_vec = svdup_n_f32(128.f);
                        res_0                 = svsub_z(pg_0, vec_in_0, offset_vec);
                        res_1                 = svsub_z(pg_1, vec_in_1, offset_vec);
                        res_2                 = svsub_z(pg_2, vec_in_2, offset_vec);
                        res_3                 = svsub_z(pg_3, vec_in_3, offset_vec);
                    }
                }

                // Store value
                const auto out = convert_float_to_int<SVEType>(res_0, res_1, res_2, res_3);
                svst1(pg, out_ptr + x, out);
                x += wrapper::svcnt<ScalarType>();
                pg   = wrapper::svwhilelt<ScalarType>(x, input_width);
                pg_0 = svunpklo(svunpklo(pg));
                pg_1 = svunpkhi(svunpklo(pg));
                pg_2 = svunpklo(svunpkhi(pg));
                pg_3 = svunpkhi(svunpkhi(pg));
            }
            while(svptest_any(all_true_pg, pg));
        }
    },
    in_it, max_it, out_it);
}
#endif /* defined(__ARM_FEATURE_SVE2) */

template <typename ScalarType>
void sve_softmax_logits_1d_float(const ITensor *in, const ITensor *max, void *const tmp,
                                 ITensor *out, const float beta, bool is_log, const Window &window)
{
    const int start_x     = in->info()->valid_region().anchor.x();
    const int input_width = in->info()->valid_region().shape.x();

    Iterator in_it(in, window);
    Iterator max_it(max, window);
    Iterator out_it(out, window);

    const auto all_true_pg = wrapper::svptrue<ScalarType>();

    execute_window_loop(window, [&](const Coordinates &)
    {
        /* Get pointers */
        const auto in_ptr  = reinterpret_cast<const ScalarType *>(in_it.ptr()) + start_x;
        const auto out_ptr = reinterpret_cast<ScalarType *>(out_it.ptr()) + start_x;
        const auto tmp_ptr = reinterpret_cast<ScalarType *>(tmp);

        ScalarType sum{ 0 };

        /* Compute exponentials and sum */
        {
            /* Get max value */
            const auto max_val = *reinterpret_cast<const ScalarType *>(max_it.ptr());
            const auto vec_max = wrapper::svdup_n(max_val);

            /* Init sum to zero */
            auto vec_sum = wrapper::svdup_n(static_cast<ScalarType>(0));

            /* Loop over row and compute exponentials and sum */
            int      x  = 0;
            svbool_t pg = wrapper::svwhilelt<ScalarType>(x, input_width);
            do
            {
                auto vec_elements = svld1(pg, in_ptr + x);
                vec_elements      = svsub_z(pg, vec_elements, vec_max);
                if(is_log)
                {
                    vec_elements = svmul_z(pg, vec_elements, wrapper::svdup_n(static_cast<ScalarType>(beta)));
                    vec_sum      = svadd_m(pg, vec_sum, wrapper::svexp_z(pg, vec_elements));
                }
                else
                {
                    vec_elements = wrapper::svexp_z(pg, svmul_z(pg, vec_elements, wrapper::svdup_n(static_cast<ScalarType>(beta))));
                    vec_sum      = svadd_m(pg, vec_sum, vec_elements);
                }
                svst1(pg, tmp_ptr + x, vec_elements);

                x += wrapper::svcnt<ScalarType>();
                pg = wrapper::svwhilelt<ScalarType>(x, input_width);
            }
            while(svptest_any(all_true_pg, pg));

            /* Reduce sum */
            sum = svaddv(all_true_pg, vec_sum);

            if(is_log)
            {
                sum = static_cast<ScalarType>(std::log(sum));
            }
            else
            {
                sum = ScalarType(1) / sum;
            }
        }

        /* Normalize exponentials */
        {
            /* Loop over row and compute softmax */
            int      x  = 0;
            svbool_t pg = wrapper::svwhilelt<ScalarType>(x, input_width);
            do
            {
                auto vec_in           = svld1(pg, tmp_ptr + x);
                auto normalized_value = wrapper::svdup_n(static_cast<ScalarType>(0));
                if(is_log)
                {
                    normalized_value = svsub_z(pg, vec_in, wrapper::svdup_n(static_cast<ScalarType>(sum)));
                }
                else
                {
                    normalized_value = svmul_z(pg, vec_in, wrapper::svdup_n(static_cast<ScalarType>(sum)));
                }
                svst1(pg, out_ptr + x, normalized_value);

                x += wrapper::svcnt<ScalarType>();
                pg = wrapper::svwhilelt<ScalarType>(x, input_width);
            }
            while(svptest_any(all_true_pg, pg));
        }
    },
    in_it, max_it, out_it);
}

} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_SVE) */

#endif /* SRC_CORE_SVE_KERNELS_SOFTMAX_LIST_H */
