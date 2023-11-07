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

#include "src/cpu/kernels/softmax/generic/sve/impl.h"

#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"

namespace arm_compute
{
namespace cpu
{
/// TODO: (COMPMID-6505) Similar to Neon(TM), this implementation be converted to
/// a single kernel that performs softmax operation. Leaving the SVE code here for
/// future references. Implementation for Neon(TM) is introduced in COMPMID-6500
template <typename ScalarType>
void sve_logits_1d_max(const ITensor *in, ITensor *out, const Window &window)
{
    const auto all_true_pg    = wrapper::svptrue<ScalarType>();
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win{window};
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator input(in, win);
    Iterator output(out, win);

    execute_window_loop(
        win,
        [&](const Coordinates &)
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
            } while (svptest_any(all_true_pg, pg));

            auto max_val = svmaxv(all_true_pg, vec_max);

            *out_ptr = max_val;
        },
        input, output);
}

template <typename ScalarType>
void sve_softmax_logits_1d_float(const ITensor *in,
                                 const ITensor *max,
                                 void *const    tmp,
                                 ITensor       *out,
                                 const float    beta,
                                 bool           is_log,
                                 const Window  &window)
{
    const int start_x     = in->info()->valid_region().anchor.x();
    const int input_width = in->info()->valid_region().shape.x();

    Iterator in_it(in, window);
    Iterator max_it(max, window);
    Iterator out_it(out, window);

    const auto all_true_pg = wrapper::svptrue<ScalarType>();

    execute_window_loop(
        window,
        [&](const Coordinates &)
        {
            /* Get pointers */
            const auto in_ptr  = reinterpret_cast<const ScalarType *>(in_it.ptr()) + start_x;
            const auto out_ptr = reinterpret_cast<ScalarType *>(out_it.ptr()) + start_x;
            const auto tmp_ptr = reinterpret_cast<ScalarType *>(tmp);

            ScalarType sum{0};

            /* Compute exponentials and sum */
            {
                /* Get max value */
                const auto max_val  = *reinterpret_cast<const ScalarType *>(max_it.ptr());
                const auto vec_max  = wrapper::svdup_n(max_val);
                const auto vec_beta = wrapper::svdup_n(static_cast<ScalarType>(beta));

                /* Init sum to zero */
                auto vec_sum = wrapper::svdup_n(static_cast<ScalarType>(0));

                /* Loop over row and compute exponentials and sum */
                int      x  = 0;
                svbool_t pg = wrapper::svwhilelt<ScalarType>(x, input_width);
                do
                {
                    auto vec_elements = svld1(pg, in_ptr + x);
                    vec_elements      = svmul_z(pg, svsub_z(pg, vec_elements, vec_max), vec_beta);
                    if (!is_log)
                    {
                        vec_elements = wrapper::svexp_z(pg, vec_elements);
                        vec_sum      = svadd_m(pg, vec_sum, vec_elements);
                    }
                    svst1(pg, tmp_ptr + x, vec_elements);

                    if (is_log)
                    {
                        vec_sum = svadd_m(pg, vec_sum, wrapper::svexp_z(pg, vec_elements));
                    }

                    x += wrapper::svcnt<ScalarType>();
                    pg = wrapper::svwhilelt<ScalarType>(x, input_width);
                } while (svptest_any(all_true_pg, pg));

                /* Reduce sum */
                sum = svaddv(all_true_pg, vec_sum);

                if (is_log)
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
                    if (is_log)
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
                } while (svptest_any(all_true_pg, pg));
            }
        },
        in_it, max_it, out_it);
}
} // namespace cpu
} // namespace arm_compute
