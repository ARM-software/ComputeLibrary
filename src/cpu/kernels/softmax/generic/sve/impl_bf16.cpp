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

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"

#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"

namespace arm_compute
{
namespace cpu
{

void sve_softmax_bf16(const ITensor *in,
                      void *const    tmp,
                      ITensor       *out,
                      const float    beta,
                      int            axis,
                      const Window  &window,
                      const void    *lut_ptr)
{
    ARM_COMPUTE_UNUSED(tmp);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(axis);

    ARM_COMPUTE_ERROR_ON_NULLPTR(lut_ptr);
    const auto lut_fp16_ptr = reinterpret_cast<const uint16_t *>(lut_ptr);

    const int start_x     = in->info()->valid_region().anchor.x();
    const int input_width = in->info()->valid_region().shape.x();

    Iterator in_it(in, window);
    Iterator out_it(out, window);

    const auto all_true_pg     = wrapper::svptrue<bfloat16_t>();
    const auto all_true_pg_f32 = wrapper::svptrue<float32_t>();
    const auto all_true_pg_u32 = wrapper::svptrue<uint32_t>();
    const int  vec_count       = wrapper::svcnt<bfloat16_t>();

    execute_window_loop(
        window,
        [&](const Coordinates &)
        {
            /* Get pointers */
            const auto in_ptr  = reinterpret_cast<const uint16_t *>(in_it.ptr()) + start_x;
            const auto out_ptr = reinterpret_cast<uint16_t *>(out_it.ptr()) + start_x;

            /* Compute Max: unlike in the conventional Softmax, we subtract the maximum value in the axis from each input (both in numerator and denominator) to reduce overall magnitude while maintaining correctness of output */
            float32_t max_val(std::numeric_limits<float32_t>::lowest());
            {
                auto vec_max = wrapper::svdup_n(support::cpp11::lowest<float32_t>());

                int            x         = 0;
                svbool_t       pg        = wrapper::svwhilelt<bfloat16_t>(x, input_width);
                const svbool_t p_32_true = svptrue_b32();

                svbool_t pg_u16      = wrapper::svwhilelt<uint16_t>(x, input_width);
                svbool_t pg_f32_low  = svunpklo(pg_u16);
                svbool_t pg_f32_high = svunpkhi(pg_u16);
                do
                {
                    const svuint16_t current_value_bf16 = svld1(pg, in_ptr + x);

                    svuint32_t current_value_u32_low  = svunpklo(current_value_bf16);
                    svuint32_t current_value_u32_high = svunpkhi(current_value_bf16);

                    current_value_u32_low  = svlsl_n_u32_z(p_32_true, current_value_u32_low, 16);
                    current_value_u32_high = svlsl_n_u32_z(p_32_true, current_value_u32_high, 16);

                    const svfloat32_t current_value_fp32_low  = svreinterpret_f32_u32(current_value_u32_low);
                    const svfloat32_t current_value_fp32_high = svreinterpret_f32_u32(current_value_u32_high);

                    vec_max = svmax_m(pg_f32_low, vec_max, current_value_fp32_low);
                    vec_max = svmax_m(pg_f32_high, vec_max, current_value_fp32_high);

                    x += vec_count;
                    pg          = wrapper::svwhilelt<bfloat16_t>(x, input_width);
                    pg_u16      = wrapper::svwhilelt<uint16_t>(x, input_width);
                    pg_f32_low  = svunpklo(pg_u16);
                    pg_f32_high = svunpkhi(pg_u16);
                } while (svptest_any(all_true_pg, pg));

                // Reduce vec to single max value
                max_val = svmaxv(all_true_pg, vec_max);
            }
            float32_t sum(0.f);
            {
                /* Init sum to zero */
                svfloat32_t       vec_sum = wrapper::svdup_n(static_cast<float32_t>(0));
                const svfloat32_t vec_max = wrapper::svdup_n(max_val);

                /* Loop over row and compute exponentials and sum */
                int x = 0;

                svbool_t pg     = wrapper::svwhilelt<bfloat16_t>(x, input_width);
                svbool_t pg_u16 = wrapper::svwhilelt<uint16_t>(x, input_width);

                svbool_t pg_f32_low  = svunpklo(pg_u16);
                svbool_t pg_f32_high = svunpkhi(pg_u16);

                do
                {
                    const svuint16_t vec_elements = svld1(pg, in_ptr + x);

                    svuint32_t current_value_u32_low  = svunpklo(vec_elements);
                    svuint32_t current_value_u32_high = svunpkhi(vec_elements);

                    current_value_u32_low  = svlsl_n_u32_z(all_true_pg_u32, current_value_u32_low, 16);
                    current_value_u32_high = svlsl_n_u32_z(all_true_pg_u32, current_value_u32_high, 16);

                    const svfloat32_t current_value_fp32_low  = svreinterpret_f32_u32(current_value_u32_low);
                    const svfloat32_t current_value_fp32_high = svreinterpret_f32_u32(current_value_u32_high);

                    /* The aforementioned (on line 71) subtraction to reduce magnitude below, effectively a division by the exponentiated maximum value in the current axis */
                    svfloat32_t vec_subbed_low_fp32  = svsub_z(pg_f32_low, current_value_fp32_low, vec_max);
                    svfloat32_t vec_subbed_high_fp32 = svsub_z(pg_f32_high, current_value_fp32_high, vec_max);

                    const svuint16_t vec_subbed_low_uint16 = svreinterpret_u16_u32(
                        svlsr_n_u32_z(all_true_pg_u32, svreinterpret_u32_f32(vec_subbed_low_fp32), 16));
                    const svuint16_t vec_subbed_high_uint16 = svreinterpret_u16_u32(
                        svlsr_n_u32_z(all_true_pg_u32, svreinterpret_u32_f32(vec_subbed_high_fp32), 16));

                    // Use LUT to get x : e^x*b
                    const svuint32_t loaded_exp_16bit_values_low = svld1uh_gather_index_u32(
                        pg_f32_low, lut_fp16_ptr, svreinterpret_u32_u16(vec_subbed_low_uint16));
                    const svuint32_t loaded_exp_16bit_values_high = svld1uh_gather_index_u32(
                        pg_f32_high, lut_fp16_ptr, svreinterpret_u32_u16(vec_subbed_high_uint16));

                    // Recombine LUT values
                    const svuint16_t exp_bf16 = svuzp1(svreinterpret_u16_u32(loaded_exp_16bit_values_low),
                                                       svreinterpret_u16_u32(loaded_exp_16bit_values_high));

                    /* This store is not the final output value, the output tensor is used to store the numerator/dividend of the softmax operation for use in the final step
                    as there are likely not enough registers for a whole axis' values */
                    svst1(pg, out_ptr + x, exp_bf16);

                    svuint32_t exp_u32_low  = svunpklo(exp_bf16);
                    svuint32_t exp_u32_high = svunpkhi(exp_bf16);

                    exp_u32_low  = svlsl_n_u32_z(all_true_pg_u32, exp_u32_low, 16);
                    exp_u32_high = svlsl_n_u32_z(all_true_pg_u32, exp_u32_high, 16);

                    const svfloat32_t exp_fp32_low  = svreinterpret_f32_u32(exp_u32_low);
                    const svfloat32_t exp_fp32_high = svreinterpret_f32_u32(exp_u32_high);

                    vec_sum = svadd_m(pg_f32_low, vec_sum, exp_fp32_low);
                    vec_sum = svadd_m(pg_f32_high, vec_sum, exp_fp32_high);

                    x += vec_count;
                    pg     = wrapper::svwhilelt<bfloat16_t>(x, input_width);
                    pg_u16 = wrapper::svwhilelt<uint16_t>(x, input_width);

                    pg_f32_low  = svunpklo(pg_u16);
                    pg_f32_high = svunpkhi(pg_u16);
                } while (svptest_any(all_true_pg, pg));

                /* Reduce sum */
                sum = svaddv(all_true_pg_f32, vec_sum);
                sum = float32_t(1) / sum;
            }

            /* Normalize exponentials */
            {
                /* Loop over row and compute softmax */
                int      x           = 0;
                svbool_t pg          = wrapper::svwhilelt<bfloat16_t>(x, input_width);
                svbool_t pg_u16      = wrapper::svwhilelt<uint16_t>(x, input_width);
                svbool_t pg_f32_low  = svunpklo(pg_u16);
                svbool_t pg_f32_high = svunpkhi(pg_u16);

                do
                {
                    const svuint16_t vec_in = svld1(pg, out_ptr + x);

                    svuint32_t current_value_u32_low  = svunpklo(vec_in);
                    svuint32_t current_value_u32_high = svunpkhi(vec_in);

                    current_value_u32_low  = svlsl_n_u32_z(all_true_pg_u32, current_value_u32_low, 16);
                    current_value_u32_high = svlsl_n_u32_z(all_true_pg_u32, current_value_u32_high, 16);

                    const svfloat32_t current_value_fp32_low  = svreinterpret_f32_u32(current_value_u32_low);
                    const svfloat32_t current_value_fp32_high = svreinterpret_f32_u32(current_value_u32_high);

                    const svfloat32_t normalized_value_fp32_low =
                        svmul_z(pg_f32_low, current_value_fp32_low, wrapper::svdup_n(sum));
                    const svfloat32_t normalized_value_fp32_high =
                        svmul_z(pg_f32_high, current_value_fp32_high, wrapper::svdup_n(sum));

                    const svuint16_t normalized_value_low_uint16 = svreinterpret_u16_u32(
                        svlsr_n_u32_z(all_true_pg_u32, svreinterpret_u32_f32(normalized_value_fp32_low), 16));
                    const svuint16_t normalized_value_high_uint16 = svreinterpret_u16_u32(
                        svlsr_n_u32_z(all_true_pg_u32, svreinterpret_u32_f32(normalized_value_fp32_high), 16));

                    const svuint16_t normalized_value_bf16 =
                        svuzp1(normalized_value_low_uint16, normalized_value_high_uint16);

                    svst1(pg, out_ptr + x, normalized_value_bf16);

                    x += vec_count;
                    pg          = wrapper::svwhilelt<bfloat16>(x, input_width);
                    pg_u16      = wrapper::svwhilelt<uint16_t>(x, input_width);
                    pg_f32_low  = svunpklo(pg_u16);
                    pg_f32_high = svunpkhi(pg_u16);
                } while (svptest_any(all_true_pg, pg));
            }
        },
        in_it, out_it);
}
} // namespace cpu
} // namespace arm_compute
