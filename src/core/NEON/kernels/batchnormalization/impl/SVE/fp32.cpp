/*
 * Copyright (c) 2020-2021,2023 Arm Limited.
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
#include "arm_compute/core/ActivationLayerInfo.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/SVEMath.h"

#include <cmath>
#include <cstddef>

#if defined(ARM_COMPUTE_ENABLE_SVE)
#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
void fp32_sve_batch_normalization(ITensor *src, ITensor *dst, const ITensor *mean, const ITensor *var, const ITensor *beta, const ITensor *gamma,
                                  float epsilon, ActivationLayerInfo &act_info, const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);

    const auto input_mean  = reinterpret_cast<const float *>(mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const float *>(var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (gamma != nullptr) ? reinterpret_cast<const float *>(gamma->ptr_to_element(Coordinates(0, 0))) : nullptr;
    const auto input_beta  = (beta != nullptr) ? reinterpret_cast<const float *>(beta->ptr_to_element(Coordinates(0, 0))) : nullptr;

    const auto epsilon_vec = svdup_n_f32(epsilon);
    const auto const_1     = svdup_n_f32(1.f);
    const auto const_0     = svdup_n_f32(0.f);
    const auto va          = svdup_n_f32(act_info.a());
    const auto vb          = svdup_n_f32(act_info.b());
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
        const auto output_ptr = reinterpret_cast<float *>(output.ptr());

        // Compute S elements per iteration
        int      x  = window_start_x;
        svbool_t pg = svwhilelt_b32(x, window_end_x);
        do
        {
            // Conctruct vectors
            const auto mean_vec  = svld1_f32(pg, input_mean + x);
            const auto var_vec   = svld1_f32(pg, input_var + x);
            const auto gamma_vec = (input_gamma != nullptr) ? svld1_f32(pg, input_gamma + x) : const_1;
            const auto beta_vec  = (input_beta != nullptr) ? svld1_f32(pg, input_beta + x) : const_0;

            // Calculate denominator
            const auto tmp         = svadd_f32_z(pg, var_vec, epsilon_vec);
            auto       denominator = svrsqrte_f32(tmp);
            denominator            = svmul_f32_z(pg, svrsqrts_f32(svmul_f32_z(pg, tmp, denominator), denominator), denominator);
            denominator            = svmul_f32_z(pg, svrsqrts_f32(svmul_f32_z(pg, tmp, denominator), denominator), denominator);

            // Calculate x bar
            const auto numerator = svsub_f32_z(pg, svld1_f32(pg, input_ptr + x), mean_vec);
            const auto x_bar     = svmul_f32_z(pg, numerator, denominator);
            auto       res       = svmla_f32_z(pg, beta_vec, x_bar, gamma_vec);

            // Perform fused activation
            if(act_info.enabled())
            {
                if(act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU)
                {
                    res = svmax_f32_z(pg, const_0, res);
                }
                else if(act_info.activation() == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
                {
                    res = svmin_f32_z(pg, va, svmax_f32_z(pg, const_0, res));
                }
                else if(act_info.activation() == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
                {
                    res = svmin_f32_z(pg, va, svmax_f32_z(pg, vb, res));
                }
            }

            // Store results
            svst1_f32(pg, output_ptr + x, res);

            x += svcntw();
            pg = svwhilelt_b32(x, window_end_x);
        }
        while(svptest_any(svptrue_b32(), pg));
    },
    input, output);
}
} // namespace cpu
} // namespace arm_compute
#endif // ENABLE_SVE
