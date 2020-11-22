/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/SVEMath.h"
#include "src/core/common/StdTypes.h"
#include "src/core/common/Validate.h"

#include <cmath>
#include <cstddef>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
void fp32_sve_activation(const ITensor *src, ITensor *dst, const ActivationLayerInfo &act_info, const Window &window)
{
    const auto                                    window_start_x = static_cast<int>(window.x().start());
    const auto                                    window_end_x   = static_cast<int>(window.x().end());
    const ActivationLayerInfo::ActivationFunction act            = act_info.activation();

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);

    const auto const_1     = svdup_n_f32(1.f);
    const auto const_0     = svdup_n_f32(0.f);
    const auto const_6     = svdup_n_f32(6.f);
    const auto const_3     = svdup_n_f32(3.f);
    const auto const_inv_6 = svdup_n_f32(0.166666667f);

    const auto va = svdup_n_f32(act_info.a());
    const auto vb = svdup_n_f32(act_info.b());
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
        const auto output_ptr = reinterpret_cast<float *>(output.ptr());

        svfloat32_t tmp;

        // Compute S elements per iteration
        int      x  = window_start_x;
        svbool_t pg = svwhilelt_b32(x, window_end_x);
        do
        {
            const auto vin = svld1_f32(pg, input_ptr + x);
            switch(act)
            {
                case ActivationLayerInfo::ActivationFunction::ABS:
                    tmp = svabs_f32_z(pg, vin);
                    break;
                case ActivationLayerInfo::ActivationFunction::LINEAR:
                    tmp = svmla_f32_z(pg, vb, va, vin);
                    break;
                case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                    tmp = svinv_f32_z(pg, svadd_f32_z(pg, const_1, svexp_f32_z(pg, svneg_f32_z(pg, vin))));
                    break;
                case ActivationLayerInfo::ActivationFunction::RELU:
                    tmp = svmax_f32_z(pg, const_0, vin);
                    break;
                case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                    tmp = svmin_f32_z(pg, va, svmax_f32_z(pg, const_0, vin));
                    break;
                case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
                    tmp = svmin_f32_z(pg, va, svmax_f32_z(pg, vb, vin));
                    break;
                case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
                    tmp = svadd_f32_z(pg, svmul_f32_z(pg, svmin_f32_z(pg, vin, const_0), va), svmax_f32_z(pg, vin, const_0));
                    break;
                case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                    tmp = svlog_f32_z(pg, svadd_f32_z(pg, const_1, svexp_f32_z(pg, vin)));
                    break;
                case ActivationLayerInfo::ActivationFunction::ELU:
                    tmp = svsel_f32(svcmpgt_f32(pg, vin, const_0), vin, svmul_f32_z(pg, va, svsub_f32_z(pg, svexp_f32_z(pg, vin), const_1)));
                    break;
                case ActivationLayerInfo::ActivationFunction::SQRT:
                    tmp = svsqrt_f32_z(pg, vin);
                    break;
                case ActivationLayerInfo::ActivationFunction::SQUARE:
                    tmp = svmul_f32_z(pg, vin, vin);
                    break;
                case ActivationLayerInfo::ActivationFunction::TANH:
                    tmp = svmul_f32_z(pg, va, svtanh_f32_z(pg, svmul_f32_z(pg, vb, vin)));
                    break;
                case ActivationLayerInfo::ActivationFunction::IDENTITY:
                    tmp = vin;
                    break;
                case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
                    tmp = svmul_f32_z(pg, vin, svmul_f32_z(pg, const_inv_6, svmin_f32_z(pg, const_6, svmax_f32_z(pg, const_0, svadd_f32_z(pg, vin, const_3)))));
                    break;
                default:
                    ARM_COMPUTE_ERROR("Unsupported activation function");
            }
            svst1_f32(pg, output_ptr + x, tmp);

            x += svcntw();
            pg = svwhilelt_b32(x, window_end_x);

        }
        while(svptest_any(svptrue_b32(), pg));
    },
    input, output);
}
} // namespace cpu
} // namespace arm_compute
#endif // __ARM_FEATURE_SVE