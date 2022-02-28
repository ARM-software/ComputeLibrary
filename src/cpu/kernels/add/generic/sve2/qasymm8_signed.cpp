/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/SVEMath.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
void add_qasymm8_signed_sve2(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(policy);

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();

    const UniformQuantizationInfo iq1_info = src0->info()->quantization_info().uniform();
    const UniformQuantizationInfo iq2_info = src1->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info  = dst->info()->quantization_info().uniform();

    const auto invvscaleo = svdup_n_f32(1.f / oq_info.scale);
    const auto voffseto   = svdup_n_f32(oq_info.offset);

    if(is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? src1 : src0;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src1 : src0;
        const auto     all_true_pg          = svptrue_b8();

        const auto vscale1  = is_broadcast_input_2 ? svdup_n_f32(iq1_info.scale) : svdup_n_f32(iq2_info.scale);
        const auto vscale2  = is_broadcast_input_2 ? svdup_n_f32(iq2_info.scale) : svdup_n_f32(iq1_info.scale);
        const auto voffset1 = is_broadcast_input_2 ? svdup_n_s32(iq1_info.offset) : svdup_n_s32(iq2_info.offset);
        const auto voffset2 = is_broadcast_input_2 ? svdup_n_s32(iq2_info.offset) : svdup_n_s32(iq1_info.offset);

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(dst, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const int8_t *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<int8_t *>(output.ptr());

            const int8_t broadcast_value     = *reinterpret_cast<const int8_t *>(broadcast_input.ptr());
            const auto   broadcast_value_vec = svdup_n_s8(broadcast_value);

            int        x    = window_start_x;
            svbool_t   pg   = svwhilelt_b8(x, window_end_x);
            const auto bf_0 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlb_s32(svmovlb_s16(broadcast_value_vec)), voffset2)), vscale2);
            const auto bf_1 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlt_s32(svmovlb_s16(broadcast_value_vec)), voffset2)), vscale2);
            const auto bf_2 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlb_s32(svmovlt_s16(broadcast_value_vec)), voffset2)), vscale2);
            const auto bf_3 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlt_s32(svmovlt_s16(broadcast_value_vec)), voffset2)), vscale2);

            do
            {
                const auto a    = svld1_s8(pg, non_broadcast_input_ptr + x);
                const auto af_0 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlb_s32(svmovlb_s16(a)), voffset1)), vscale1);
                const auto af_1 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlt_s32(svmovlb_s16(a)), voffset1)), vscale1);
                const auto af_2 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlb_s32(svmovlt_s16(a)), voffset1)), vscale1);
                const auto af_3 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlt_s32(svmovlt_s16(a)), voffset1)), vscale1);

                const auto rf_0 = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffseto, svadd_f32_z(pg, af_0, bf_0), invvscaleo));
                const auto rf_1 = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffseto, svadd_f32_z(pg, af_1, bf_1), invvscaleo));
                const auto rf_2 = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffseto, svadd_f32_z(pg, af_2, bf_2), invvscaleo));
                const auto rf_3 = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffseto, svadd_f32_z(pg, af_3, bf_3), invvscaleo));

                const auto pa  = svqxtnt_s32(svqxtnb_s32(rf_0), rf_1);
                const auto pb  = svqxtnt_s32(svqxtnb_s32(rf_2), rf_3);
                const auto res = svqxtnt_s16(svqxtnb_s16(pa), pb);

                svst1_s8(pg, output_ptr + x, res);

                x += svcntb();
                pg = svwhilelt_b8(x, window_end_x);
            }
            while(svptest_any(all_true_pg, pg));
        },
        broadcast_input, non_broadcast_input, output);
    }
    else
    {
        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(src0, input1_win);
        Iterator input2(src1, input2_win);
        Iterator output(dst, win);

        const auto vscale1  = svdup_n_f32(iq1_info.scale);
        const auto vscale2  = svdup_n_f32(iq2_info.scale);
        const auto voffset1 = svdup_n_s32(iq1_info.offset);
        const auto voffset2 = svdup_n_s32(iq2_info.offset);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const int8_t *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const int8_t *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

            int      x  = window_start_x;
            svbool_t pg = svwhilelt_b8(x, window_end_x);
            do
            {
                const auto a = svld1_s8(pg, input1_ptr + x);
                const auto b = svld1_s8(pg, input2_ptr + x);

                const auto af_0 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlb_s32(svmovlb_s16(a)), voffset1)), vscale1);
                const auto af_1 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlt_s32(svmovlb_s16(a)), voffset1)), vscale1);
                const auto af_2 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlb_s32(svmovlt_s16(a)), voffset1)), vscale1);
                const auto af_3 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlt_s32(svmovlt_s16(a)), voffset1)), vscale1);

                const auto bf_0 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlb_s32(svmovlb_s16(b)), voffset2)), vscale2);
                const auto bf_1 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlt_s32(svmovlb_s16(b)), voffset2)), vscale2);
                const auto bf_2 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlb_s32(svmovlt_s16(b)), voffset2)), vscale2);
                const auto bf_3 = svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlt_s32(svmovlt_s16(b)), voffset2)), vscale2);

                const auto rf_0 = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffseto, svadd_f32_z(pg, af_0, bf_0), invvscaleo));
                const auto rf_1 = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffseto, svadd_f32_z(pg, af_1, bf_1), invvscaleo));
                const auto rf_2 = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffseto, svadd_f32_z(pg, af_2, bf_2), invvscaleo));
                const auto rf_3 = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffseto, svadd_f32_z(pg, af_3, bf_3), invvscaleo));

                const auto pa  = svqxtnt_s32(svqxtnb_s32(rf_0), rf_1);
                const auto pb  = svqxtnt_s32(svqxtnb_s32(rf_2), rf_3);
                const auto res = svqxtnt_s16(svqxtnb_s16(pa), pb);

                svst1_s8(pg, output_ptr + x, res);

                x += svcntb();
                pg = svwhilelt_b8(x, window_end_x);
            }
            while(svptest_any(svptrue_b8(), pg));
        },
        input1, input2, output);
    }
}
} // namespace cpu
} // namespace arm_compute
