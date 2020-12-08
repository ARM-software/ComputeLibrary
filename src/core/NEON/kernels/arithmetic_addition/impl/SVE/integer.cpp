/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#if defined(__ARM_FEATURE_SVE)
#include "src/core/NEON/SVEMath.h"
#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
void arithmetic_addition_U8_U8_S16_sve(const ITensor *in1, const ITensor *in2, ITensor *out, const ConvertPolicy &policy, const Window &window)
{
    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(in1, input1_win);
    Iterator input2(in2, input2_win);
    Iterator output(out, win);

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const auto all_true_pg    = svptrue_b8();

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const uint8_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

        if(policy == ConvertPolicy::WRAP)
        {
            int      x    = window_start_x;
            svbool_t pg_u = svwhilelt_b8(x, window_end_x);
            svbool_t pg_0 = svwhilelt_b16(x, window_end_x);
            svbool_t pg_1 = svwhilelt_b16(x, static_cast<int>(window_end_x + svcnth()));
            do
            {
                const auto vin1 = svld1(pg_u, input1_ptr + x);
                const auto vin2 = svld1(pg_u, input2_ptr + x);

                const auto vin1_lo = svreinterpret_s16_u16(svunpklo(vin1));
                const auto vin1_hi = svreinterpret_s16_u16(svunpkhi(vin1));
                const auto vin2_lo = svreinterpret_s16_u16(svunpklo(vin2));
                const auto vin2_hi = svreinterpret_s16_u16(svunpkhi(vin2));
                svst1(pg_0, output_ptr + x, svqadd(vin1_lo, vin2_lo));
                svst1(pg_1, output_ptr + x + svcnth(), svqadd(vin1_hi, vin2_hi));

                x += svcntb();
                pg_u = svwhilelt_b8(x, window_end_x);
                pg_0 = svwhilelt_b16(x, window_end_x);
                pg_1 = svwhilelt_b16(x, static_cast<int>(window_end_x + svcnth()));
            }
            while(svptest_any(all_true_pg, pg_u));
        }
        else
        {
            int      x    = window_start_x;
            svbool_t pg_u = svwhilelt_b8(x, window_end_x);
            svbool_t pg_0 = svwhilelt_b16(x, window_end_x);
            svbool_t pg_1 = svwhilelt_b16(x, static_cast<int>(window_end_x + svcnth()));
            do
            {
                const auto vin1 = svld1(pg_u, input1_ptr + x);
                const auto vin2 = svld1(pg_u, input2_ptr + x);

                const auto vin1_lo = svreinterpret_s16_u16(svunpklo(vin1));
                const auto vin1_hi = svreinterpret_s16_u16(svunpkhi(vin1));
                const auto vin2_lo = svreinterpret_s16_u16(svunpklo(vin2));
                const auto vin2_hi = svreinterpret_s16_u16(svunpkhi(vin2));
                svst1(pg_0, output_ptr + x, svqadd(vin1_lo, vin2_lo));
                svst1(pg_1, output_ptr + x + svcnth(), svqadd(vin1_hi, vin2_hi));

                x += svcntb();
                pg_u = svwhilelt_b8(x, window_end_x);
                pg_0 = svwhilelt_b16(x, window_end_x);
                pg_1 = svwhilelt_b16(x, static_cast<int>(window_end_x + svcnth()));
            }
            while(svptest_any(all_true_pg, pg_u));
        }
    },
    input1, input2, output);
}

void arithmetic_addition_S16_U8_S16_sve(const ITensor *in1, const ITensor *in2, ITensor *out, const ConvertPolicy &policy, const Window &window)
{
    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(in1, input1_win);
    Iterator input2(in2, input2_win);
    Iterator output(out, win);

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const auto all_true_pg    = svptrue_b8();

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const int16_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

        if(policy == ConvertPolicy::WRAP)
        {
            int      x    = window_start_x;
            svbool_t pg_u = svwhilelt_b8(x, window_end_x);
            svbool_t pg_0 = svwhilelt_b16(x, window_end_x);
            svbool_t pg_1 = svwhilelt_b16(x + static_cast<int>(svcnth()), window_end_x);
            do
            {
                const auto vin1_0  = svld1_s16(pg_0, input1_ptr + x);
                const auto vin1_1  = svld1_s16(pg_1, input1_ptr + x + svcnth());
                const auto vin2_u8 = svld1_u8(pg_u, input2_ptr + x);
                const auto vin2_0  = svreinterpret_s16_u16(svunpklo(vin2_u8));
                const auto vin2_1  = svreinterpret_s16_u16(svunpkhi(vin2_u8));
                svst1_s16(pg_0, output_ptr + x, svadd_s16_z(pg_0, vin1_0, vin2_0));
                svst1_s16(pg_1, output_ptr + x, svadd_s16_z(pg_1, vin1_1, vin2_1));

                x += svcnth();
                pg_u = svwhilelt_b8(x, window_end_x);
                pg_0 = svwhilelt_b16(x, window_end_x);
                pg_1 = svwhilelt_b16(x + static_cast<int>(svcnth()), window_end_x);
            }
            while(svptest_any(all_true_pg, pg_u));
        }
        else
        {
            int      x    = window_start_x;
            svbool_t pg_u = svwhilelt_b8(x, window_end_x);
            svbool_t pg_0 = svwhilelt_b16(x, window_end_x);
            svbool_t pg_1 = svwhilelt_b16(x + static_cast<int>(svcnth()), window_end_x);
            do
            {
                const auto vin1_0  = svld1_s16(pg_0, input1_ptr + x);
                const auto vin1_1  = svld1_s16(pg_1, input1_ptr + x);
                const auto vin2_u8 = svld1_u8(pg_u, input2_ptr + x);
                const auto vin2_0  = svreinterpret_s16_u16(svunpklo(vin2_u8));
                const auto vin2_1  = svreinterpret_s16_u16(svunpkhi(vin2_u8));

                svst1_s16(pg_0, output_ptr + x, svqadd(vin1_0, vin2_0));
                svst1_s16(pg_1, output_ptr + x, svqadd(vin1_1, vin2_1));

                x += svcnth();
                pg_u = svwhilelt_b8(x, window_end_x);
                pg_0 = svwhilelt_b16(x, window_end_x);
                pg_1 = svwhilelt_b16(x + static_cast<int>(svcnth()), window_end_x);
            }
            while(svptest_any(all_true_pg, pg_u));
        }
    },
    input1, input2, output);
}

void arithmetic_addition_U8_S16_S16_sve(const ITensor *input1, const ITensor *input2, ITensor *output, const ConvertPolicy &policy, const Window &window)
{
    // Simply swap the two input buffers:
    arithmetic_addition_S16_U8_S16_sve(input2, input1, output, policy, window);
}
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_SVE) */