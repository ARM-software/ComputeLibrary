/*
 * Copyright (c) 2017-2021, 2024 Arm Limited.
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

#ifndef ACL_SRC_CPU_KERNELS_DIRECTCONV2D_OUTPUT_STAGE_GENERIC_NEON_FLOAT_IMPL_H
#define ACL_SRC_CPU_KERNELS_DIRECTCONV2D_OUTPUT_STAGE_GENERIC_NEON_FLOAT_IMPL_H

#include "arm_compute/core/Helpers.h" // Iterator
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

#include "src/core/NEON/wrapper/wrapper.h"

#include <cstdint>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
template <typename T>
void output_stage_nchw_fp(ITensor       *src,
                          const ITensor *bias,
                          const Window  &window,
                          ITensor       *dst,
                          int            result_fixedpoint_multiplier,
                          int            result_shift,
                          int            result_offset_after_shift)
{
    const bool has_bias = bias != nullptr;
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    ARM_COMPUTE_ERROR_ON(src->info()->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_UNUSED(result_fixedpoint_multiplier);
    ARM_COMPUTE_UNUSED(result_shift);
    ARM_COMPUTE_UNUSED(result_offset_after_shift);

    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();
    const int window_step_x  = 16 / src->info()->element_size();
    Window    win            = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, win);
    Iterator out(dst, win);
    execute_window_loop(
        win,
        [&](const Coordinates &id)
        {
            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                // Get bias and pointer to input
                const auto in_ptr = reinterpret_cast<const T *>(in.ptr()) + x;
                auto       v_in   = wrapper::vloadq(in_ptr);

                // Accumulate bias
                if (has_bias)
                {
                    const auto vb = wrapper::vdup_n(
                        *reinterpret_cast<const T *>(bias->ptr_to_element(Coordinates(id.z()))), ExactTagType{});
                    v_in = wrapper::vadd(v_in, vb);
                }

                const auto out_ptr = reinterpret_cast<T *>(out.ptr()) + x;
                wrapper::vstore(out_ptr, v_in);
            }

            // Left-overs loop
            for (; x < window_end_x; ++x)
            {
                // Get bias and pointer to input
                auto s_in = *(reinterpret_cast<const T *>(in.ptr()) + x);

                // Accumulate bias
                if (has_bias)
                {
                    const auto b = *reinterpret_cast<const T *>(bias->ptr_to_element(Coordinates(id.z())));
                    s_in += b;
                }

                *(reinterpret_cast<T *>(out.ptr()) + x) = s_in;
            }
        },
        in, out);
}

template <typename T>
void output_stage_nhwc_fp(ITensor       *src,
                          const ITensor *bias,
                          const Window  &window,
                          ITensor       *dst,
                          int            result_fixedpoint_multiplier,
                          int            result_shift,
                          int            result_offset_after_shift)
{
    const bool has_bias = bias != nullptr;
    ARM_COMPUTE_UNUSED(result_fixedpoint_multiplier);
    ARM_COMPUTE_UNUSED(result_shift);
    ARM_COMPUTE_UNUSED(result_offset_after_shift);

    Window window_bias = window;
    window_bias.set(Window::DimX, Window::Dimension(0, 1, 1));
    window_bias.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_bias.set(Window::DimZ, Window::Dimension(0, 0, 0));
    window_bias.set(3, Window::Dimension(0, 0, 0));

    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();
    const int window_step_x  = 16 / src->info()->element_size();
    Window    win            = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, win);
    Iterator bi(bias, window_bias);
    Iterator out(dst, win);

    execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                // Get bias and pointer to input
                const auto in_ptr = reinterpret_cast<const T *>(in.ptr());
                auto       v_in   = wrapper::vloadq(in_ptr + x);

                // Accumulate bias
                if (has_bias)
                {
                    const auto bias_ptr = reinterpret_cast<T *>(bi.ptr()) + x;
                    v_in                = wrapper::vadd(v_in, wrapper::vloadq(bias_ptr));
                }

                const auto out_ptr = reinterpret_cast<T *>(out.ptr());
                wrapper::vstore(out_ptr + x, v_in);
            }

            // Left-overs loop
            for (; x < window_end_x; ++x)
            {
                // Get bias and pointer to input
                auto s_in = *(reinterpret_cast<const T *>(in.ptr()) + x);

                // Accumulate bias
                if (has_bias)
                {
                    const auto bias_ptr = reinterpret_cast<T *>(bi.ptr()) + x;
                    s_in += *bias_ptr;
                }

                const auto out_ptr = reinterpret_cast<T *>(out.ptr());
                *(out_ptr + x)     = s_in;
            }
        },
        in, bi, out);
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_DIRECTCONV2D_OUTPUT_STAGE_GENERIC_NEON_FLOAT_IMPL_H
