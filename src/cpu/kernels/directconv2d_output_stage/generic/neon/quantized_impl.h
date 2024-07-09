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

#ifndef ACL_SRC_CPU_KERNELS_DIRECTCONV2D_OUTPUT_STAGE_GENERIC_NEON_QUANTIZED_IMPL_H
#define ACL_SRC_CPU_KERNELS_DIRECTCONV2D_OUTPUT_STAGE_GENERIC_NEON_QUANTIZED_IMPL_H

#include "arm_compute/core/Helpers.h" // Iterator
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include <cstdint>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

template <typename TOut>
void output_stage_nchw_quant(ITensor       *src,
                             const ITensor *bias,
                             const Window  &window,
                             ITensor       *dst,
                             int            result_fixedpoint_multiplier,
                             int            result_shift,
                             int            result_offset_after_shift)
{
    const bool has_bias = bias != nullptr;
    using VectorType    = typename wrapper::traits::neon_bitvector_t<TOut, wrapper::traits::BitWidth::W128>;
    using TagType       = typename wrapper::traits::neon_bitvector_tag_t<TOut, wrapper::traits::BitWidth::W128>;

    const int32x4_t result_offset_after_shift_s32 = vdupq_n_s32(result_offset_after_shift);

    const VectorType min = wrapper::vdup_n(std::numeric_limits<TOut>::lowest(), TagType{});
    const VectorType max = wrapper::vdup_n(std::numeric_limits<TOut>::max(), TagType{});

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
                const auto  in_ptr = reinterpret_cast<int32_t *>(in.ptr()) + x;
                int32x4x4_t v_in = {{wrapper::vloadq(in_ptr), wrapper::vloadq(in_ptr + 4), wrapper::vloadq(in_ptr + 8),
                                     wrapper::vloadq(in_ptr + 12)}};

                // Accumulate bias
                if (has_bias)
                {
                    const auto vb = wrapper::vdup_n(
                        *reinterpret_cast<const int32_t *>(bias->ptr_to_element(Coordinates(id.z()))), TagType{});
                    v_in = {{wrapper::vadd(v_in.val[0], vb), wrapper::vadd(v_in.val[1], vb),
                             wrapper::vadd(v_in.val[2], vb), wrapper::vadd(v_in.val[3], vb)}};
                }

                const auto out_ptr = reinterpret_cast<TOut *>(out.ptr()) + x;
                wrapper::vstore(out_ptr, finalize_quantization(v_in, result_fixedpoint_multiplier, result_shift,
                                                               result_offset_after_shift_s32, min, max, false));
            }

            // Left-overs loop
            for (; x < window_end_x; ++x)
            {
                // Get bias and pointer to input
                int32_t s_in = *(reinterpret_cast<const int32_t *>(in.ptr()) + x);

                // Accumulate bias
                if (has_bias)
                {
                    const auto b = *reinterpret_cast<const int32_t *>(bias->ptr_to_element(Coordinates(id.z())));
                    s_in += b;
                }

                const auto out_ptr = reinterpret_cast<TOut *>(out.ptr()) + x;
                *out_ptr =
                    finalize_quantization(s_in, result_fixedpoint_multiplier, result_shift, result_offset_after_shift,
                                          std::numeric_limits<TOut>::lowest(), std::numeric_limits<TOut>::max(), false);
            }
        },
        in, out);
}
template <
    typename TOut,
    typename std::enable_if<std::is_same<TOut, uint8_t>::value || std::is_same<TOut, int8_t>::value, int>::type = 0>
void output_stage_nhwc_quant(ITensor       *src,
                             const ITensor *bias,
                             const Window  &window,
                             ITensor       *dst,
                             int            result_fixedpoint_multiplier,
                             int            result_shift,
                             int            result_offset_after_shift)
{
    const bool has_bias = bias != nullptr;
    using VectorType    = typename wrapper::traits::neon_bitvector_t<TOut, wrapper::traits::BitWidth::W128>;
    using TagType       = typename wrapper::traits::neon_bitvector_tag_t<TOut, wrapper::traits::BitWidth::W128>;

    const int32x4_t result_offset_after_shift_s32 = vdupq_n_s32(result_offset_after_shift);

    const VectorType min = wrapper::vdup_n(std::numeric_limits<TOut>::lowest(), TagType{});
    const VectorType max = wrapper::vdup_n(std::numeric_limits<TOut>::max(), TagType{});

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
                const auto  in_ptr = reinterpret_cast<int32_t *>(in.ptr()) + x;
                int32x4x4_t v_in   = {{
                      wrapper::vloadq(in_ptr),
                      wrapper::vloadq(in_ptr + 4),
                      wrapper::vloadq(in_ptr + 8),
                      wrapper::vloadq(in_ptr + 12),
                }};

                // Accumulate bias
                if (has_bias)
                {
                    const auto bias_ptr = reinterpret_cast<int32_t *>(bi.ptr()) + x;

                    wrapper::vadd(v_in.val[0], wrapper::vloadq(bias_ptr));
                    wrapper::vadd(v_in.val[1], wrapper::vloadq(bias_ptr + 4));
                    wrapper::vadd(v_in.val[2], wrapper::vloadq(bias_ptr + 8));
                    wrapper::vadd(v_in.val[3], wrapper::vloadq(bias_ptr + 12));
                }

                const auto out_ptr = reinterpret_cast<TOut *>(out.ptr()) + x;
                wrapper::vstore(out_ptr, finalize_quantization(v_in, result_fixedpoint_multiplier, result_shift,
                                                               result_offset_after_shift_s32, min, max, false));
            }

            // Left-overs loop
            for (; x < window_end_x; ++x)
            {
                // Get bias and pointer to input
                const auto in_ptr = reinterpret_cast<int32_t *>(in.ptr()) + x;
                int32_t    s_in   = *in_ptr;

                // Accumulate bias
                if (has_bias)
                {
                    const auto bias_ptr = reinterpret_cast<int32_t *>(bi.ptr()) + x;
                    s_in += *bias_ptr;
                }

                const auto out_ptr = reinterpret_cast<TOut *>(out.ptr()) + x;
                *out_ptr =
                    finalize_quantization(s_in, result_fixedpoint_multiplier, result_shift, result_offset_after_shift,
                                          std::numeric_limits<TOut>::lowest(), std::numeric_limits<TOut>::max(), false);
            }
        },
        in, bi, out);
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_DIRECTCONV2D_OUTPUT_STAGE_GENERIC_NEON_QUANTIZED_IMPL_H
