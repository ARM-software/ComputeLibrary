/*
 * Copyright (c) 2016-2020 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEArithmeticSubtractionKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NESymm.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

namespace arm_compute
{
namespace
{
template <typename T>
inline typename std::enable_if<std::is_same<T, int8_t>::value, int8_t>::type
quantize(float val, const QuantizationInfo &info)
{
    return quantize_qasymm8_signed(val, info);
}

template <typename T>
inline typename std::enable_if<std::is_same<T, uint8_t>::value, uint8_t>::type
quantize(float val, const QuantizationInfo &info)
{
    return quantize_qasymm8(val, info);
}

template <typename T>
void sub_same(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window, bool is_sat)
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x         = 16 / sizeof(T);
    const auto    window_start_x        = static_cast<int>(window.x().start());
    const auto    window_end_x          = static_cast<int>(window.x().end());
    const bool    is_broadcast_across_x = (input1_win.x().step() == 0) || (input2_win.x().step() == 0);

    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    if(is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? in2 : in1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const T *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<T *>(output.ptr());

            const T    broadcast_value     = *reinterpret_cast<const T *>(broadcast_input.ptr());
            const auto broadcast_value_vec = wrapper::vdup_n(broadcast_value, ExactTagType{});

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto non_broadcast_v = wrapper::vloadq(non_broadcast_input_ptr + x);
                auto       res             = is_sat ? wrapper::vqsub(broadcast_value_vec, non_broadcast_v) : wrapper::vsub(broadcast_value_vec, non_broadcast_v);
                if(is_broadcast_input_2)
                {
                    res = wrapper::vmul(res, wrapper::vdup_n(static_cast<T>(-1), ExactTagType{}));
                }
                wrapper::vstore(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto non_broadcast_v = *(non_broadcast_input_ptr + x);
                auto       res             = is_sat ? wrapper::sub_sat(broadcast_value, non_broadcast_v) : broadcast_value - non_broadcast_v;
                if(is_broadcast_input_2)
                {
                    res = static_cast<T>(-1) * res;
                }

                *(output_ptr + x) = res;
            }
        },
        broadcast_input, non_broadcast_input, output);
    }
    else
    {
        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(in1, input1_win);
        Iterator input2(in2, input2_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const T *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const T *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<T *>(output.ptr());

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto val1 = wrapper::vloadq(input1_ptr + x);
                const auto val2 = wrapper::vloadq(input2_ptr + x);
                const auto res  = is_sat ? wrapper::vqsub(val1, val2) : wrapper::vsub(val1, val2);
                wrapper::vstore(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto val1   = *(input1_ptr + x);
                const auto val2   = *(input2_ptr + x);
                *(output_ptr + x) = is_sat ? wrapper::sub_sat(val1, val2) : val1 - val2;
            }
        },
        input1, input2, output);
    }
}

template <typename T>
void sub_quantized(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window, bool is_sat)
{
    ARM_COMPUTE_UNUSED(is_sat);

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x         = 16;
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = (input1_win.x().step() == 0) || (input2_win.x().step() == 0);

    const UniformQuantizationInfo iq1_info = in1->info()->quantization_info().uniform();
    const UniformQuantizationInfo iq2_info = in2->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info  = out->info()->quantization_info().uniform();

    const float32x4_t invvscaleo = vdupq_n_f32(1.f / oq_info.scale);
    const float32x4_t voffseto   = vdupq_n_f32(oq_info.offset);

    if(is_broadcast_across_x)
    {
        const bool                    is_broadcast_input_2 = input2_win.x().step() == 0;
        Window                        broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window                        non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor                *broadcast_tensor     = is_broadcast_input_2 ? in2 : in1;
        const ITensor                *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;
        const UniformQuantizationInfo broadcast_qinfo      = broadcast_tensor->info()->quantization_info().uniform();
        const UniformQuantizationInfo non_broadcast_qinfo  = non_broadcast_tensor->info()->quantization_info().uniform();
        const float32x4_t             vscale1              = is_broadcast_input_2 ? vdupq_n_f32(iq1_info.scale) : vdupq_n_f32(iq2_info.scale);
        const float32x4_t             vscale2              = is_broadcast_input_2 ? vdupq_n_f32(iq2_info.scale) : vdupq_n_f32(iq1_info.scale);
        const int32x4_t               voffset1             = is_broadcast_input_2 ? vdupq_n_s32(iq1_info.offset) : vdupq_n_s32(iq2_info.offset);
        const int32x4_t               voffset2             = is_broadcast_input_2 ? vdupq_n_s32(iq2_info.offset) : vdupq_n_s32(iq1_info.offset);

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const T *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<T *>(output.ptr());

            const auto broadcast_value     = *reinterpret_cast<const T *>(broadcast_input.ptr());
            const auto broadcast_value_vec = wrapper::vdup_n(static_cast<T>(broadcast_value), wrapper::traits::vector_128_tag{});

            const float32x4x4_t bf =
            {
                {
                    vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(wrapper::vmovl(wrapper::vgetlow(broadcast_value_vec))))), voffset2)), vscale2),
                    vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(wrapper::vmovl(wrapper::vgetlow(broadcast_value_vec))))), voffset2)), vscale2),
                    vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(wrapper::vmovl(wrapper::vgethigh(broadcast_value_vec))))), voffset2)), vscale2),
                    vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(wrapper::vmovl(wrapper::vgethigh(broadcast_value_vec))))), voffset2)), vscale2),
                }
            };

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto a = wrapper::vloadq(non_broadcast_input_ptr + x);

                const float32x4x4_t af =
                {
                    {
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(wrapper::vmovl(wrapper::vgetlow(a))))), voffset1)), vscale1),
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(wrapper::vmovl(wrapper::vgetlow(a))))), voffset1)), vscale1),
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(wrapper::vmovl(wrapper::vgethigh(a))))), voffset1)), vscale1),
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(wrapper::vmovl(wrapper::vgethigh(a))))), voffset1)), vscale1),
                    }
                };

                const int32x4x4_t rf =
                {
                    {
#ifdef __aarch64_
                        vcvtnq_s32_f32(vmlaq_f32(voffseto, !is_broadcast_input_2 ? vsubq_f32(bf.val[0], af.val[0]) : vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtnq_s32_f32(vmlaq_f32(voffseto, !is_broadcast_input_2 ? vsubq_f32(bf.val[1], af.val[1]) : vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
                        vcvtnq_s32_f32(vmlaq_f32(voffseto, !is_broadcast_input_2 ? vsubq_f32(bf.val[2], af.val[2]) : vsubq_f32(af.val[2], bf.val[2]), invvscaleo)),
                        vcvtnq_s32_f32(vmlaq_f32(voffseto, !is_broadcast_input_2 ? vsubq_f32(bf.val[3], af.val[3]) : vsubq_f32(af.val[3], bf.val[3]), invvscaleo)),
#else  //__aarch64__
                        vcvtq_s32_f32(vmlaq_f32(voffseto, !is_broadcast_input_2 ? vsubq_f32(bf.val[0], af.val[0]) : vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtq_s32_f32(vmlaq_f32(voffseto, !is_broadcast_input_2 ? vsubq_f32(bf.val[1], af.val[1]) : vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
                        vcvtq_s32_f32(vmlaq_f32(voffseto, !is_broadcast_input_2 ? vsubq_f32(bf.val[2], af.val[2]) : vsubq_f32(af.val[2], bf.val[2]), invvscaleo)),
                        vcvtq_s32_f32(vmlaq_f32(voffseto, !is_broadcast_input_2 ? vsubq_f32(bf.val[3], af.val[3]) : vsubq_f32(af.val[3], bf.val[3]), invvscaleo)),
#endif //__aarch64__
                    }
                };

                const auto pa = wrapper::vqmov<T>(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
                const auto pb = wrapper::vqmov<T>(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
                wrapper::vstore(output_ptr + x, wrapper::vcombine(pa, pb));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const float afs   = static_cast<int32_t>(*(non_broadcast_input_ptr + x) - non_broadcast_qinfo.offset) * non_broadcast_qinfo.scale;
                const float bfs   = static_cast<int32_t>(broadcast_value - broadcast_qinfo.offset) * broadcast_qinfo.scale;
                *(output_ptr + x) = quantize<T>(is_broadcast_input_2 ? afs - bfs : bfs - afs, out->info()->quantization_info());
            }
        },
        broadcast_input, non_broadcast_input, output);
    }
    else
    {
        const float32x4_t vscale1  = vdupq_n_f32(iq1_info.scale);
        const float32x4_t vscale2  = vdupq_n_f32(iq2_info.scale);
        const int32x4_t   voffset1 = vdupq_n_s32(iq1_info.offset);
        const int32x4_t   voffset2 = vdupq_n_s32(iq2_info.offset);

        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(in1, input1_win);
        Iterator input2(in2, input2_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const T *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const T *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<T *>(output.ptr());

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto a = wrapper::vloadq(input1_ptr + x);
                const auto b = wrapper::vloadq(input2_ptr + x);

                const float32x4x4_t af =
                {
                    {
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(wrapper::vmovl(wrapper::vgetlow(a))))), voffset1)), vscale1),
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(wrapper::vmovl(wrapper::vgetlow(a))))), voffset1)), vscale1),
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(wrapper::vmovl(wrapper::vgethigh(a))))), voffset1)), vscale1),
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(wrapper::vmovl(wrapper::vgethigh(a))))), voffset1)), vscale1),
                    }
                };

                const float32x4x4_t bf =
                {
                    {
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(wrapper::vmovl(wrapper::vgetlow(b))))), voffset2)), vscale2),
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(wrapper::vmovl(wrapper::vgetlow(b))))), voffset2)), vscale2),
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(wrapper::vmovl(wrapper::vgethigh(b))))), voffset2)), vscale2),
                        vmulq_f32(vcvtq_f32_s32(vsubq_s32(wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(wrapper::vmovl(wrapper::vgethigh(b))))), voffset2)), vscale2),
                    }
                };

                const int32x4x4_t rf =
                {
                    {
#ifdef __aarch64__
                        vcvtnq_s32_f32(vmlaq_f32(voffseto, vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtnq_s32_f32(vmlaq_f32(voffseto, vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
                        vcvtnq_s32_f32(vmlaq_f32(voffseto, vsubq_f32(af.val[2], bf.val[2]), invvscaleo)),
                        vcvtnq_s32_f32(vmlaq_f32(voffseto, vsubq_f32(af.val[3], bf.val[3]), invvscaleo)),
#else  //__aarch64__
                        vcvtq_s32_f32(vmlaq_f32(voffseto, vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtq_s32_f32(vmlaq_f32(voffseto, vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
                        vcvtq_s32_f32(vmlaq_f32(voffseto, vsubq_f32(af.val[2], bf.val[2]), invvscaleo)),
                        vcvtq_s32_f32(vmlaq_f32(voffseto, vsubq_f32(af.val[3], bf.val[3]), invvscaleo)),
#endif //__aarch64__
                    }
                };

                const auto pa = wrapper::vqmov<T>(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
                const auto pb = wrapper::vqmov<T>(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
                wrapper::vstore(output_ptr + x, wrapper::vcombine(pa, pb));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const float afs = static_cast<int32_t>((*(input1_ptr + x)) - iq1_info.offset) * iq1_info.scale;
                const float bfs = static_cast<int32_t>((*(input2_ptr + x)) - iq2_info.offset) * iq2_info.scale;

                *(output_ptr + x) = quantize<T>((afs - bfs), out->info()->quantization_info());
            }
        },
        input1, input2, output);
    }
}

void sub_QSYMM16_QSYMM16_QSYMM16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window, bool is_sat)
{
    ARM_COMPUTE_UNUSED(is_sat);

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x         = 8;
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = (input1_win.x().step() == 0) || (input2_win.x().step() == 0);

    const UniformQuantizationInfo iq1_info = in1->info()->quantization_info().uniform();
    const UniformQuantizationInfo iq2_info = in2->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info  = out->info()->quantization_info().uniform();

    const float32x4_t vscale1    = vdupq_n_f32(iq1_info.scale);
    const float32x4_t vscale2    = vdupq_n_f32(iq2_info.scale);
    const float32x4_t invvscaleo = vdupq_n_f32(1.f / oq_info.scale);

    if(is_broadcast_across_x)
    {
        const bool                    is_broadcast_input_2 = input2_win.x().step() == 0;
        Window                        broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window                        non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor                *broadcast_tensor     = is_broadcast_input_2 ? in2 : in1;
        const ITensor                *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;
        const UniformQuantizationInfo broadcast_qinfo      = broadcast_tensor->info()->quantization_info().uniform();
        const UniformQuantizationInfo non_broadcast_qinfo  = non_broadcast_tensor->info()->quantization_info().uniform();

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const int16_t *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<int16_t *>(output.ptr());

            const int16_t   broadcast_value     = *reinterpret_cast<const int16_t *>(broadcast_input.ptr());
            const int16x8_t broadcast_value_vec = vdupq_n_s16(broadcast_value);

            const float32x4x2_t bf =
            {
                {
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(broadcast_value_vec))), vscale2),
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(broadcast_value_vec))), vscale2),
                }
            };
            const float bfs = static_cast<int32_t>(broadcast_value) * broadcast_qinfo.scale;

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const int16x8_t     a = vld1q_s16(non_broadcast_input_ptr + x);
                const float32x4x2_t af =
                {
                    {
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))), vscale1),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a))), vscale1),
                    }
                };

                const int32x4x4_t rf =
                {
                    {
#ifdef __aarch64__
                        vcvtnq_s32_f32(vmulq_f32(is_broadcast_input_2 ? vsubq_f32(bf.val[0], af.val[0]) : vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtnq_s32_f32(vmulq_f32(is_broadcast_input_2 ? vsubq_f32(bf.val[1], af.val[1]) : vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
#else  //__aarch64__
                        vcvtq_s32_f32(vmulq_f32(is_broadcast_input_2 ? vsubq_f32(bf.val[0], af.val[0]) : vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtq_s32_f32(vmulq_f32(is_broadcast_input_2 ? vsubq_f32(bf.val[1], af.val[1]) : vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
#endif //__aarch64__
                    }
                };

                const int16x8_t pa = vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1]));
                vst1q_s16(output_ptr + x, pa);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const float afs   = static_cast<int32_t>(*(non_broadcast_input_ptr + x)) * non_broadcast_qinfo.scale;
                *(output_ptr + x) = quantize_qsymm16(is_broadcast_input_2 ? (bfs - afs) : (afs - bfs), oq_info);
            }
        },
        broadcast_input, non_broadcast_input, output);
    }
    else
    {
        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(in1, input1_win);
        Iterator input2(in2, input2_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const int16_t *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const int16_t *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const int16x8_t a = vld1q_s16(input1_ptr + x);
                const int16x8_t b = vld1q_s16(input2_ptr + x);

                const float32x4x2_t af =
                {
                    {
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))), vscale1),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a))), vscale1),
                    }
                };

                const float32x4x2_t bf =
                {
                    {
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(b))), vscale2),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(b))), vscale2),
                    }
                };

                const int32x4x2_t rf =
                {
                    {
#ifdef __aarch64__
                        vcvtnq_s32_f32(vmulq_f32(vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtnq_s32_f32(vmulq_f32(vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
#else  //__aarch64__
                        vcvtq_s32_f32(vmulq_f32(vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtq_s32_f32(vmulq_f32(vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
#endif //__aarch64__
                    }
                };

                const int16x8_t pa = vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1]));
                vst1q_s16(output_ptr + x, pa);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const float afs   = static_cast<int32_t>((*(input1_ptr + x))) * iq1_info.scale;
                const float bfs   = static_cast<int32_t>((*(input2_ptr + x))) * iq2_info.scale;
                *(output_ptr + x) = quantize_qsymm16((afs - bfs), out->info()->quantization_info());
            }
        },
        input1, input2, output);
    }
}

void sub_S16_U8_S16_impl(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window, bool is_sat, bool is_swapped)
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

    const int  window_step_x  = 8;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const int16_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

        if(!is_sat)
        {
            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto vin1 = wrapper::vloadq(input1_ptr + x);
                const auto vin2 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input2_ptr + x)));
                const auto res  = is_swapped ? wrapper::vsub(vin2, vin1) : wrapper::vsub(vin1, vin2);
                wrapper::vstore(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto res    = is_swapped ? static_cast<int16_t>(*(input2_ptr + x)) - *(input1_ptr + x) : *(input1_ptr + x) - static_cast<int16_t>(*(input2_ptr + x));
                *(output_ptr + x) = res;
            }
        }
        else
        {
            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto vin1 = wrapper::vloadq(input1_ptr + x);
                const auto vin2 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input2_ptr + x)));
                const auto res  = is_swapped ? wrapper::vqsub(vin2, vin1) : wrapper::vqsub(vin1, vin2);
                wrapper::vstore(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto res    = is_swapped ? wrapper::sub_sat(static_cast<int16_t>(*(input2_ptr + x)), *(input1_ptr + x)) : wrapper::sub_sat(*(input1_ptr + x), static_cast<int16_t>(*(input2_ptr + x)));
                *(output_ptr + x) = res;
            }
        }
    },
    input1, input2, output);
}

void sub_S16_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window, bool is_sat)
{
    sub_S16_U8_S16_impl(in1, in2, out, window, is_sat, false);
}

void sub_U8_S16_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window, bool is_sat)
{
    // Swap arguments
    sub_S16_U8_S16_impl(in2, in1, out, window, is_sat, true);
}

void sub_U8_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window, bool is_sat)
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

    const int  window_step_x  = 8;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const uint8_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

        if(!is_sat)
        {
            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto vin1 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input1_ptr + x)));
                const auto vin2 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input2_ptr + x)));
                wrapper::vstore(output_ptr + x, wrapper::vsub(vin1, vin2));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(output_ptr + x) = static_cast<int16_t>(*(input1_ptr + x)) - static_cast<int16_t>(*(input2_ptr + x));
            }
        }
        else
        {
            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto vin1 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input1_ptr + x)));
                const auto vin2 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input2_ptr + x)));
                wrapper::vstore(output_ptr + x, wrapper::vqsub(vin1, vin2));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(output_ptr + x) = wrapper::sub_sat(static_cast<int16_t>(*(input1_ptr + x)),
                                                     static_cast<int16_t>(*(input2_ptr + x)));
            }
        }
    },
    input1, input2, output);
}

inline Status validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM16, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input2, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM16, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM16, DataType::S16, DataType::F16, DataType::F32);

    const TensorShape out_shape = TensorShape::broadcast_shape(input1.tensor_shape(), input2.tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::U8)
        && !(input1.data_type() == DataType::QASYMM8 && input2.data_type() == DataType::QASYMM8)
        && !(input1.data_type() == DataType::QASYMM8_SIGNED && input2.data_type() == DataType::QASYMM8_SIGNED)
        && !(input1.data_type() == DataType::QSYMM16 && input2.data_type() == DataType::QSYMM16)
        && !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::U8)
        && !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::S16)
        && !(input1.data_type() == DataType::S16 && input2.data_type() == DataType::U8)
        && !(input1.data_type() == DataType::S16 && input2.data_type() == DataType::S16)
        && !(input1.data_type() == DataType::F32 && input2.data_type() == DataType::F32)
        && !(input1.data_type() == DataType::F16 && input2.data_type() == DataType::F16),
        "You called subtract with the wrong image formats");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        input1.data_type() == DataType::QASYMM8_SIGNED && input2.data_type() == DataType::QASYMM8_SIGNED && policy == ConvertPolicy::WRAP
        && input1.data_type() == DataType::QASYMM8 && input2.data_type() == DataType::QASYMM8 && policy == ConvertPolicy::WRAP
        && input1.data_type() == DataType::QSYMM16 && input2.data_type() == DataType::QSYMM16 && policy == ConvertPolicy::WRAP,
        "Convert policy cannot be WRAP if datatype is QASYMM8 or QASYMM8_SIGNED");

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::U8 && output.data_type() == DataType::U8)
            && !(input1.data_type() == DataType::QASYMM8 && input2.data_type() == DataType::QASYMM8 && output.data_type() == DataType::QASYMM8)
            && !(input1.data_type() == DataType::QASYMM8_SIGNED && input2.data_type() == DataType::QASYMM8_SIGNED && output.data_type() == DataType::QASYMM8_SIGNED)
            && !(input1.data_type() == DataType::QSYMM16 && input2.data_type() == DataType::QSYMM16 && output.data_type() == DataType::QSYMM16)
            && !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::U8 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::S16 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::S16 && input2.data_type() == DataType::U8 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::S16 && input2.data_type() == DataType::S16 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::F32 && input2.data_type() == DataType::F32 && output.data_type() == DataType::F32)
            && !(input1.data_type() == DataType::F16 && input2.data_type() == DataType::F16 && output.data_type() == DataType::F16),
            "You called subtract with the wrong image formats");

        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
                                        "Wrong shape for output");
    }
    return Status{};
}
} // namespace

NEArithmeticSubtractionKernel::NEArithmeticSubtractionKernel()
    : _func(nullptr), _policy(ConvertPolicy::WRAP)
{
}

void NEArithmeticSubtractionKernel::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1, *input2, *output, policy));

    _policy = policy;

    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*input1, *input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    // Auto initialize output if not initialized
    set_shape_if_empty(*output, out_shape);

    switch(input1->data_type())
    {
        case DataType::U8:
            if(input2->data_type() == DataType::U8 && output->data_type() == DataType::U8)
            {
                _func = &sub_same<uint8_t>;
            }
            else if(input2->data_type() == DataType::U8 && output->data_type() == DataType::S16)
            {
                _func = &sub_U8_U8_S16;
            }
            else
            {
                _func = &sub_U8_S16_S16;
            }
            break;
        case DataType::QASYMM8:
            _func = &sub_quantized<uint8_t>;
            set_data_type_if_unknown(*output, DataType::QASYMM8);
            break;
        case DataType::QASYMM8_SIGNED:
            _func = &sub_quantized<int8_t>;
            set_data_type_if_unknown(*output, DataType::QASYMM8_SIGNED);
            break;
        case DataType::S16:
            if(input2->data_type() == DataType::U8)
            {
                _func = &sub_S16_U8_S16;
            }
            else
            {
                _func = &sub_same<int16_t>;
            }
            set_format_if_unknown(*output, Format::S16);
            break;
        case DataType::QSYMM16:
            _func = &sub_QSYMM16_QSYMM16_QSYMM16;
            set_data_type_if_unknown(*output, DataType::QSYMM16);
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = &sub_same<float16_t>;
            set_format_if_unknown(*output, Format::F16);
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
            _func = &sub_same<float>;
            set_format_if_unknown(*output, Format::F32);
            break;
        default:
            _func = nullptr;
    }

    // NEArithmeticSubtractionKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(valid_region);
    Window win = calculate_max_window(valid_region, Steps());

    INEKernel::configure(win);
}

Status NEArithmeticSubtractionKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output, policy));

    return Status{};
}

void NEArithmeticSubtractionKernel::run_op(const InputTensorMap &inputs, const OutputTensorMap &outputs, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    // Dispatch kernel
    (*_func)(inputs.at(TensorType::ACL_SRC_0), inputs.at(TensorType::ACL_SRC_1), outputs.at(TensorType::ACL_DST), window, (_policy == ConvertPolicy::SATURATE));
}
} // namespace arm_compute
