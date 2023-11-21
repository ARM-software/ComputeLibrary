/*
 * Copyright (c) 2020-2023 Arm Limited.
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

#include "src/cpu/kernels/add/generic/neon/impl.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/utils/misc/Traits.h"

#include "src/core/NEON/wrapper/wrapper.h"
namespace arm_compute
{
namespace cpu
{
bool sub_q8_neon_fixedpoint_possible(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    return add_sub_q8_neon_fixedpoint_possible(src0, src1, dst, false);
}

bool add_q8_neon_fixedpoint_possible(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    return add_sub_q8_neon_fixedpoint_possible(src0, src1, dst, true);
}

bool add_sub_q8_neon_fixedpoint_possible(const ITensorInfo *src0,
                                         const ITensorInfo *src1,
                                         const ITensorInfo *dst,
                                         bool               is_addition)
{
    const auto iq0 = src0->quantization_info().uniform();
    const auto iq1 = src1->quantization_info().uniform();
    const auto oq  = dst->quantization_info().uniform();

    const auto scale0 = iq0.scale / oq.scale;
    const auto scale1 = iq1.scale / oq.scale;

    if (scale0 < -15.f || scale0 > 15.f || scale1 < -15.f || scale1 > 15.f)
    {
        // The scale factor cannot be stored as 5.11 signed fixed-point number.
        return false;
    }

    const auto offset = float(oq.offset) - scale0 * float(iq0.offset) - scale1 * float(iq1.offset);

    const auto max_acc = is_addition ? ((std::abs(scale0) + std::abs(scale1)) * 256.f + std::abs(offset))
                                     : ((std::abs(scale0) - std::abs(scale1)) * 256.f + std::abs(offset));

    if (max_acc > 1048575.f) // 2^20 - 1
    {
        // It might not be possible to store the result as 21.11 signed fixed-point number.
        return false;
    }

    return true;
}

template <typename ScalarType>
void add_q8_neon_fixedpoint(
    const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)
{
    add_sub_q8_neon_fixedpoint<ScalarType>(src0, src1, dst, policy, window, true /*is_addition*/);
}

template <typename ScalarType>
void add_sub_q8_neon_fixedpoint(const ITensor       *src0,
                                const ITensor       *src1,
                                ITensor             *dst,
                                const ConvertPolicy &policy,
                                const Window        &window,
                                bool                 is_addition)
{
    ARM_COMPUTE_UNUSED(policy);

    const auto in0_info = src0->info();
    const auto in1_info = src1->info();

    const auto &in0_shape = in0_info->tensor_shape();
    const auto &in1_shape = in1_info->tensor_shape();

    // Create input windows.
    Window in0_win = window.broadcast_if_dimension_le_one(in0_shape);
    Window in1_win = window.broadcast_if_dimension_le_one(in1_shape);

    // Clear the x dimension on the execution window as we process the whole row each iteration.
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x         = 16;
    const auto    window_start_x        = window.x().start();
    const auto    window_end_x          = window.x().end();
    const auto    is_broadcast_across_x = in0_shape.x() != in1_shape.x();

    const auto iq0_info  = in0_info->quantization_info().uniform();
    const auto iq1_info  = in1_info->quantization_info().uniform();
    const auto oq_info   = dst->info()->quantization_info().uniform();
    const auto in0_scale = iq0_info.scale / oq_info.scale;
    const auto in1_scale = is_addition ? (iq1_info.scale / oq_info.scale) : (-(iq1_info.scale / oq_info.scale));
    const auto offset = float(oq_info.offset) - in0_scale * float(iq0_info.offset) - in1_scale * float(iq1_info.offset);

    constexpr float _2pow11        = 2048;
    const auto      in0_scale_5p11 = static_cast<int16_t>(support::cpp11::lround(in0_scale * _2pow11));
    const auto      in1_scale_5p11 = static_cast<int16_t>(support::cpp11::lround(in1_scale * _2pow11));
    const auto      offset_21p11   = static_cast<int32_t>(support::cpp11::lround(offset * _2pow11));

    constexpr uint8_t shift_amount_remainder = 3;

    if (is_broadcast_across_x)
    {
        // Prefix: a = non-broadcast, b = broadcast.

        const auto is_broadcast_input_1 = in1_win.x().step() == 0;
        auto       a_win                = is_broadcast_input_1 ? in0_win : in1_win;
        auto       b_win                = is_broadcast_input_1 ? in1_win : in0_win;
        const auto a_tensor             = is_broadcast_input_1 ? src0 : src1;
        const auto b_tensor             = is_broadcast_input_1 ? src1 : src0;

        const auto a_scale_5p11  = is_broadcast_input_1 ? in0_scale_5p11 : in1_scale_5p11;
        const auto b_scale       = is_broadcast_input_1 ? in1_scale : in0_scale;
        const auto a_vscale_5p11 = wrapper::vdup_n(a_scale_5p11, wrapper::traits::vector_64_tag());

#ifndef __aarch64__
        const auto a_scale = is_broadcast_input_1 ? in0_scale : in1_scale;
#endif // __aarch64__

        // Clear the x dimension on the execution window as we process the whole row each iteration.
        a_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator a_input_it(a_tensor, a_win);
        Iterator b_input_it(b_tensor, b_win);
        Iterator out_it(dst, win);

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                const auto a_ptr   = reinterpret_cast<const ScalarType *>(a_input_it.ptr());
                const auto b_ptr   = reinterpret_cast<const ScalarType *>(b_input_it.ptr());
                const auto out_ptr = reinterpret_cast<ScalarType *>(out_it.ptr());

                const auto b_val                   = *b_ptr;
                const auto b_scaled                = b_scale * b_val;
                const auto b_scaled_21p11          = static_cast<int32_t>(support::cpp11::lround(b_scaled * _2pow11));
                const auto b_scaled_offseted_21p11 = b_scaled_21p11 + offset_21p11;
                const auto b_vscaled_offseted_21p11 =
                    wrapper::vdup_n(b_scaled_offseted_21p11, wrapper::traits::vector_128_tag());

#ifndef __aarch64__
                const auto b_scaled_offseted = b_scaled + offset;
#endif // __aarch64__

                int x = window_start_x;

                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    // Load the input.
                    const auto a_vin_8p0 = wrapper::vloadq(a_ptr + x);

                    // Widen the non-broadcast elements to signed 16-bit regardless of the input signedness.
                    const auto a_vin_16p0_0 = wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(a_vin_8p0)));
                    const auto a_vin_16p0_1 = wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(a_vin_8p0)));

                    // Multiply the non-broadcast elements by the scale factor, add the scaled broadcast elements and the offset.
                    // Widen and store the result in 32-bit integer.
                    const auto vout_21p11_00 =
                        wrapper::vmlal(b_vscaled_offseted_21p11, wrapper::vgetlow(a_vin_16p0_0), a_vscale_5p11);
                    const auto vout_21p11_01 =
                        wrapper::vmlal(b_vscaled_offseted_21p11, wrapper::vgethigh(a_vin_16p0_0), a_vscale_5p11);
                    const auto vout_21p11_10 =
                        wrapper::vmlal(b_vscaled_offseted_21p11, wrapper::vgetlow(a_vin_16p0_1), a_vscale_5p11);
                    const auto vout_21p11_11 =
                        wrapper::vmlal(b_vscaled_offseted_21p11, wrapper::vgethigh(a_vin_16p0_1), a_vscale_5p11);

                    // Remove 3 bits of the fractional part, round, narrow to 16-bit and saturate the result.
                    const auto vout_8p8_0 =
                        wrapper::vcombine(wrapper::vqrshrn_ex<shift_amount_remainder, ScalarType>(vout_21p11_00),
                                          wrapper::vqrshrn_ex<shift_amount_remainder, ScalarType>(vout_21p11_01));
                    const auto vout_8p8_1 =
                        wrapper::vcombine(wrapper::vqrshrn_ex<shift_amount_remainder, ScalarType>(vout_21p11_10),
                                          wrapper::vqrshrn_ex<shift_amount_remainder, ScalarType>(vout_21p11_11));

                    // Remove 8 bits of the fractional part, round, narrow to 8-bit and saturate the result.
                    const auto vout_8p0 =
                        wrapper::vcombine(wrapper::vqrshrn<8>(vout_8p8_0), wrapper::vqrshrn<8>(vout_8p8_1));

                    // Store the result.
                    wrapper::vstore(out_ptr + x, vout_8p0);
                }

                // Process the left-over elements.
                for (; x < window_end_x; ++x)
                {
#ifdef __aarch64__
                    out_ptr[x] = wrapper::vqrshrn<8>(wrapper::vqrshrn_ex<shift_amount_remainder, ScalarType>(
                        int32_t(a_ptr[x]) * a_scale_5p11 + b_scaled_offseted_21p11));
#else  // __aarch64__
                    out_ptr[x] = utility::clamp<int, ScalarType>(
                        support::cpp11::lround(float(a_ptr[x]) * a_scale + b_scaled_offseted));
#endif // __aarch64__
                }
            },
            b_input_it, a_input_it, out_it);
    }
    else
    {
        const auto vscale0_5p11  = wrapper::vdup_n(in0_scale_5p11, wrapper::traits::vector_64_tag());
        const auto vscale1_5p11  = wrapper::vdup_n(in1_scale_5p11, wrapper::traits::vector_64_tag());
        const auto voffset_21p11 = wrapper::vdup_n(offset_21p11, wrapper::traits::vector_128_tag());

        // Clear the x dimension on the execution window as we process the whole row each iteration.
        in0_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        in1_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator in0_it(src0, in0_win);
        Iterator in1_it(src1, in1_win);
        Iterator out_it(dst, win);

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                const auto in0_ptr = reinterpret_cast<const ScalarType *>(in0_it.ptr());
                const auto in1_ptr = reinterpret_cast<const ScalarType *>(in1_it.ptr());
                const auto out_ptr = reinterpret_cast<ScalarType *>(out_it.ptr());

                int x = window_start_x;

                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    // Load the inputs.
                    const auto vin0_8p0 = wrapper::vloadq(in0_ptr + x);
                    const auto vin1_8p0 = wrapper::vloadq(in1_ptr + x);

                    // Widen the input elements to signed 16-bit regardless of the input signedness.
                    const auto vin0_16p0_0 = wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(vin0_8p0)));
                    const auto vin0_16p0_1 = wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(vin0_8p0)));
                    const auto vin1_16p0_0 = wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(vin1_8p0)));
                    const auto vin1_16p0_1 = wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(vin1_8p0)));

                    // Multiply the input elements by the scale factor and add the offset.
                    // Widen and store the result in 32-bit integer.
                    const auto vscaled0_offseted_21p11_00 =
                        wrapper::vmlal(voffset_21p11, wrapper::vgetlow(vin0_16p0_0), vscale0_5p11);
                    const auto vscaled0_offseted_21p11_01 =
                        wrapper::vmlal(voffset_21p11, wrapper::vgethigh(vin0_16p0_0), vscale0_5p11);
                    const auto vscaled0_offseted_21p11_10 =
                        wrapper::vmlal(voffset_21p11, wrapper::vgetlow(vin0_16p0_1), vscale0_5p11);
                    const auto vscaled0_offseted_21p11_11 =
                        wrapper::vmlal(voffset_21p11, wrapper::vgethigh(vin0_16p0_1), vscale0_5p11);

                    const auto vout_21p11_00 =
                        wrapper::vmlal(vscaled0_offseted_21p11_00, wrapper::vgetlow(vin1_16p0_0), vscale1_5p11);
                    const auto vout_21p11_01 =
                        wrapper::vmlal(vscaled0_offseted_21p11_01, wrapper::vgethigh(vin1_16p0_0), vscale1_5p11);
                    const auto vout_21p11_10 =
                        wrapper::vmlal(vscaled0_offseted_21p11_10, wrapper::vgetlow(vin1_16p0_1), vscale1_5p11);
                    const auto vout_21p11_11 =
                        wrapper::vmlal(vscaled0_offseted_21p11_11, wrapper::vgethigh(vin1_16p0_1), vscale1_5p11);

                    // Remove 3 bits of the fractional part, round, narrow to 16-bit and saturate the result.
                    const auto vout_8p8_0 =
                        wrapper::vcombine(wrapper::vqrshrn_ex<shift_amount_remainder, ScalarType>(vout_21p11_00),
                                          wrapper::vqrshrn_ex<shift_amount_remainder, ScalarType>(vout_21p11_01));
                    const auto vout_8p8_1 =
                        wrapper::vcombine(wrapper::vqrshrn_ex<shift_amount_remainder, ScalarType>(vout_21p11_10),
                                          wrapper::vqrshrn_ex<shift_amount_remainder, ScalarType>(vout_21p11_11));

                    // Remove 8 bits of the fractional part, round, narrow to 8-bit and saturate the result.
                    const auto vout_8p0 =
                        wrapper::vcombine(wrapper::vqrshrn<8>(vout_8p8_0), wrapper::vqrshrn<8>(vout_8p8_1));

                    // Store the result.
                    wrapper::vstore(out_ptr + x, vout_8p0);
                }

                // Process the left-over elements.
                for (; x < window_end_x; ++x)
                {
#ifdef __aarch64__
                    out_ptr[x] = wrapper::vqrshrn<8>(wrapper::vqrshrn_ex<shift_amount_remainder, ScalarType>(
                        int32_t(in0_ptr[x]) * in0_scale_5p11 + int32_t(in1_ptr[x]) * in1_scale_5p11 + offset_21p11));
#else  // __aarch64__
                    out_ptr[x] = utility::clamp<int, ScalarType>(
                        support::cpp11::lround(float(in0_ptr[x]) * in0_scale + float(in1_ptr[x]) * in1_scale + offset));
#endif // __aarch64__
                }
            },
            in0_it, in1_it, out_it);
    }
}

void add_sub_qasymm8_neon(const ITensor       *src0,
                          const ITensor       *src1,
                          ITensor             *dst,
                          const ConvertPolicy &policy,
                          const Window        &window,
                          bool                 is_addition)
{
    ARM_COMPUTE_UNUSED(policy);

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x         = 16;
    const auto    window_start_x        = static_cast<int>(window.x().start());
    const auto    window_end_x          = static_cast<int>(window.x().end());
    const bool    is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();

    const UniformQuantizationInfo iq1_info = src0->info()->quantization_info().uniform();
    const UniformQuantizationInfo iq2_info = src1->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info  = dst->info()->quantization_info().uniform();

    const auto scale1 = iq1_info.scale / oq_info.scale;
    const auto scale2 = is_addition ? (iq2_info.scale / oq_info.scale) : (-(iq2_info.scale / oq_info.scale));
    const auto offset = float(oq_info.offset) - scale1 * float(iq1_info.offset) - scale2 * float(iq2_info.offset);

    if (is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? src1 : src0;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src1 : src0;

        const auto af_scale = is_broadcast_input_2 ? scale1 : scale2;
        const auto bf_scale = is_broadcast_input_2 ? scale2 : scale1;
        const auto vscale1  = vdupq_n_f32(af_scale);

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(dst, win);

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                const auto non_broadcast_input_ptr = non_broadcast_input.ptr();
                const auto output_ptr              = output.ptr();

                const auto broadcast_value = *broadcast_input.ptr();
                const auto bf              = vdupq_n_f32(float(broadcast_value) * scale2 + offset);
                const auto bfs             = float(broadcast_value) * bf_scale + offset;

                // Compute S elements per iteration
                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const uint8x16_t a = vld1q_u8(non_broadcast_input_ptr + x);

                    const auto a_u16_0 = vmovl_u8(vget_low_u8(a));
                    const auto a_u16_1 = vmovl_u8(vget_high_u8(a));

                    const auto af_0 = vmlaq_f32(bf, vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_u16_0))), vscale1);
                    const auto af_1 = vmlaq_f32(bf, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_u16_0))), vscale1);
                    const auto af_2 = vmlaq_f32(bf, vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_u16_1))), vscale1);
                    const auto af_3 = vmlaq_f32(bf, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_u16_1))), vscale1);

                    int32x4_t rf_0{};
                    int32x4_t rf_1{};
                    int32x4_t rf_2{};
                    int32x4_t rf_3{};

#ifdef __aarch64__
                    rf_0 = vcvtnq_s32_f32(af_0);
                    rf_1 = vcvtnq_s32_f32(af_1);
                    rf_2 = vcvtnq_s32_f32(af_2);
                    rf_3 = vcvtnq_s32_f32(af_3);
#else  //__aarch64__
                    rf_0          = vcvtq_s32_f32(af_0);
                    rf_1          = vcvtq_s32_f32(af_1);
                    rf_2          = vcvtq_s32_f32(af_2);
                    rf_3          = vcvtq_s32_f32(af_3);
#endif //__aarch64__

                    const uint8x8_t pa = vqmovun_s16(vcombine_s16(vqmovn_s32(rf_0), vqmovn_s32(rf_1)));
                    const uint8x8_t pb = vqmovun_s16(vcombine_s16(vqmovn_s32(rf_2), vqmovn_s32(rf_3)));
                    vst1q_u8(output_ptr + x, vcombine_u8(pa, pb));
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    const auto result = float(non_broadcast_input_ptr[x]) * af_scale + bfs;
#ifdef __aarch64__
                    output_ptr[x] = utility::clamp<int, uint8_t>(support::cpp11::lround(result));
#else  // __aarch64__
                    output_ptr[x] = utility::clamp<int, uint8_t>(support::cpp11::trunc(result));
#endif // __aarch64__
                }
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

        const auto vscale1 = vdupq_n_f32(scale1);
        const auto vscale2 = vdupq_n_f32(scale2);
        const auto voffset = vdupq_n_f32(offset);

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                const auto input1_ptr = input1.ptr();
                const auto input2_ptr = input2.ptr();
                const auto output_ptr = output.ptr();

                // Compute S elements per iteration
                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const uint8x16_t a = vld1q_u8(input1_ptr + x);
                    const uint8x16_t b = vld1q_u8(input2_ptr + x);

                    const auto a_u16_0 = vmovl_u8(vget_low_u8(a));
                    const auto a_u16_1 = vmovl_u8(vget_high_u8(a));
                    const auto b_u16_0 = vmovl_u8(vget_low_u8(b));
                    const auto b_u16_1 = vmovl_u8(vget_high_u8(b));

                    const auto af_0 = vmlaq_f32(voffset, vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_u16_0))), vscale1);
                    const auto af_1 = vmlaq_f32(voffset, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_u16_0))), vscale1);
                    const auto af_2 = vmlaq_f32(voffset, vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_u16_1))), vscale1);
                    const auto af_3 = vmlaq_f32(voffset, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_u16_1))), vscale1);

                    const auto bf_0 = vmlaq_f32(af_0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16_0))), vscale2);
                    const auto bf_1 = vmlaq_f32(af_1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_u16_0))), vscale2);
                    const auto bf_2 = vmlaq_f32(af_2, vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16_1))), vscale2);
                    const auto bf_3 = vmlaq_f32(af_3, vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_u16_1))), vscale2);

                    int32x4_t rf_0{};
                    int32x4_t rf_1{};
                    int32x4_t rf_2{};
                    int32x4_t rf_3{};

#ifdef __aarch64__
                    rf_0 = vcvtnq_s32_f32(bf_0);
                    rf_1 = vcvtnq_s32_f32(bf_1);
                    rf_2 = vcvtnq_s32_f32(bf_2);
                    rf_3 = vcvtnq_s32_f32(bf_3);
#else  //__aarch64__
                    rf_0          = vcvtq_s32_f32(bf_0);
                    rf_1          = vcvtq_s32_f32(bf_1);
                    rf_2          = vcvtq_s32_f32(bf_2);
                    rf_3          = vcvtq_s32_f32(bf_3);
#endif //__aarch64__

                    const uint8x8_t pa = vqmovun_s16(vcombine_s16(vqmovn_s32(rf_0), vqmovn_s32(rf_1)));
                    const uint8x8_t pb = vqmovun_s16(vcombine_s16(vqmovn_s32(rf_2), vqmovn_s32(rf_3)));
                    vst1q_u8(output_ptr + x, vcombine_u8(pa, pb));
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    const auto result = float(input1_ptr[x]) * scale1 + float(input2_ptr[x]) * scale2 + offset;
#ifdef __aarch64__
                    output_ptr[x] = utility::clamp<int, uint8_t>(support::cpp11::lround(result));
#else  // __aarch64__
                    output_ptr[x] = utility::clamp<int, uint8_t>(support::cpp11::trunc(result));
#endif // __aarch64__
                }
            },
            input1, input2, output);
    }
}

void add_sub_qasymm8_signed_neon(const ITensor       *src0,
                                 const ITensor       *src1,
                                 ITensor             *dst,
                                 const ConvertPolicy &policy,
                                 const Window        &window,
                                 bool                 is_addition)
{
    ARM_COMPUTE_UNUSED(policy);

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x         = 16;
    const auto    window_start_x        = static_cast<int>(window.x().start());
    const auto    window_end_x          = static_cast<int>(window.x().end());
    const bool    is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();

    const UniformQuantizationInfo iq1_info = src0->info()->quantization_info().uniform();
    const UniformQuantizationInfo iq2_info = src1->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info  = dst->info()->quantization_info().uniform();

    const auto scale1 = iq1_info.scale / oq_info.scale;
    const auto scale2 = is_addition ? (iq2_info.scale / oq_info.scale) : (-(iq2_info.scale / oq_info.scale));
    const auto offset = float(oq_info.offset) - scale1 * float(iq1_info.offset) - scale2 * float(iq2_info.offset);

    if (is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? src1 : src0;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src1 : src0;

        const auto af_scale = is_broadcast_input_2 ? scale1 : scale2;
        const auto bf_scale = is_broadcast_input_2 ? scale2 : scale1;
        const auto vscale1  = vdupq_n_f32(af_scale);

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(dst, win);

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                const auto non_broadcast_input_ptr = reinterpret_cast<const int8_t *>(non_broadcast_input.ptr());
                const auto output_ptr              = reinterpret_cast<int8_t *>(output.ptr());

                const auto broadcast_value = *reinterpret_cast<const int8_t *>(broadcast_input.ptr());
                const auto bf              = vdupq_n_f32(float(broadcast_value) * scale2 + offset);
                const auto bfs             = float(broadcast_value) * bf_scale + offset;

                // Compute S elements per iteration
                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const int8x16_t a = vld1q_s8(non_broadcast_input_ptr + x);

                    const auto a_s16_0 = vmovl_s8(vget_low_s8(a));
                    const auto a_s16_1 = vmovl_s8(vget_high_s8(a));

                    const auto af_0 = vmlaq_f32(bf, vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_s16_0))), vscale1);
                    const auto af_1 = vmlaq_f32(bf, vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_s16_0))), vscale1);
                    const auto af_2 = vmlaq_f32(bf, vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_s16_1))), vscale1);
                    const auto af_3 = vmlaq_f32(bf, vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_s16_1))), vscale1);

                    int32x4_t rf_0{};
                    int32x4_t rf_1{};
                    int32x4_t rf_2{};
                    int32x4_t rf_3{};

#ifdef __aarch64__
                    rf_0 = vcvtnq_s32_f32(af_0);
                    rf_1 = vcvtnq_s32_f32(af_1);
                    rf_2 = vcvtnq_s32_f32(af_2);
                    rf_3 = vcvtnq_s32_f32(af_3);
#else  //__aarch64__
                    rf_0          = vcvtq_s32_f32(af_0);
                    rf_1          = vcvtq_s32_f32(af_1);
                    rf_2          = vcvtq_s32_f32(af_2);
                    rf_3          = vcvtq_s32_f32(af_3);
#endif //__aarch64__

                    const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf_0), vqmovn_s32(rf_1)));
                    const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf_2), vqmovn_s32(rf_3)));
                    vst1q_s8(output_ptr + x, vcombine_s8(pa, pb));
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    const auto result = float(non_broadcast_input_ptr[x]) * af_scale + bfs;
#ifdef __aarch64__
                    output_ptr[x] = utility::clamp<int, int8_t>(support::cpp11::lround(result));
#else  // __aarch64__
                    output_ptr[x] = utility::clamp<int, int8_t>(support::cpp11::trunc(result));
#endif // __aarch64__
                }
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

        const auto vscale1 = vdupq_n_f32(scale1);
        const auto vscale2 = vdupq_n_f32(scale2);
        const auto voffset = vdupq_n_f32(offset);

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                const auto input1_ptr = reinterpret_cast<const int8_t *>(input1.ptr());
                const auto input2_ptr = reinterpret_cast<const int8_t *>(input2.ptr());
                const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

                // Compute S elements per iteration
                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const int8x16_t a = vld1q_s8(input1_ptr + x);
                    const int8x16_t b = vld1q_s8(input2_ptr + x);

                    const auto a_s16_0 = vmovl_s8(vget_low_s8(a));
                    const auto a_s16_1 = vmovl_s8(vget_high_s8(a));
                    const auto b_s16_0 = vmovl_s8(vget_low_s8(b));
                    const auto b_s16_1 = vmovl_s8(vget_high_s8(b));

                    const auto af_0 = vmlaq_f32(voffset, vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_s16_0))), vscale1);
                    const auto af_1 = vmlaq_f32(voffset, vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_s16_0))), vscale1);
                    const auto af_2 = vmlaq_f32(voffset, vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_s16_1))), vscale1);
                    const auto af_3 = vmlaq_f32(voffset, vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_s16_1))), vscale1);

                    const auto bf_0 = vmlaq_f32(af_0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_s16_0))), vscale2);
                    const auto bf_1 = vmlaq_f32(af_1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_s16_0))), vscale2);
                    const auto bf_2 = vmlaq_f32(af_2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_s16_1))), vscale2);
                    const auto bf_3 = vmlaq_f32(af_3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_s16_1))), vscale2);

                    int32x4_t rf_0{};
                    int32x4_t rf_1{};
                    int32x4_t rf_2{};
                    int32x4_t rf_3{};

#ifdef __aarch64__
                    rf_0 = vcvtnq_s32_f32(bf_0);
                    rf_1 = vcvtnq_s32_f32(bf_1);
                    rf_2 = vcvtnq_s32_f32(bf_2);
                    rf_3 = vcvtnq_s32_f32(bf_3);
#else  //__aarch64__
                    rf_0          = vcvtq_s32_f32(bf_0);
                    rf_1          = vcvtq_s32_f32(bf_1);
                    rf_2          = vcvtq_s32_f32(bf_2);
                    rf_3          = vcvtq_s32_f32(bf_3);
#endif //__aarch64__

                    const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf_0), vqmovn_s32(rf_1)));
                    const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf_2), vqmovn_s32(rf_3)));
                    vst1q_s8(output_ptr + x, vcombine_s8(pa, pb));
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    const auto result = float(input1_ptr[x]) * scale1 + float(input2_ptr[x]) * scale2 + offset;
#ifdef __aarch64__
                    output_ptr[x] = utility::clamp<int, int8_t>(support::cpp11::lround(result));
#else  // __aarch64__
                    output_ptr[x] = utility::clamp<int, int8_t>(support::cpp11::trunc(result));
#endif // __aarch64__
                }
            },
            input1, input2, output);
    }
}

template void add_q8_neon_fixedpoint<int8_t>(
    const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window);
template void add_q8_neon_fixedpoint<uint8_t>(
    const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window);

template void add_sub_q8_neon_fixedpoint<int8_t>(const ITensor       *src0,
                                                 const ITensor       *src1,
                                                 ITensor             *dst,
                                                 const ConvertPolicy &policy,
                                                 const Window        &window,
                                                 bool                 is_addition);
template void add_sub_q8_neon_fixedpoint<uint8_t>(const ITensor       *src0,
                                                  const ITensor       *src1,
                                                  ITensor             *dst,
                                                  const ConvertPolicy &policy,
                                                  const Window        &window,
                                                  bool                 is_addition);

void add_sub_qasymm8_neon(const ITensor       *src0,
                          const ITensor       *src1,
                          ITensor             *dst,
                          const ConvertPolicy &policy,
                          const Window        &window,
                          bool                 is_addition);
void add_sub_qasymm8_signed_neon(const ITensor       *src0,
                                 const ITensor       *src1,
                                 ITensor             *dst,
                                 const ConvertPolicy &policy,
                                 const Window        &window,
                                 bool                 is_addition);

} // namespace cpu
} // namespace arm_compute
