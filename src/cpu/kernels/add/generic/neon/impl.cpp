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

#include "src/cpu/kernels/add/generic/neon/impl.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/wrapper/wrapper.h"
namespace arm_compute
{
namespace cpu
{
template <typename ScalarType>
void add_same_neon(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)
{
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<ScalarType, wrapper::traits::BitWidth::W128>;

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x         = 16 / sizeof(ScalarType);
    const auto    window_start_x        = static_cast<int>(window.x().start());
    const auto    window_end_x          = static_cast<int>(window.x().end());
    const bool    is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();

    if(is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? src1 : src0;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src1 : src0;

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(dst, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const ScalarType *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<ScalarType *>(output.ptr());

            const ScalarType broadcast_value     = *reinterpret_cast<const ScalarType *>(broadcast_input.ptr());
            const auto       broadcast_value_vec = wrapper::vdup_n(broadcast_value, ExactTagType{});

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto non_broadcast_v = wrapper::vloadq(non_broadcast_input_ptr + x);
                const auto res             = (policy == ConvertPolicy::SATURATE) ? wrapper::vqadd(broadcast_value_vec, non_broadcast_v) : wrapper::vadd(broadcast_value_vec, non_broadcast_v);
                wrapper::vstore(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto non_broadcast_v = *(non_broadcast_input_ptr + x);
                *(output_ptr + x)          = (policy == ConvertPolicy::SATURATE) ? wrapper::add_sat(broadcast_value, non_broadcast_v) : broadcast_value + non_broadcast_v;
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

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const ScalarType *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const ScalarType *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<ScalarType *>(output.ptr());

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto val1 = wrapper::vloadq(input1_ptr + x);
                const auto val2 = wrapper::vloadq(input2_ptr + x);
                const auto res  = (policy == ConvertPolicy::SATURATE) ? wrapper::vqadd(val1, val2) : wrapper::vadd(val1, val2);
                wrapper::vstore(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto val1   = *(input1_ptr + x);
                const auto val2   = *(input2_ptr + x);
                *(output_ptr + x) = (policy == ConvertPolicy::SATURATE) ? wrapper::add_sat(val1, val2) : val1 + val2;
            }
        },
        input1, input2, output);
    }
}

bool add_q8_neon_fixedpoint_possible(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    const auto iq0 = src0->quantization_info().uniform();
    const auto iq1 = src1->quantization_info().uniform();
    const auto oq = dst->quantization_info().uniform();

    const auto scale0 = iq0.scale / oq.scale;
    const auto scale1 = iq1.scale / oq.scale;

    if(scale0 < -31.f || scale0 > 31.f || scale1 < -31.f || scale1 > 31.f)
    {
        // The scale factor cannot be stored as 6.10 signed fixed-point number.
        return false;
    }

    const auto offset = float(oq.offset) - scale0 * float(iq0.offset) - scale1 * float(iq1.offset);
    const auto max_acc = (std::abs(scale0) + std::abs(scale1)) * 1024.f + std::abs(offset);

    if(max_acc > 2097151.f)  // 2^21 - 1
    {
        // It might not be possible to store the result as 22.10 signed fixed-point number.
        return false;
    }

    return true;
}

template <typename ScalarType>
void add_q8_neon_fixedpoint(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)
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

    constexpr int window_step_x = 16;
    const auto window_start_x = window.x().start();
    const auto window_end_x = window.x().end();
    const auto is_broadcast_across_x = in0_shape.x() != in1_shape.x();

    const auto iq0_info = in0_info->quantization_info().uniform();
    const auto iq1_info = in1_info->quantization_info().uniform();
    const auto oq_info = dst->info()->quantization_info().uniform();

    const auto in0_scale = iq0_info.scale / oq_info.scale;
    const auto in1_scale = iq1_info.scale / oq_info.scale;
    const auto offset = float(oq_info.offset) - in0_scale * float(iq0_info.offset) - in1_scale * float(iq1_info.offset);

    const auto in0_scale_6p10 = static_cast<int16_t>(support::cpp11::lround(in0_scale * 1024.f));
    const auto in1_scale_6p10 = static_cast<int16_t>(support::cpp11::lround(in1_scale * 1024.f));
    const auto offset_22p10 = static_cast<int32_t>(support::cpp11::lround(offset * 1024.f));

    if(is_broadcast_across_x)
    {
        // Prefix: a = non-broadcast, b = broadcast.

        const auto is_broadcast_input_1 = in1_win.x().step() == 0;
        auto a_win = is_broadcast_input_1 ? in0_win : in1_win;
        auto b_win = is_broadcast_input_1 ? in1_win : in0_win;
        const auto a_tensor = is_broadcast_input_1 ? src0 : src1;
        const auto b_tensor = is_broadcast_input_1 ? src1 : src0;

        const auto a_scale_6p10 = is_broadcast_input_1 ? in0_scale_6p10 : in1_scale_6p10;
        const auto b_scale = is_broadcast_input_1 ? in1_scale : in0_scale;
        const auto a_vscale_6p10 = wrapper::vdup_n(a_scale_6p10, wrapper::traits::vector_64_tag());

#ifndef __aarch64__
        const auto a_scale = is_broadcast_input_1 ? in0_scale : in1_scale;
#endif // __aarch64__

        // Clear the x dimension on the execution window as we process the whole row each iteration.
        a_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator a_input_it(a_tensor, a_win);
        Iterator b_input_it(b_tensor, b_win);
        Iterator out_it(dst, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto a_ptr = reinterpret_cast<const ScalarType *>(a_input_it.ptr());
            const auto b_ptr = reinterpret_cast<const ScalarType *>(b_input_it.ptr());
            const auto out_ptr = reinterpret_cast<ScalarType *>(out_it.ptr());

            const auto b_val = *b_ptr;
            const auto b_scaled = b_scale * b_val;
            const auto b_scaled_22p10 = static_cast<int32_t>(support::cpp11::lround(b_scaled * 1024.f));
            const auto b_scaled_offseted_22p10 = b_scaled_22p10 + offset_22p10;
            const auto b_vscaled_offseted_22p10 = wrapper::vdup_n(b_scaled_offseted_22p10, wrapper::traits::vector_128_tag());

#ifndef __aarch64__
            const auto b_scaled_offseted = b_scaled + offset;
#endif // __aarch64__

            int x = window_start_x;

            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                // Load the input.
                const auto a_vin_8p0 = wrapper::vloadq(a_ptr + x);

                // Widen the non-broadcast elements to signed 16-bit regardless of the input signedness.
                const auto a_vin_16p0_0 = wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(a_vin_8p0)));
                const auto a_vin_16p0_1 = wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(a_vin_8p0)));

                // Multiply the non-broadcast elements by the scale factor, add the scaled broadcast elements and the offset.
                // Widen and store the result in 32-bit integer.
                const auto vout_22p10_00 = wrapper::vmlal(b_vscaled_offseted_22p10, wrapper::vgetlow(a_vin_16p0_0), a_vscale_6p10);
                const auto vout_22p10_01 = wrapper::vmlal(b_vscaled_offseted_22p10, wrapper::vgethigh(a_vin_16p0_0), a_vscale_6p10);
                const auto vout_22p10_10 = wrapper::vmlal(b_vscaled_offseted_22p10, wrapper::vgetlow(a_vin_16p0_1), a_vscale_6p10);
                const auto vout_22p10_11 = wrapper::vmlal(b_vscaled_offseted_22p10, wrapper::vgethigh(a_vin_16p0_1), a_vscale_6p10);

                // Remove 2 bits of the fractional part, round, narrow to 16-bit and saturate the result.
                const auto vout_8p8_0 = wrapper::vcombine(
                    wrapper::vqrshrn_ex<2, ScalarType>(vout_22p10_00),
                    wrapper::vqrshrn_ex<2, ScalarType>(vout_22p10_01)
                );
                const auto vout_8p8_1 = wrapper::vcombine(
                    wrapper::vqrshrn_ex<2, ScalarType>(vout_22p10_10),
                    wrapper::vqrshrn_ex<2, ScalarType>(vout_22p10_11)
                );

                // Remove 8 bits of the fractional part, round, narrow to 8-bit and saturate the result.
                const auto vout_8p0 = wrapper::vcombine(
                    wrapper::vqrshrn<8>(vout_8p8_0),
                    wrapper::vqrshrn<8>(vout_8p8_1)
                );

                // Store the result.
                wrapper::vstore(out_ptr + x, vout_8p0);
            }

            // Process the left-over elements.
            for(; x < window_end_x; ++x)
            {
#ifdef __aarch64__
                out_ptr[x] = wrapper::vqrshrn<8>(wrapper::vqrshrn_ex<2, ScalarType>(int32_t(a_ptr[x]) * a_scale_6p10 + b_scaled_offseted_22p10));
#else // __aarch64__
                out_ptr[x] = utility::clamp<int, ScalarType>(support::cpp11::lround(float(a_ptr[x]) * a_scale + b_scaled_offseted));
#endif // __aarch64__
            }
        },
        b_input_it, a_input_it, out_it);
    }
    else
    {
        const auto vscale0_6p10 = wrapper::vdup_n(in0_scale_6p10, wrapper::traits::vector_64_tag());
        const auto vscale1_6p10 = wrapper::vdup_n(in1_scale_6p10, wrapper::traits::vector_64_tag());
        const auto voffset_22p10 = wrapper::vdup_n(offset_22p10, wrapper::traits::vector_128_tag());

        // Clear the x dimension on the execution window as we process the whole row each iteration.
        in0_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        in1_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator in0_it(src0, in0_win);
        Iterator in1_it(src1, in1_win);
        Iterator out_it(dst, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto in0_ptr = reinterpret_cast<const ScalarType *>(in0_it.ptr());
            const auto in1_ptr = reinterpret_cast<const ScalarType *>(in1_it.ptr());
            const auto out_ptr = reinterpret_cast<ScalarType *>(out_it.ptr());

            int x = window_start_x;

            for(; x <= (window_end_x - window_step_x); x += window_step_x)
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
                const auto vscaled0_offseted_22p10_00 = wrapper::vmlal(voffset_22p10, wrapper::vgetlow(vin0_16p0_0), vscale0_6p10);
                const auto vscaled0_offseted_22p10_01 = wrapper::vmlal(voffset_22p10, wrapper::vgethigh(vin0_16p0_0), vscale0_6p10);
                const auto vscaled0_offseted_22p10_10 = wrapper::vmlal(voffset_22p10, wrapper::vgetlow(vin0_16p0_1), vscale0_6p10);
                const auto vscaled0_offseted_22p10_11 = wrapper::vmlal(voffset_22p10, wrapper::vgethigh(vin0_16p0_1), vscale0_6p10);

                const auto vout_22p10_00 = wrapper::vmlal(vscaled0_offseted_22p10_00, wrapper::vgetlow(vin1_16p0_0), vscale1_6p10);
                const auto vout_22p10_01 = wrapper::vmlal(vscaled0_offseted_22p10_01, wrapper::vgethigh(vin1_16p0_0), vscale1_6p10);
                const auto vout_22p10_10 = wrapper::vmlal(vscaled0_offseted_22p10_10, wrapper::vgetlow(vin1_16p0_1), vscale1_6p10);
                const auto vout_22p10_11 = wrapper::vmlal(vscaled0_offseted_22p10_11, wrapper::vgethigh(vin1_16p0_1), vscale1_6p10);

                // Remove 2 bits of the fractional part, round, narrow to 16-bit and saturate the result.
                const auto vout_8p8_0 = wrapper::vcombine(
                    wrapper::vqrshrn_ex<2, ScalarType>(vout_22p10_00),
                    wrapper::vqrshrn_ex<2, ScalarType>(vout_22p10_01)
                );
                const auto vout_8p8_1 = wrapper::vcombine(
                    wrapper::vqrshrn_ex<2, ScalarType>(vout_22p10_10),
                    wrapper::vqrshrn_ex<2, ScalarType>(vout_22p10_11)
                );

                // Remove 8 bits of the fractional part, round, narrow to 8-bit and saturate the result.
                const auto vout_8p0 = wrapper::vcombine(
                    wrapper::vqrshrn<8>(vout_8p8_0),
                    wrapper::vqrshrn<8>(vout_8p8_1)
                );

                // Store the result.
                wrapper::vstore(out_ptr + x, vout_8p0);
            }

            // Process the left-over elements.
            for(; x < window_end_x; ++x)
            {
#ifdef __aarch64__
                out_ptr[x] = wrapper::vqrshrn<8>(wrapper::vqrshrn_ex<2, ScalarType>(int32_t(in0_ptr[x]) * in0_scale_6p10 + int32_t(in1_ptr[x]) * in1_scale_6p10 + offset_22p10));
#else // __aarch64__
                out_ptr[x] = utility::clamp<int, ScalarType>(support::cpp11::lround(float(in0_ptr[x]) * in0_scale + float(in1_ptr[x]) * in1_scale + offset));
#endif // __aarch64__
            }
        },
        in0_it, in1_it, out_it);
    }
}

template void add_same_neon<float>(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window);
template void add_same_neon<uint8_t>(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window);
template void add_same_neon<int32_t>(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window);
template void add_same_neon<int16_t>(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
template void add_same_neon<float16_t>(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window);
#endif /* (__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */

template void add_q8_neon_fixedpoint<int8_t>(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window);
template void add_q8_neon_fixedpoint<uint8_t>(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window);

} // namespace cpu
} // namespace arm_compute
