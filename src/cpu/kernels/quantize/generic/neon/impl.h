/*
 * Copyright (c) 2024-2025 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_QUANTIZE_GENERIC_NEON_IMPL_H
#define ACL_SRC_CPU_KERNELS_QUANTIZE_GENERIC_NEON_IMPL_H

#include "arm_compute/core/Helpers.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"

namespace arm_compute
{
namespace cpu
{
constexpr auto window_step = 16;

template <typename T>
inline float32x4x4_t load_value(const T *input_ptr)
{
    using Tx16_t = typename wrapper::traits::neon_vector<T, 16>::type;
    return arm_compute::convert_to_float32x4x4<Tx16_t>(wrapper::vloadq(input_ptr));
}

template <>
inline float32x4x4_t load_value(const float *input_ptr)
{
    return {wrapper::vloadq(input_ptr), wrapper::vloadq(input_ptr + 4), wrapper::vloadq(input_ptr + 8),
            wrapper::vloadq(input_ptr + 12)};
}
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline float32x4x4_t load_value(const float16_t *input_ptr)
{
    return {vcvt_f32_f16(wrapper::vload(input_ptr)), vcvt_f32_f16(wrapper::vload(input_ptr + 4)),
            vcvt_f32_f16(wrapper::vload(input_ptr + 8)), vcvt_f32_f16(wrapper::vload(input_ptr + 12))};
}

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <typename element_type>
using vector_type = wrapper::traits::neon_vector_t<element_type, window_step>;

template <typename quantized_type>
inline vector_type<quantized_type> vquantize_qasymm8(const float32x4x4_t &qv, const UniformQuantizationInfo &qi);

template <>
inline vector_type<uint8_t> vquantize_qasymm8<uint8_t>(const float32x4x4_t &qv, const UniformQuantizationInfo &qi)
{
    return vquantize(qv, qi);
}

template <>
inline vector_type<int8_t> vquantize_qasymm8<int8_t>(const float32x4x4_t &qv, const UniformQuantizationInfo &qi)
{
    return vquantize_signed(qv, qi);
}

template <typename quantized_type>
inline vector_type<quantized_type> vquantize_qasymm8(const float32x4x4_t &qv, const UniformRequantizationInfo &qi);

template <>
inline vector_type<uint8_t> vquantize_qasymm8<uint8_t>(const float32x4x4_t &qv, const UniformRequantizationInfo &qi)
{
    return vquantize(qv, qi);
}

template <>
inline vector_type<int8_t> vquantize_qasymm8<int8_t>(const float32x4x4_t &qv, const UniformRequantizationInfo &qi)
{
    return vquantize_signed(qv, qi);
}

template <typename TOut, typename = typename std::enable_if<std::is_signed<TOut>::value, bool>::type>
inline int8x16_t recombine_8_16(int16x8_t lower, int16x8_t upper)
{
    return wrapper::vcombine(wrapper::vqmovn(lower), wrapper::vqmovn(upper));
}

template <typename TOut, typename = typename std::enable_if<std::is_unsigned<TOut>::value, bool>::type>
inline uint8x16_t recombine_8_16(int16x8_t lower, int16x8_t upper)
{
    return wrapper::vcombine(wrapper::vqmovun(lower), wrapper::vqmovun(upper));
}

template <typename TIn, typename TOut>
void run_quantize_qsymm8(const ITensor *src, ITensor *dst, const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    const UniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
    UniformQuantizationInfo       uqinfo    = dst->info()->quantization_info().uniform();
    uqinfo                                  = compute_requantization_scale_offset(uqinfo_in, uqinfo);

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(
        win_collapsed,
        [&](const Coordinates &)
        {
            auto input_ptr  = reinterpret_cast<const TIn *>(input.ptr());
            auto output_ptr = reinterpret_cast<TOut *>(output.ptr());
            int  x          = window_start_x;
            for (; x <= (window_end_x - window_step); x += window_step)
            {
                wrapper::vstore(&output_ptr[x], vquantize_qasymm8<TOut>(load_value(&input_ptr[x]), uqinfo));
            }
            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                output_ptr[x] = quantize_qsymm8(input_ptr[x], dst->info()->quantization_info());
            }
        },
        input, output);
}

inline float32x4x4_t vquantize_qsymm8_per_channel_scalevalues(const float32x4x4_t &vin, float scale)
{
    // pre-compute reciprocal of scale
    const float32x4_t inv_s = vdupq_n_f32(1.f / scale);

    return {vmulq_f32(vin.val[0], inv_s), vmulq_f32(vin.val[1], inv_s), vmulq_f32(vin.val[2], inv_s),
            vmulq_f32(vin.val[3], inv_s)};
}

inline int8x16_t vconvert_to_int8(int32x4x4_t vals)
{
    // clamp bounds
    const int32_t qmin = std::numeric_limits<int8_t>::min(); // -128
    const int32_t qmax = std::numeric_limits<int8_t>::max(); // +127

    const int32x4_t v_qmin = vdupq_n_s32(qmin);
    const int32x4_t v_qmax = vdupq_n_s32(qmax);
    vals.val[0]            = vmaxq_s32(v_qmin, vminq_s32(vals.val[0], v_qmax));
    vals.val[1]            = vmaxq_s32(v_qmin, vminq_s32(vals.val[1], v_qmax));
    vals.val[2]            = vmaxq_s32(v_qmin, vminq_s32(vals.val[2], v_qmax));
    vals.val[3]            = vmaxq_s32(v_qmin, vminq_s32(vals.val[3], v_qmax));

    // 3) narrow 32->16 for each
    const int16x4_t v_s16_0 = vqmovn_s32(vals.val[0]);
    const int16x4_t v_s16_1 = vqmovn_s32(vals.val[1]);
    const int16x4_t v_s16_2 = vqmovn_s32(vals.val[2]);
    const int16x4_t v_s16_3 = vqmovn_s32(vals.val[3]);

    // 4) combine into two int16x8 vectors
    const int16x8_t v_s16x8_0 = vcombine_s16(v_s16_0, v_s16_1);
    const int16x8_t v_s16x8_1 = vcombine_s16(v_s16_2, v_s16_3);

    // 5) saturating narrow 16->8
    const int8x8_t v_s8_0 = vqmovn_s16(v_s16x8_0);
    const int8x8_t v_s8_1 = vqmovn_s16(v_s16x8_1);

    // 6) combine into one int8x16
    return vcombine_s8(v_s8_0, v_s8_1);
}

#ifdef __aarch64__
inline int32x4x4_t vconvert_to_int32(const float32x4x4_t &vals, arm_compute::RoundingPolicy rp)
{
    if (rp == RoundingPolicy::TO_NEAREST_EVEN)
    {
        return {vcvtaq_s32_f32(vals.val[0]), vcvtaq_s32_f32(vals.val[1]), vcvtaq_s32_f32(vals.val[2]),
                vcvtaq_s32_f32(vals.val[3])};
    }
    else
    {
        return {vcvtq_s32_f32(vals.val[0]), vcvtq_s32_f32(vals.val[1]), vcvtq_s32_f32(vals.val[2]),
                vcvtq_s32_f32(vals.val[3])};
    }
}

inline int8x16_t vquantize_qsymm8_per_channel(const float32x4x4_t         vin,
                                              float                       scale, // per-channel scale
                                              arm_compute::RoundingPolicy rp)
{
    auto vscaled_vals = vquantize_qsymm8_per_channel_scalevalues(vin, scale);
    return vconvert_to_int8(vconvert_to_int32(vscaled_vals, rp));
}
#else  //__aarch64__

inline int32x4x4_t vconvert_to_int32(const float32x4x4_t &vals)
{
    return {// on AArch32 only truncating vcvtq is available
            vcvtq_s32_f32(vals.val[0]), vcvtq_s32_f32(vals.val[1]), vcvtq_s32_f32(vals.val[2]),
            vcvtq_s32_f32(vals.val[3])};
}

inline int8x16_t vquantize_qsymm8_per_channel(const float32x4x4_t vin, float scale)
{
    auto vscaled_vals = vquantize_qsymm8_per_channel_scalevalues(vin, scale);
    return vconvert_to_int8(vconvert_to_int32(vscaled_vals));
}
#endif //__aarch64__

template <typename TIn, typename TOut>
void run_quantize_qsymm8_per_channel(const ITensor *src, ITensor *dst, const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());

    const unsigned int channel_idx =
        get_data_layout_dimension_index(dst->info()->data_layout(), DataLayoutDimension::CHANNEL);

    Window     win_collapsed = window.collapse_if_possible(window, Window::DimX, Window::DimZ);
    const auto window_end_x  = static_cast<int>(win_collapsed.x().end());

    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator    input(src, win_collapsed);
    Iterator    output(dst, win_collapsed);
    const auto &qinfo = dst->info()->quantization_info();

    execute_window_loop(
        win_collapsed,
        [&](const Coordinates &coord)
        {
            auto         input_ptr  = reinterpret_cast<const TIn *>(input.ptr());
            auto         output_ptr = reinterpret_cast<TOut *>(output.ptr());
            int          x          = window_start_x;
            const size_t ch         = coord[channel_idx];
            const float  scale      = qinfo.scale()[ch];
            for (; x <= (window_end_x - window_step); x += window_step)
            {
                const auto vin = load_value(&input_ptr[x]);
#ifdef __aarch64__
                const auto vout = vquantize_qsymm8_per_channel(vin, scale, RoundingPolicy::TO_NEAREST_EVEN);
#else  //__aarch64__
                const auto vout = vquantize_qsymm8_per_channel(vin, scale /* RoundingPolicy::TO_ZERO */);
#endif //__aarch64__
                wrapper::vstore(&output_ptr[x], vout);
            }
            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
#ifdef __aarch64__
                output_ptr[x] = quantize_qsymm8_per_channel(input_ptr[x], dst->info()->quantization_info(), ch,
                                                            RoundingPolicy::TO_NEAREST_EVEN);
#else  //__aarch64__
                output_ptr[x]   = quantize_qsymm8_per_channel(input_ptr[x], dst->info()->quantization_info(), ch,
                                                              RoundingPolicy::TO_ZERO);
#endif //__aarch64__
            }
        },
        input, output);
}

template <typename TIn, typename TOut>
void run_requantize_offset_only_convert(const ITensor *src, ITensor *dst, const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    // Calculate output offset difference.
    const UniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
    UniformQuantizationInfo       uqinfo    = dst->info()->quantization_info().uniform();
    uqinfo                                  = compute_requantization_scale_offset(uqinfo_in, uqinfo);

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);

    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Duplicate offset in signed vector format
    const int8x16_t offset = wrapper::vdup_n(static_cast<int8_t>(uqinfo.offset), wrapper::traits::vector_128_tag{});

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(
        win_collapsed,
        [&](const Coordinates &)
        {
            auto input_ptr  = reinterpret_cast<const TIn *>(input.ptr());
            auto output_ptr = reinterpret_cast<TOut *>(output.ptr());
            int  x          = window_start_x;
            for (; x <= (window_end_x - window_step); x += window_step)
            {
                const wrapper::traits::neon_vector_t<TIn, window_step> qv =
                    wrapper::vloadq(input_ptr + x); // load 128 bit vector of 8 bit datatype

                // Signed addition.
                auto res = vaddq_s8(reinterpret_cast<int8x16_t>(qv), offset);

                // Output is dependent on datatype.
                wrapper::vstore(&output_ptr[x],
                                reinterpret_cast<wrapper::traits::neon_vector_t<TOut, window_step>>(res));
            }
            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                auto result   = uqinfo.offset + static_cast<int32_t>(input_ptr[x]);
                output_ptr[x] = static_cast<TOut>(result);
            }
        },
        input, output);
}

template <typename TIn, typename TOut>
void run_requantize_offset_only(const ITensor *src, ITensor *dst, const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    const UniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
    UniformQuantizationInfo       uqinfo    = dst->info()->quantization_info().uniform();
    uqinfo                                  = compute_requantization_scale_offset(uqinfo_in, uqinfo);

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Duplicate offset in signed vector format
    const int16x8_t offset = wrapper::vdup_n(static_cast<int16_t>(uqinfo.offset), wrapper::traits::vector_128_tag{});

    const int32_t low_bound   = (dst->info()->data_type() == DataType::QASYMM8) ? 0 : -128;
    const int32_t upper_bound = (dst->info()->data_type() == DataType::QASYMM8) ? 255 : 127;

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(
        win_collapsed,
        [&](const Coordinates &)
        {
            auto  input_ptr  = reinterpret_cast<const TIn *>(input.ptr());
            TOut *output_ptr = reinterpret_cast<TOut *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step); x += window_step)
            {
                const auto qv    = wrapper::vloadq(input_ptr + x); // load 128 bit vector of 8 bit datatype
                int16x8_t  lower = reinterpret_cast<int16x8_t>(wrapper::vmovl(wrapper::vgetlow(qv)));
                int16x8_t  upper = reinterpret_cast<int16x8_t>(wrapper::vmovl(wrapper::vgethigh(qv)));

                // Signed addition.
                lower = wrapper::vqadd(lower, offset);
                upper = wrapper::vqadd(upper, offset);

                // Output is dependent on datatype.
                auto res = recombine_8_16<TOut>(lower, upper);
                wrapper::vstore(&output_ptr[x], res);
            }
            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                // Add offset and clamp result to within the range of the output datatype.
                int32_t result = uqinfo.offset + static_cast<int32_t>(input_ptr[x]);
                result         = utility::clamp<int32_t>(result, low_bound, upper_bound);

                // Cast result to output datatype.
                output_ptr[x] = static_cast<TOut>(result);
            }
        },
        input, output);
}

template <typename TIn, typename TOut>
void run_quantize_qasymm8(const ITensor *src, ITensor *dst, const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    constexpr bool is_8bit_int = std::is_same<TIn, int8_t>::value || std::is_same<TIn, uint8_t>::value;

    const UniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
    UniformQuantizationInfo       uqinfo    = dst->info()->quantization_info().uniform();
    UniformRequantizationInfo     reqinfo(1.f, 0);

    if (is_8bit_int)
    {
        reqinfo = compute_requantization_scale_float_offset(uqinfo_in, uqinfo);
    }

#ifdef __aarch64__
    constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
    constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_ZERO;
#endif //__aarch64__

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(
        win_collapsed,
        [&](const Coordinates &)
        {
            auto input_ptr  = reinterpret_cast<const TIn *>(input.ptr());
            auto output_ptr = reinterpret_cast<TOut *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step); x += window_step)
            {
                if (is_8bit_int)
                {
                    wrapper::vstore(&output_ptr[x], vquantize_qasymm8<TOut>(load_value(&input_ptr[x]), reqinfo));
                }
                else
                {
                    wrapper::vstore(&output_ptr[x], vquantize_qasymm8<TOut>(load_value(&input_ptr[x]), uqinfo));
                }
            }
            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                if (is_8bit_int)
                {
                    output_ptr[x] = Qasymm8QuantizationHelper<TOut>::quantize(input_ptr[x], reqinfo, rounding_policy);
                }
                else
                {
                    output_ptr[x] = Qasymm8QuantizationHelper<TOut>::quantize(input_ptr[x], uqinfo, rounding_policy);
                }
            }
        },
        input, output);
}

template <typename T>
void run_quantize_qasymm16(const ITensor *src, ITensor *dst, const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    const UniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
    UniformQuantizationInfo       uqinfo    = dst->info()->quantization_info().uniform();
    if (is_data_type_quantized_asymmetric(src->info()->data_type()))
    {
        uqinfo = compute_requantization_scale_offset(uqinfo_in, uqinfo);
    }
#ifdef __aarch64__
    constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
    constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_ZERO;
#endif //__aarch64__

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(
        win_collapsed,
        [&](const Coordinates &)
        {
            auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
            auto output_ptr = reinterpret_cast<uint16_t *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step); x += window_step)
            {
                uint16x8x2_t tmp = vquantize_qasymm16(load_value(&input_ptr[x]), uqinfo);
                vst1q_u16(&output_ptr[x], tmp.val[0]);
                vst1q_u16(&output_ptr[x + 8], tmp.val[1]);
            }
            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                output_ptr[x] = quantize_qasymm16(input_ptr[x], uqinfo, rounding_policy);
            }
        },
        input, output);
}
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_QUANTIZE_GENERIC_NEON_IMPL_H
