/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#include "src/cpu/kernels/CpuMulKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NESymm.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace
{
#if defined(ENABLE_FP32_KERNELS)
    static constexpr size_t default_mws_N1_fp32_neon = 22447;
    static constexpr size_t default_mws_V1_fp32_neon = 38982;
#endif /* ENABLE_FP32_KERNELS */
    static constexpr size_t default_mws_other_platforms_1d_tensor = 10240;
}
namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
const float       scale255_constant      = 1.f / 255.f;
const float32x4_t scale255_constant_f32q = vdupq_n_f32(scale255_constant);
const float32x4_t positive_round_f32q    = vdupq_n_f32(0.5f);

inline Status validate_arguments(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy)
{
    ARM_COMPUTE_UNUSED(overflow_policy);
    ARM_COMPUTE_UNUSED(rounding_policy);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src1, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S16, DataType::S32, DataType::QSYMM16, DataType::F16,
                                                         DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src2, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S16, DataType::S32, DataType::QSYMM16, DataType::F16,
                                                         DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::S16, DataType::QSYMM16,
                                                         DataType::S32, DataType::F16, DataType::F32);
    if(is_data_type_quantized(src1->data_type()) || is_data_type_quantized(src2->data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src1, src2);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(overflow_policy == ConvertPolicy::WRAP, "ConvertPolicy cannot be WRAP if datatype is quantized");
    }

    if(dst->total_size() > 0)
    {
        const TensorShape &out_shape = TensorShape::broadcast_shape(src1->tensor_shape(), src2->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst->tensor_shape(), 0), "Wrong shape for dst");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
        // clang-format off
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            !(src1->data_type() == src2->data_type() && src2->data_type() == dst->data_type()) &&
            !(src1->data_type() == DataType::U8 && src2->data_type() == DataType::U8 && dst->data_type() == DataType::S16) &&
            !(src1->data_type() == DataType::U8 && src2->data_type() == DataType::S16 && dst->data_type() == DataType::S16) &&
            !(src1->data_type() == DataType::S16 && src2->data_type() == DataType::U8 && dst->data_type() == DataType::S16) &&
            !(src1->data_type() == DataType::S16 && src2->data_type() == DataType::U8 && dst->data_type() == DataType::S16) &&
            !(src1->data_type() == DataType::QSYMM16 && src2->data_type() == DataType::QSYMM16 && dst->data_type() == DataType::S32)
            , "Invalid data type combination");
        // clang-format on
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(src1->data_type() == DataType::S16 && dst->data_type() == DataType::S32 && scale != 1.f, "Unsupported scale for QSYMM16 inputs and S32 dst");
    }

    if(std::abs(scale - scale255_constant) < 0.00001f)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(rounding_policy != RoundingPolicy::TO_NEAREST_UP && rounding_policy != RoundingPolicy::TO_NEAREST_EVEN);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(src1->data_type() == DataType::S32 && src2->data_type() == DataType::S32 && dst->data_type() == DataType::S32,
                                        "Scale == 1/255 is not supported if input and dst are of data type S32");
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON(rounding_policy != RoundingPolicy::TO_ZERO);

        int         exponent            = 0;
        const float normalized_mantissa = std::frexp(scale, &exponent);

        // Use int scaling if factor is equal to 1/2^n for 0 <= n <= 15
        // frexp returns 0.5 as mantissa which means that the exponent will be in the range of -1 <= e <= 14
        // Moreover, it will be negative as we deal with 1/2^n
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(!((normalized_mantissa == 0.5f) && (-14 <= exponent) && (exponent <= 1)), "Scale value not supported (Should be 1/(2^n) or 1/255");
    }

    return Status{};
}

/* Scales a given vector by 1/255.
 *
 * @note This does not work for all cases. e.g. for float of 0.49999999999999994 and large floats.
 *
 * @param in Input vector to scale.
 * @return   Scaled dst rounded to nearest (round half up).
 */
inline int32x4_t scale255_S32_S32(int32x4_t in)
{
    // Scale
    const float32x4_t tmp = vmulq_f32(vcvtq_f32_s32(in), scale255_constant_f32q);
    // Round to nearest (round half up)
    // Add +0.5 for all values
    // Afterwards vcvt rounds toward zero
    return vcvtq_s32_f32(vaddq_f32(tmp, positive_round_f32q));
}

inline uint16x8_t scale255_U16_U16(uint16x8_t in)
{
    const int32x4_t tmp_s1 = scale255_S32_S32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(in))));
    const int32x4_t tmp_s2 = scale255_S32_S32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(in))));
    return vreinterpretq_u16_s16(vcombine_s16(vmovn_s32(tmp_s2), vmovn_s32(tmp_s1)));
}

template <typename T>
inline typename std::enable_if<std::is_same<T, int8_t>::value, int8x16_t>::type
vquantize(float32x4x4_t val, const UniformQuantizationInfo &info)
{
    return vquantize_signed(val, info);
}

template <typename T>
inline typename std::enable_if<std::is_same<T, uint8_t>::value, uint8x16_t>::type
vquantize(float32x4x4_t val, const UniformQuantizationInfo &info)
{
    return vquantize(val, info);
}

template <typename T>
void mul_saturate_quantized_8(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, float scale)
{
    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x         = 16 / sizeof(T);
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = src1->info()->tensor_shape().x() != src2->info()->tensor_shape().x();

    const UniformQuantizationInfo output_qua_info = out->info()->quantization_info().uniform();
    const UniformQuantizationInfo tmp_qua_info    = { output_qua_info.scale / scale, output_qua_info.offset };

    if(is_broadcast_across_x)
    {
        const bool                    is_broadcast_input_2 = input2_win.x().step() == 0;
        Window                        broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window                        non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor                *broadcast_tensor     = is_broadcast_input_2 ? src2 : src1;
        const ITensor                *non_broadcast_tensor = !is_broadcast_input_2 ? src2 : src1;
        const UniformQuantizationInfo broadcast_qinfo      = broadcast_tensor->info()->quantization_info().uniform();
        const UniformQuantizationInfo non_broadcast_qinfo  = non_broadcast_tensor->info()->quantization_info().uniform();

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator dst(out, win);

        using ExactTagType = typename wrapper::traits::neon_vector<T, window_step_x>::tag_type;

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const T *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<T *>(dst.ptr());

            const auto broadcast_value     = *reinterpret_cast<const T *>(broadcast_input.ptr());
            const auto broadcast_value_vec = wrapper::vdup_n(broadcast_value, ExactTagType{});

            // Compute window_step_x elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto non_broadcast_v = wrapper::vloadq(non_broadcast_input_ptr + x);

                // Dequantize inputs
                const float32x4x4_t in1_f32x4x4 = vdequantize(non_broadcast_v, non_broadcast_qinfo);
                const float32x4x4_t in2_f32x4x4 = vdequantize(broadcast_value_vec, broadcast_qinfo);

                const float32x4x4_t out_f32x4x4 =
                {
                    vmulq_f32(in1_f32x4x4.val[0], in2_f32x4x4.val[0]),
                    vmulq_f32(in1_f32x4x4.val[1], in2_f32x4x4.val[1]),
                    vmulq_f32(in1_f32x4x4.val[2], in2_f32x4x4.val[2]),
                    vmulq_f32(in1_f32x4x4.val[3], in2_f32x4x4.val[3]),
                };

                // Quantize dst
                const auto result = vquantize<T>(out_f32x4x4, tmp_qua_info);
                wrapper::vstore(output_ptr + x, result);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                // Dequantize inputs
                const T     src1    = *(non_broadcast_input_ptr + x);
                const float tmp_in1 = Qasymm8QuantizationHelper<T>::dequantize(src1, non_broadcast_qinfo);
                const float tmp_in2 = Qasymm8QuantizationHelper<T>::dequantize(broadcast_value, broadcast_qinfo);
                const float tmp_f   = tmp_in1 * tmp_in2;

                // Quantize dst
                const auto tmp_qua = Qasymm8QuantizationHelper<T>::quantize(tmp_f, tmp_qua_info);
                *(output_ptr + x)  = tmp_qua;
            }
        },
        broadcast_input, non_broadcast_input, dst);
    }
    else
    {
        const UniformQuantizationInfo input1_qua_info = src1->info()->quantization_info().uniform();
        const UniformQuantizationInfo input2_qua_info = src2->info()->quantization_info().uniform();

        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(src1, input1_win);
        Iterator input2(src2, input2_win);
        Iterator dst(out, win);

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const T *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const T *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<T *>(dst.ptr());

            // Compute window_step_x elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto input1_q = wrapper::vloadq(input1_ptr + x);
                const auto input2_q = wrapper::vloadq(input2_ptr + x);

                // Dequantize inputs
                const float32x4x4_t in1_f32x4x4 = vdequantize(input1_q, input1_qua_info);
                const float32x4x4_t in2_f32x4x4 = vdequantize(input2_q, input2_qua_info);

                const float32x4x4_t out_f32x4x4 =
                {
                    vmulq_f32(in1_f32x4x4.val[0], in2_f32x4x4.val[0]),
                    vmulq_f32(in1_f32x4x4.val[1], in2_f32x4x4.val[1]),
                    vmulq_f32(in1_f32x4x4.val[2], in2_f32x4x4.val[2]),
                    vmulq_f32(in1_f32x4x4.val[3], in2_f32x4x4.val[3]),
                };

                // Quantize dst
                const auto result = vquantize<T>(out_f32x4x4, tmp_qua_info);
                wrapper::vstore(output_ptr + x, result);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                // Dequantize inputs
                const T     src1    = *(input1_ptr + x);
                const T     src2    = *(input2_ptr + x);
                const float tmp_in1 = Qasymm8QuantizationHelper<T>::dequantize(src1, input1_qua_info);
                const float tmp_in2 = Qasymm8QuantizationHelper<T>::dequantize(src2, input2_qua_info);
                const float tmp_f   = tmp_in1 * tmp_in2;

                // Quantize dst
                const auto tmp_qua = Qasymm8QuantizationHelper<T>::quantize(tmp_f, tmp_qua_info);
                *(output_ptr + x)  = tmp_qua;
            }
        },
        input1, input2, dst);
    }
}

bool mul_q8_neon_fixedpoint_possible(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, float scale)
{
    const auto iq0 = src0->quantization_info().uniform();
    const auto iq1 = src1->quantization_info().uniform();
    const auto oq  = dst->quantization_info().uniform();

    const auto multiplier = ((iq0.scale * iq1.scale) / oq.scale) * scale;

    if(multiplier < -8191.f || multiplier > 8191.f)
    {
        //The multiplier cannot be stored as a 14.18 signed fixed-point number
        return false;
    }

    const auto offset_out = float(oq.offset);

    const auto max_result = multiplier * (256) * (256) + offset_out;

    if(max_result > 8191.f)
    {
        //It might not be possible to store the result as a 14.18 signed fixed-point number.
        return false;
    }

    return true;
}

template <typename ScalarType>
void mul_q8_neon_fixedpoint(const ITensor *src0, const ITensor *src1, ITensor *dst, const Window &window, float scale)
{
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

    const auto iq0_info = in0_info->quantization_info().uniform();
    const auto iq1_info = in1_info->quantization_info().uniform();
    const auto oq_info  = dst->info()->quantization_info().uniform();

    const auto in0_offset = iq0_info.offset;
    const auto in1_offset = iq1_info.offset;
    const auto out_offset = oq_info.offset;
    const auto multiplier = ((iq0_info.scale * iq1_info.scale) / oq_info.scale) * scale;

    constexpr int32_t two_pwr18i = 262144;
    constexpr float   two_pwr18f = 262144.f;

    const auto in0_offset_16p0  = static_cast<int16_t>(in0_offset);
    const auto in1_offset_16p0  = static_cast<int16_t>(in1_offset);
    const auto out_offset_14p18 = static_cast<int32_t>(out_offset * two_pwr18i);
    const auto multiplier_14p18 = static_cast<int32_t>(multiplier * two_pwr18f);

    if(is_broadcast_across_x)
    {
        // Prefix: a = non-broadcast, b = broadcast.

        const auto is_broadcast_input_1 = in1_win.x().step() == 0;
        auto       a_win                = is_broadcast_input_1 ? in0_win : in1_win;
        auto       b_win                = is_broadcast_input_1 ? in1_win : in0_win;
        const auto a_tensor             = is_broadcast_input_1 ? src0 : src1;
        const auto b_tensor             = is_broadcast_input_1 ? src1 : src0;

        const auto a_offset_16p0 = is_broadcast_input_1 ? in0_offset_16p0 : in1_offset_16p0;
        const auto b_offset_16p0 = is_broadcast_input_1 ? in1_offset : in0_offset;
#ifndef __aarch64__
        const auto a_offset = is_broadcast_input_1 ? in0_offset : in1_offset;
        const auto b_offset = is_broadcast_input_1 ? in1_offset : in0_offset;
#endif //__aarch64__
        const auto a_voffset_16p0 = wrapper::vdup_n(a_offset_16p0, wrapper::traits::vector_64_tag());

        // Clear the x dimension on the execution window as we process the whole row each iteration.
        a_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator a_input_it(a_tensor, a_win);
        Iterator b_input_it(b_tensor, b_win);
        Iterator out_it(dst, win);

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto a_ptr   = reinterpret_cast<const ScalarType *>(a_input_it.ptr());
            const auto b_ptr   = reinterpret_cast<const ScalarType *>(b_input_it.ptr());
            const auto out_ptr = reinterpret_cast<ScalarType *>(out_it.ptr());

            const auto b_val            = *b_ptr;
            const auto b_offseted_32p0  = static_cast<int32_t>(b_val - b_offset_16p0);
            const auto b_voffseted_32p0 = wrapper::vdup_n(b_offseted_32p0, wrapper::traits::vector_128_tag());

            const auto vmultiplier_14p18 = wrapper::vdup_n(multiplier_14p18, wrapper::traits::vector_128_tag());
            const auto voffsetout_14p18  = wrapper::vdup_n(out_offset_14p18, wrapper::traits::vector_128_tag());

            int x = window_start_x;

            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                // Load the inputs.
                const auto a_vin_8p0 = wrapper::vloadq(a_ptr + x);

                // Widen the non-broadcast elements to signed 16-bit regardless of the input signedness.
                const auto a_vin_16p0_0 = wrapper::vreinterpret(wrapper::vmovl(wrapper::vgetlow(a_vin_8p0)));
                const auto a_vin_16p0_1 = wrapper::vreinterpret(wrapper::vmovl(wrapper::vgethigh(a_vin_8p0)));

                const auto voffseted_32p0_00 = wrapper::vsubl(wrapper::vgetlow(a_vin_16p0_0), a_voffset_16p0);
                const auto voffseted_32p0_01 = wrapper::vsubl(wrapper::vgethigh(a_vin_16p0_0), a_voffset_16p0);
                const auto voffseted_32p0_10 = wrapper::vsubl(wrapper::vgetlow(a_vin_16p0_1), a_voffset_16p0);
                const auto voffseted_32p0_11 = wrapper::vsubl(wrapper::vgethigh(a_vin_16p0_1), a_voffset_16p0);

                const auto vinnermul_32p0_00 = wrapper::vmul(voffseted_32p0_00, b_voffseted_32p0);
                const auto vinnermul_32p0_01 = wrapper::vmul(voffseted_32p0_01, b_voffseted_32p0);
                const auto vinnermul_32p0_10 = wrapper::vmul(voffseted_32p0_10, b_voffseted_32p0);
                const auto vinnermul_32p0_11 = wrapper::vmul(voffseted_32p0_11, b_voffseted_32p0);

                const auto vout_14p18_00 = wrapper::vmla(voffsetout_14p18, vinnermul_32p0_00, vmultiplier_14p18);
                const auto vout_14p18_01 = wrapper::vmla(voffsetout_14p18, vinnermul_32p0_01, vmultiplier_14p18);
                const auto vout_14p18_10 = wrapper::vmla(voffsetout_14p18, vinnermul_32p0_10, vmultiplier_14p18);
                const auto vout_14p18_11 = wrapper::vmla(voffsetout_14p18, vinnermul_32p0_11, vmultiplier_14p18);

                // These shift rights are to revert the multiplication by twopwr18. Hard limit of a maximum shift by 8 requires multiple shift instructions to achieve this.
                const auto vout_15p1_00 = wrapper::vqrshrn_ex<8, ScalarType>(wrapper::vshrq_n<8>(vout_14p18_00));
                const auto vout_15p1_01 = wrapper::vqrshrn_ex<8, ScalarType>(wrapper::vshrq_n<8>(vout_14p18_01));
                const auto vout_15p1_10 = wrapper::vqrshrn_ex<8, ScalarType>(wrapper::vshrq_n<8>(vout_14p18_10));
                const auto vout_15p1_11 = wrapper::vqrshrn_ex<8, ScalarType>(wrapper::vshrq_n<8>(vout_14p18_11));

                const auto vout_15p1_0 = wrapper::vcombine(
                                             vout_15p1_00,
                                             vout_15p1_01);

                const auto vout_15p1_1 = wrapper::vcombine(
                                             vout_15p1_10,
                                             vout_15p1_11);
                const auto out_ptr = reinterpret_cast<ScalarType *>(out_it.ptr());

                const auto vout_8p0 = wrapper::vcombine(
                                          wrapper::vqrshrn<2>(vout_15p1_0),
                                          wrapper::vqrshrn<2>(vout_15p1_1));
                wrapper::vstore(out_ptr + x, vout_8p0);
            }

            //Process the left-over elements.
            for(; x < window_end_x; ++x)
            {
#ifdef __aarch64__
                out_ptr[x] = wrapper::vqrshrn<2>(wrapper::vqrshrn_ex<8, ScalarType>(wrapper::vshrq_n<8>((multiplier_14p18 * (int32_t(a_ptr[x]) - a_offset_16p0) * (int32_t(
                                                                                                             b_val) - b_offset_16p0)) + out_offset_14p18)));
#else  //__aarch64__
                out_ptr[x] = utility::clamp<int32_t, ScalarType>(support::cpp11::lround(multiplier * ((float(a_ptr[x]) - a_offset) * (float(b_val) - b_offset)) + float(out_offset)));
#endif //__aarch64__
            }
        },
        a_input_it, b_input_it, out_it);
    }
    else
    {
        const auto voffset0_16p0     = wrapper::vdup_n(in0_offset_16p0, wrapper::traits::vector_64_tag());
        const auto voffset1_16p0     = wrapper::vdup_n(in1_offset_16p0, wrapper::traits::vector_64_tag());
        const auto voffsetout_14p18  = wrapper::vdup_n(out_offset_14p18, wrapper::traits::vector_128_tag());
        const auto vmultiplier_14p18 = wrapper::vdup_n(multiplier_14p18, wrapper::traits::vector_128_tag());

        // Clear the x dimension on the execution window as we process the whole row each iteration.
        in0_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        in1_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator in0_it(src0, in0_win);
        Iterator in1_it(src1, in1_win);
        Iterator out_it(dst, win);

        execute_window_loop(
            win, [&](const Coordinates &)
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

                const auto voffseted0_32p0_00 = wrapper::vsubl(wrapper::vgetlow(vin0_16p0_0), voffset0_16p0);
                const auto voffseted0_32p0_01 = wrapper::vsubl(wrapper::vgethigh(vin0_16p0_0), voffset0_16p0);
                const auto voffseted0_32p0_10 = wrapper::vsubl(wrapper::vgetlow(vin0_16p0_1), voffset0_16p0);
                const auto voffseted0_32p0_11 = wrapper::vsubl(wrapper::vgethigh(vin0_16p0_1), voffset0_16p0);

                const auto voffseted1_32p0_00 = wrapper::vsubl(wrapper::vgetlow(vin1_16p0_0), voffset1_16p0);
                const auto voffseted1_32p0_01 = wrapper::vsubl(wrapper::vgethigh(vin1_16p0_0), voffset1_16p0);
                const auto voffseted1_32p0_10 = wrapper::vsubl(wrapper::vgetlow(vin1_16p0_1), voffset1_16p0);
                const auto voffseted1_32p0_11 = wrapper::vsubl(wrapper::vgethigh(vin1_16p0_1), voffset1_16p0);

                const auto vinnermul_32p0_00 = wrapper::vmul(voffseted0_32p0_00, voffseted1_32p0_00);
                const auto vinnermul_32p0_01 = wrapper::vmul(voffseted0_32p0_01, voffseted1_32p0_01);
                const auto vinnermul_32p0_10 = wrapper::vmul(voffseted0_32p0_10, voffseted1_32p0_10);
                const auto vinnermul_32p0_11 = wrapper::vmul(voffseted0_32p0_11, voffseted1_32p0_11);

                const auto vout_14p18_00 = wrapper::vmla(voffsetout_14p18, vinnermul_32p0_00, vmultiplier_14p18);
                const auto vout_14p18_01 = wrapper::vmla(voffsetout_14p18, vinnermul_32p0_01, vmultiplier_14p18);
                const auto vout_14p18_10 = wrapper::vmla(voffsetout_14p18, vinnermul_32p0_10, vmultiplier_14p18);
                const auto vout_14p18_11 = wrapper::vmla(voffsetout_14p18, vinnermul_32p0_11, vmultiplier_14p18);

                // These shift rights are to revert the multiplication by twopwr18. Hard limit of a maximum shift by 8 requires multiple shift instructions to achieve this.
                const auto vout_14p2_00 = wrapper::vqrshrn_ex<8, ScalarType>(wrapper::vshrq_n<8>(vout_14p18_00));
                const auto vout_14p2_01 = wrapper::vqrshrn_ex<8, ScalarType>(wrapper::vshrq_n<8>(vout_14p18_01));
                const auto vout_14p2_10 = wrapper::vqrshrn_ex<8, ScalarType>(wrapper::vshrq_n<8>(vout_14p18_10));
                const auto vout_14p2_11 = wrapper::vqrshrn_ex<8, ScalarType>(wrapper::vshrq_n<8>(vout_14p18_11));

                const auto vout_14p2_0 = wrapper::vcombine(
                                             vout_14p2_00,
                                             vout_14p2_01);

                const auto vout_14p2_1 = wrapper::vcombine(
                                             vout_14p2_10,
                                             vout_14p2_11);

                const auto vout_8p0 = wrapper::vcombine(
                                          wrapper::vqrshrn<2>(vout_14p2_0),
                                          wrapper::vqrshrn<2>(vout_14p2_1));
                wrapper::vstore(out_ptr + x, vout_8p0);
            }

            //Process the left-over elements.
            for(; x < window_end_x; ++x)
            {
#ifdef __aarch64__
                out_ptr[x] = wrapper::vqrshrn<2>(wrapper::vqrshrn_ex<8, ScalarType>(wrapper::vshrq_n<8>((multiplier_14p18 * (int32_t(in0_ptr[x]) - in0_offset_16p0) * (int32_t(
                                                                                                             in1_ptr[x]) - in1_offset_16p0)) + out_offset_14p18)));
#else  //__aarch64__
                out_ptr[x] = utility::clamp<int32_t, ScalarType>(support::cpp11::lround(multiplier * ((float(in0_ptr[x]) - in0_offset) * (float(in1_ptr[x]) - in1_offset)) + float(out_offset)));
#endif //__aarch64__
            }
        },
        in0_it, in1_it, out_it);
    }
}

void mul_saturate_QSYMM16_QSYMM16_QSYMM16(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, float scale)
{
    const UniformQuantizationInfo input1_qua_info = src1->info()->quantization_info().uniform();
    const UniformQuantizationInfo input2_qua_info = src2->info()->quantization_info().uniform();
    const UniformQuantizationInfo output_qua_info = out->info()->quantization_info().uniform();

    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(src1, input1_win);
    Iterator input2(src2, input2_win);
    Iterator dst(out, win);

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    const UniformQuantizationInfo tmp_qua_info = { output_qua_info.scale / scale, output_qua_info.offset };

    execute_window_loop(
        win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const qsymm16_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const qsymm16_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<qsymm16_t *>(dst.ptr());

        // Compute window_step_x elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const qsymm16x8x2_t input1_q =
            {
                {
                    vld1q_s16(input1_ptr + x),
                    vld1q_s16(input1_ptr + x + 8),
                }
            };
            const qsymm16x8x2_t input2_q =
            {
                {
                    vld1q_s16(input2_ptr + x),
                    vld1q_s16(input2_ptr + x + 8),
                }
            };

            // Dequantize inputs
            const float32x4x4_t in1_f32x4x4 = vdequantize(input1_q, input1_qua_info);
            const float32x4x4_t in2_f32x4x4 = vdequantize(input2_q, input2_qua_info);

            const float32x4x4_t out_f32x4x4 =
            {
                vmulq_f32(in1_f32x4x4.val[0], in2_f32x4x4.val[0]),
                vmulq_f32(in1_f32x4x4.val[1], in2_f32x4x4.val[1]),
                vmulq_f32(in1_f32x4x4.val[2], in2_f32x4x4.val[2]),
                vmulq_f32(in1_f32x4x4.val[3], in2_f32x4x4.val[3]),
            };

            const qsymm16x8x2_t result = vquantize_qsymm16(out_f32x4x4, tmp_qua_info);
            vst1q_s16(output_ptr + x, result.val[0]);
            vst1q_s16(output_ptr + x + 8, result.val[1]);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            // Dequantize inputs
            float tmp_in1 = static_cast<float>(*(input1_ptr + x)) * input1_qua_info.scale;
            float tmp_in2 = static_cast<float>(*(input2_ptr + x)) * input2_qua_info.scale;
            float tmp_f   = tmp_in1 * tmp_in2;

            // Quantize dst, lrintf() has same rounding mode as vcombine_s16
            int32_t   tmp     = lrintf(tmp_f / tmp_qua_info.scale);
            qsymm16_t tmp_qua = static_cast<qsymm16_t>(tmp > SHRT_MAX) ? SHRT_MAX : ((tmp < SHRT_MIN) ? SHRT_MIN : tmp);
            *(output_ptr + x) = tmp_qua;
        }
    },
    input1, input2, dst);
}

void mul_QSYMM16_QSYMM16_S32(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, int scale)
{
    ARM_COMPUTE_UNUSED(scale);

    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(src1, input1_win);
    Iterator input2(src2, input2_win);
    Iterator dst(out, win);

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    execute_window_loop(
        win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const qsymm16_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const qsymm16_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int32_t *>(dst.ptr());

        // Compute window_step_x elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const qsymm16x8x2_t input1_q =
            {
                {
                    vld1q_s16(input1_ptr + x),
                    vld1q_s16(input1_ptr + x + 8),
                }
            };
            const qsymm16x8x2_t input2_q =
            {
                {
                    vld1q_s16(input2_ptr + x),
                    vld1q_s16(input2_ptr + x + 8),
                }
            };

            const int32x4x4_t in1_s32 =
            {
                {
                    vmovl_s16(vget_low_s16(input1_q.val[0])),
                    vmovl_s16(vget_high_s16(input1_q.val[0])),
                    vmovl_s16(vget_low_s16(input1_q.val[1])),
                    vmovl_s16(vget_high_s16(input1_q.val[1])),
                }
            };
            const int32x4x4_t in2_s32 =
            {
                {
                    vmovl_s16(vget_low_s16(input2_q.val[0])),
                    vmovl_s16(vget_high_s16(input2_q.val[0])),
                    vmovl_s16(vget_low_s16(input2_q.val[1])),
                    vmovl_s16(vget_high_s16(input2_q.val[1])),
                }
            };

            const int32x4x4_t result =
            {
                {
                    vmulq_s32(in1_s32.val[0], in2_s32.val[0]),
                    vmulq_s32(in1_s32.val[1], in2_s32.val[1]),
                    vmulq_s32(in1_s32.val[2], in2_s32.val[2]),
                    vmulq_s32(in1_s32.val[3], in2_s32.val[3]),
                }
            };

            vst1q_s32(output_ptr + x, result.val[0]);
            vst1q_s32(output_ptr + x + 4, result.val[1]);
            vst1q_s32(output_ptr + x + 8, result.val[2]);
            vst1q_s32(output_ptr + x + 12, result.val[3]);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            int32_t tmp       = static_cast<int32_t>(*(input1_ptr + x)) * static_cast<int32_t>(*(input2_ptr + x));
            *(output_ptr + x) = tmp;
        }
    },
    input1, input2, dst);
}

template <bool is_scale255, bool is_sat>
void mul_U8_U8_U8(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, int n)
{
    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(src1, input1_win);
    Iterator input2(src2, input2_win);
    Iterator dst(out, win);

    const int  window_step_x  = 16 / sizeof(uint8_t);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    execute_window_loop(
        win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const uint8_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<uint8_t *>(dst.ptr());

        // Compute window_step_x elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const uint8x16_t ta1 = wrapper::vloadq(input1_ptr + x);
            const uint8x16_t ta2 = wrapper::vloadq(input2_ptr + x);

            uint16x8_t       tmp1_high = vmovl_u8(vget_high_u8(ta1));
            const uint16x8_t tmp2_high = vmovl_u8(vget_high_u8(ta2));
            uint16x8_t       tmp1_low  = vmovl_u8(vget_low_u8(ta1));
            const uint16x8_t tmp2_low  = vmovl_u8(vget_low_u8(ta2));

            tmp1_high = vmulq_u16(tmp1_high, tmp2_high);
            tmp1_low  = vmulq_u16(tmp1_low, tmp2_low);

            if(is_scale255)
            {
                tmp1_high = scale255_U16_U16(tmp1_high);
                tmp1_low  = scale255_U16_U16(tmp1_low);
            }
            else
            {
                const int16x8_t vn = vdupq_n_s16(-n);

                if(is_sat)
                {
                    tmp1_high = vqshlq_u16(tmp1_high, vn);
                    tmp1_low  = vqshlq_u16(tmp1_low, vn);
                }
                else
                {
                    tmp1_high = vshlq_u16(tmp1_high, vn);
                    tmp1_low  = vshlq_u16(tmp1_low, vn);
                }
            }
            if(is_sat)
            {
                vst1q_u8(output_ptr + x, vcombine_u8(vqmovn_u16(tmp1_low), vqmovn_u16(tmp1_high)));
            }
            else
            {
                vst1q_u8(output_ptr + x, vcombine_u8(vmovn_u16(tmp1_low), vmovn_u16(tmp1_high)));
            }
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            uint16_t tmp = static_cast<uint16_t>(*(input1_ptr + x)) * static_cast<uint16_t>(*(input2_ptr + x));

            if(is_scale255)
            {
                float tmp_f = static_cast<float>(tmp) * scale255_constant;
                tmp         = static_cast<uint16_t>(tmp_f + 0.5f);
            }
            else
            {
                tmp >>= n;
            }
            if(is_sat && tmp > 255)
            {
                tmp = 255;
            }
            *(output_ptr + x) = static_cast<uint8_t>(tmp);
        }
    },
    input1, input2, dst);
}

template <bool is_scale255, bool is_sat>
inline int16x8_t mul_S16_S16_S16_n_loop(const int16x8_t &src1, const int16x8_t &src2, int n)
{
    int32x4_t       tmp1_high = vmovl_s16(vget_high_s16(src1));
    const int32x4_t tmp2_high = vmovl_s16(vget_high_s16(src2));
    int32x4_t       tmp1_low  = vmovl_s16(vget_low_s16(src1));
    const int32x4_t tmp2_low  = vmovl_s16(vget_low_s16(src2));

    tmp1_high = vmulq_s32(tmp1_high, tmp2_high);
    tmp1_low  = vmulq_s32(tmp1_low, tmp2_low);

    if(is_scale255)
    {
        tmp1_high = scale255_S32_S32(tmp1_high);
        tmp1_low  = scale255_S32_S32(tmp1_low);
    }
    else
    {
        // Right shift amount
        const int32x4_t vn = vdupq_n_s32(-n);
        // Left shift amount
        const int32x4_t vnl = vdupq_n_s32(n);
        // Calculate conversion bit
        const uint32x4_t tmp1_high_u  = vreinterpretq_u32_s32(tmp1_high);
        const uint32x4_t tmp1_low_u   = vreinterpretq_u32_s32(tmp1_low);
        const uint32x4_t sign_high    = vshrq_n_u32(tmp1_high_u, 31);
        const uint32x4_t sign_low     = vshrq_n_u32(tmp1_low_u, 31);
        const int32x4_t  sign_high_s  = vreinterpretq_s32_u32(sign_high);
        const int32x4_t  sign_low_s   = vreinterpretq_s32_u32(sign_low);
        const int32x4_t  convert_high = vsubq_s32(vshlq_s32(sign_high_s, vnl), sign_high_s);
        const int32x4_t  convert_low  = vsubq_s32(vshlq_s32(sign_low_s, vnl), sign_low_s);
        if(is_sat)
        {
            tmp1_high = vqshlq_s32(vaddq_s32(tmp1_high, convert_high), vn);
            tmp1_low  = vqshlq_s32(vaddq_s32(tmp1_low, convert_low), vn);
        }
        else
        {
            tmp1_high = vshlq_s32(vaddq_s32(tmp1_high, convert_high), vn);
            tmp1_low  = vshlq_s32(vaddq_s32(tmp1_low, convert_low), vn);
        }
    }

    if(is_sat)
    {
        return vcombine_s16(vqmovn_s32(tmp1_low), vqmovn_s32(tmp1_high));
    }
    else
    {
        return vcombine_s16(vmovn_s32(tmp1_low), vmovn_s32(tmp1_high));
    }
}

template <bool is_scale255, bool is_sat>
inline int16x8x2_t mul_S16_S16_S16_n_k(const int16x8x2_t &src1, const int16x8x2_t &src2, int n)
{
    const int16x8x2_t result =
    {
        {
            // First 8 elements
            mul_S16_S16_S16_n_loop<is_scale255, is_sat>(src1.val[0], src2.val[0], n),
            // Second 8 elements
            mul_S16_S16_S16_n_loop<is_scale255, is_sat>(src1.val[1], src2.val[1], n)
        }
    };

    return result;
}

template <bool is_scale255, bool is_sat>
void mul_S16_S16_S16(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, int n)
{
    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(src1, input1_win);
    Iterator input2(src2, input2_win);
    Iterator dst(out, win);

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    execute_window_loop(
        win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const int16_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const int16_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int16_t *>(dst.ptr());

        // Compute window_step_x elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const int16x8x2_t ta1 =
            {
                {
                    vld1q_s16(input1_ptr + x),
                    vld1q_s16(input1_ptr + x + 8),
                }
            };
            const int16x8x2_t ta2 =
            {
                {
                    vld1q_s16(input2_ptr + x),
                    vld1q_s16(input2_ptr + x + 8),
                }
            };
            const int16x8x2_t result = mul_S16_S16_S16_n_k<is_scale255, is_sat>(ta1, ta2, n);

            vst1q_s16(output_ptr + x, result.val[0]);
            vst1q_s16(output_ptr + x + 8, result.val[1]);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            int32_t tmp = static_cast<int32_t>(*(input1_ptr + x)) * static_cast<int32_t>(*(input2_ptr + x));

            if(is_scale255)
            {
                float tmp_f = static_cast<float>(tmp) * scale255_constant;

                tmp = static_cast<int32_t>(tmp_f + 0.5f);
            }
            else
            {
                if(tmp >= 0)
                {
                    tmp >>= n;
                }
                else
                {
                    uint32_t mask = (1u << n) - 1;
                    tmp           = (tmp + static_cast<int32_t>(mask)) >> n;
                }
            }
            if(is_sat)
            {
                tmp = (tmp > SHRT_MAX) ? SHRT_MAX : ((tmp < SHRT_MIN) ? SHRT_MIN : tmp);
            }
            *(output_ptr + x) = static_cast<int16_t>(tmp);
        }
    },
    input1, input2, dst);
}

template <bool is_sat>
inline int32x4_t mul_S32_S32_S32_n_loop(const int32x4_t &src1, const int32x4_t &src2, int n)
{
    const int32x2_t input1_1 = vget_low_s32(src1);
    const int32x2_t input2_1 = vget_low_s32(src2);
    const int32x2_t input1_2 = vget_high_s32(src1);
    const int32x2_t input2_2 = vget_high_s32(src2);

    int64x2_t tmp_1 = vmull_s32(input1_1, input2_1);
    int64x2_t tmp_2 = vmull_s32(input1_2, input2_2);

    // Apply scaling, conversion and rounding (round to zero)
    // Right shift amount
    const int64x2_t vn = vdupq_n_s64(-n);
    // Left shift amount
    const int64x2_t vnl = vdupq_n_s64(n);
    // Calculate conversion bit
    const uint64x2_t tmp_1_u   = vreinterpretq_u64_s64(tmp_1);
    const uint64x2_t sign_1    = vshrq_n_u64(tmp_1_u, 63);
    const int64x2_t  sign_1_s  = vreinterpretq_s64_u64(sign_1);
    const int64x2_t  convert_1 = vsubq_s64(vshlq_s64(sign_1_s, vnl), sign_1_s);

    const uint64x2_t tmp_2_u   = vreinterpretq_u64_s64(tmp_2);
    const uint64x2_t sign_2    = vshrq_n_u64(tmp_2_u, 63);
    const int64x2_t  sign_2_s  = vreinterpretq_s64_u64(sign_2);
    const int64x2_t  convert_2 = vsubq_s64(vshlq_s64(sign_2_s, vnl), sign_2_s);
    if(is_sat)
    {
        tmp_1 = vqshlq_s64(vaddq_s64(tmp_1, convert_1), vn);
        tmp_2 = vqshlq_s64(vaddq_s64(tmp_2, convert_2), vn);
        return vcombine_s32(vqmovn_s64(tmp_1), vqmovn_s64(tmp_2));
    }
    else
    {
        tmp_1 = vshlq_s64(vaddq_s64(tmp_1, convert_1), vn);
        tmp_2 = vshlq_s64(vaddq_s64(tmp_2, convert_2), vn);
        return vcombine_s32(vmovn_s64(tmp_1), vmovn_s64(tmp_2));
    }
}

template <bool is_sat>
inline int32x4x2_t mul_S32_S32_S32_n_k(const int32x4x2_t &src1, const int32x4x2_t &src2, int n)
{
    const int32x4x2_t result =
    {
        {
            // First 4 elements
            mul_S32_S32_S32_n_loop<is_sat>(src1.val[0], src2.val[0], n),
            // Second 4 elements
            mul_S32_S32_S32_n_loop<is_sat>(src1.val[1], src2.val[1], n)
        }
    };

    return result;
}

template <bool is_sat>
void mul_S32_S32_S32(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, int n)
{
    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x         = 8;
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = src1->info()->tensor_shape().x() != src2->info()->tensor_shape().x();

    if(is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? src2 : src1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src2 : src1;

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator dst(out, win);

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const int32_t *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<int32_t *>(dst.ptr());

            const int32_t broadcast_value     = *reinterpret_cast<const int32_t *>(broadcast_input.ptr());
            const auto    broadcast_value_vec = vdupq_n_s32(broadcast_value);

            // Compute window_step_x elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const int32x4x2_t broadcast_v =
                {
                    {
                        broadcast_value_vec,
                        broadcast_value_vec,
                    }
                };
                const int32x4x2_t non_broadcast_v =
                {
                    {
                        vld1q_s32(non_broadcast_input_ptr + x),
                        vld1q_s32(non_broadcast_input_ptr + x + 4),
                    }
                };
                const int32x4x2_t result = mul_S32_S32_S32_n_k<is_sat>(broadcast_v, non_broadcast_v, n);

                vst1q_s32(output_ptr + x, result.val[0]);
                vst1q_s32(output_ptr + x + 4, result.val[1]);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                int64_t tmp = static_cast<int64_t>(broadcast_value) * static_cast<int64_t>(*(non_broadcast_input_ptr + x));

                if(tmp >= 0)
                {
                    tmp >>= n;
                }
                else
                {
                    uint64_t mask = ((uint64_t)1u << n) - 1;
                    tmp           = (tmp + static_cast<int64_t>(mask)) >> n;
                }
                if(is_sat)
                {
                    tmp = utility::clamp<int64_t, int32_t>(tmp);
                }
                *(output_ptr + x) = static_cast<int32_t>(tmp);
            }
        },
        broadcast_input, non_broadcast_input, dst);
    }
    else
    {
        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(src1, input1_win);
        Iterator input2(src2, input2_win);
        Iterator dst(out, win);

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const int32_t *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const int32_t *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<int32_t *>(dst.ptr());

            // Compute window_step_x elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const int32x4x2_t ta1 =
                {
                    {
                        vld1q_s32(input1_ptr + x),
                        vld1q_s32(input1_ptr + x + 4),
                    }
                };
                const int32x4x2_t ta2 =
                {
                    {
                        vld1q_s32(input2_ptr + x),
                        vld1q_s32(input2_ptr + x + 4),
                    }
                };
                const int32x4x2_t result = mul_S32_S32_S32_n_k<is_sat>(ta1, ta2, n);

                vst1q_s32(output_ptr + x, result.val[0]);
                vst1q_s32(output_ptr + x + 4, result.val[1]);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                int64_t tmp = static_cast<int64_t>(*(input1_ptr + x)) * static_cast<int64_t>(*(input2_ptr + x));

                if(tmp >= 0)
                {
                    tmp >>= n;
                }
                else
                {
                    uint64_t mask = ((uint64_t)1u << n) - 1;
                    tmp           = (tmp + static_cast<int64_t>(mask)) >> n;
                }
                if(is_sat)
                {
                    tmp = utility::clamp<int64_t, int32_t>(tmp);
                }
                *(output_ptr + x) = static_cast<int32_t>(tmp);
            }
        },
        input1, input2, dst);
    }
}

void mul_F32_F32_F32(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, float scale)
{
    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x         = 16 / sizeof(float);
    const auto    window_start_x        = static_cast<int>(window.x().start());
    const auto    window_end_x          = static_cast<int>(window.x().end());
    const bool    is_broadcast_across_x = src1->info()->tensor_shape().x() != src2->info()->tensor_shape().x();

    using ExactTagType = typename wrapper::traits::neon_vector<float, window_step_x>::tag_type;

    if(is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? src2 : src1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src2 : src1;

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator dst(out, win);

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const float *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<float *>(dst.ptr());

            const float broadcast_value     = *reinterpret_cast<const float *>(broadcast_input.ptr());
            const auto  broadcast_value_vec = wrapper::vdup_n(broadcast_value, ExactTagType{});
            const auto  scale_vec           = wrapper::vdup_n(scale, ExactTagType{});

            // Compute window_step_x elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto non_broadcast_v = wrapper::vloadq(non_broadcast_input_ptr + x);
                auto       res             = wrapper::vmul(wrapper::vmul(broadcast_value_vec, non_broadcast_v), scale_vec);
                wrapper::vstore(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto non_broadcast_v = *(non_broadcast_input_ptr + x);
                *(output_ptr + x)          = broadcast_value * non_broadcast_v * scale;
            }
        },
        broadcast_input, non_broadcast_input, dst);
    }
    else
    {
        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(src1, input1_win);
        Iterator input2(src2, input2_win);
        Iterator dst(out, win);

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const float *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const float *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<float *>(dst.ptr());

            // Compute window_step_x elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto ta1       = wrapper::vloadq(input1_ptr + x);
                const auto ta2       = wrapper::vloadq(input2_ptr + x);
                const auto scale_vec = wrapper::vdup_n(scale, ExactTagType{});
                const auto res       = wrapper::vmul(wrapper::vmul(ta1, ta2), scale_vec);
                wrapper::vstore(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto ta1    = *(input1_ptr + x);
                const auto ta2    = *(input2_ptr + x);
                *(output_ptr + x) = ta1 * ta2 * scale;
            }
        },
        input1, input2, dst);
    }
}

void c_mul_F32_F32_F32_n(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window)
{
    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x         = 8 / sizeof(float);
    const auto    window_start_x        = static_cast<int>(window.x().start());
    const auto    window_end_x          = static_cast<int>(window.x().end());
    const bool    is_broadcast_across_x = src1->info()->tensor_shape().x() != src2->info()->tensor_shape().x();

    using ExactTagType = typename wrapper::traits::neon_vector<float, 2>::tag_type;

    if(is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? src2 : src1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src2 : src1;

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator dst(out, win);

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const float *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<float *>(dst.ptr());

            const float broadcast_value = *reinterpret_cast<const float *>(broadcast_input.ptr());

            // Compute window_step_x elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto  a = wrapper::vloadq(non_broadcast_input_ptr + 2 * x);
                float32x4_t b = vdupq_n_f32(broadcast_value);

                const float32x4_t mask  = { -1.0f, 1.0f, -1.0f, 1.0f };
                const float32x2_t tmp00 = wrapper::vdup_n(wrapper::vgetlane(a, 0), ExactTagType{});
                const float32x2_t tmp01 = wrapper::vdup_n(wrapper::vgetlane(a, 1), ExactTagType{});
                const float32x2_t tmp10 = wrapper::vdup_n(wrapper::vgetlane(a, 2), ExactTagType{});
                const float32x2_t tmp11 = wrapper::vdup_n(wrapper::vgetlane(a, 3), ExactTagType{});

                const float32x4_t tmp0 = wrapper::vcombine(tmp00, tmp10);
                const float32x4_t tmp1 = wrapper::vcombine(tmp01, tmp11);

                float32x4_t res = wrapper::vmul(tmp0, b);
                b               = wrapper::vmul(b, mask);

                res = wrapper::vmla(res, tmp1, b);
                wrapper::vstore(output_ptr + 2 * x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto non_broadcast_value0 = *(non_broadcast_input_ptr + 2 * x);
                const auto non_broadcast_value1 = *(non_broadcast_input_ptr + 2 * x + 1);
                auto       res1                 = broadcast_value * (non_broadcast_value0 - non_broadcast_value1);
                auto       res2                 = broadcast_value * (non_broadcast_value1 + non_broadcast_value0);
                *(output_ptr + 2 * x)           = res1;
                *(output_ptr + 2 * x + 1)       = res2;
            }
        },
        broadcast_input, non_broadcast_input, dst);
    }
    else
    {
        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(src1, input1_win);
        Iterator input2(src2, input2_win);
        Iterator dst(out, win);

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const float *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const float *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<float *>(dst.ptr());

            // Compute window_step_x elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const float32x4_t a = wrapper::vloadq(input1_ptr + 2 * x);
                float32x4_t       b = wrapper::vloadq(input2_ptr + 2 * x);

                const float32x4_t mask  = { -1.0f, 1.0f, -1.0f, 1.0f };
                const float32x2_t tmp00 = wrapper::vdup_n(wrapper::vgetlane(a, 0), ExactTagType{});
                const float32x2_t tmp01 = wrapper::vdup_n(wrapper::vgetlane(a, 1), ExactTagType{});
                const float32x2_t tmp10 = wrapper::vdup_n(wrapper::vgetlane(a, 2), ExactTagType{});
                const float32x2_t tmp11 = wrapper::vdup_n(wrapper::vgetlane(a, 3), ExactTagType{});

                const float32x4_t tmp0 = wrapper::vcombine(tmp00, tmp10);
                const float32x4_t tmp1 = wrapper::vcombine(tmp01, tmp11);

                float32x4_t res = wrapper::vmul(tmp0, b);

                b = wrapper::vrev64(b);
                b = wrapper::vmul(b, mask);

                res = wrapper::vmla(res, tmp1, b);
                wrapper::vstore(output_ptr + 2 * x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto a0             = *(input1_ptr + 2 * x);
                const auto a1             = *(input1_ptr + 2 * x + 1);
                const auto b0             = *(input2_ptr + 2 * x);
                const auto b1             = *(input2_ptr + 2 * x + 1);
                auto       res1           = a0 * b0 - a1 * b1;
                auto       res2           = a0 * b1 + a1 * b0;
                *(output_ptr + 2 * x)     = res1;
                *(output_ptr + 2 * x + 1) = res2;
            }
        },
        input1, input2, dst);
    }
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void mul_F16_F16_F16(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, float scale)
{
    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    constexpr int window_step_x         = 16;
    const auto    window_start_x        = static_cast<int>(window.x().start());
    const auto    window_end_x          = static_cast<int>(window.x().end());
    const bool    is_broadcast_across_x = src1->info()->tensor_shape().x() != src2->info()->tensor_shape().x();
    if(is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? src2 : src1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src2 : src1;
        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator dst(out, win);
        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto          non_broadcast_input_ptr = reinterpret_cast<const float16_t *>(non_broadcast_input.ptr());
            const auto          output_ptr              = reinterpret_cast<float16_t *>(dst.ptr());
            const auto          broadcast_value         = *reinterpret_cast<const float16_t *>(broadcast_input.ptr());
            const float16x8x2_t broadcast_value_vec     =
            {
                {
                    vdupq_n_f16(broadcast_value),
                    vdupq_n_f16(broadcast_value),
                }
            };
            const auto scale_vec = vdupq_n_f16(scale);
            // Compute window_step_x elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const float16x8x2_t non_broadcast_v =
                {
                    {
                        vld1q_f16(non_broadcast_input_ptr + x),
                        vld1q_f16(non_broadcast_input_ptr + x + 8),
                    }
                };
                const float16x8x2_t result =
                {
                    {
                        vmulq_f16(vmulq_f16(broadcast_value_vec.val[0], non_broadcast_v.val[0]), scale_vec),
                        vmulq_f16(vmulq_f16(broadcast_value_vec.val[1], non_broadcast_v.val[1]), scale_vec),
                    }
                };
                vst1q_f16(output_ptr + x, result.val[0]);
                vst1q_f16(output_ptr + x + 8, result.val[1]);
            }
            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto non_broadcast_v = *(non_broadcast_input_ptr + x);
                *(output_ptr + x)          = broadcast_value * non_broadcast_v * scale;
            }
        },
        broadcast_input, non_broadcast_input, dst);
    }
    else
    {
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        Iterator input1(src1, input1_win);
        Iterator input2(src2, input2_win);
        Iterator dst(out, win);
        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const float16_t *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const float16_t *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<float16_t *>(dst.ptr());
            // Compute window_step_x elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const float16x8x2_t ta1 =
                {
                    {
                        vld1q_f16(input1_ptr + x),
                        vld1q_f16(input1_ptr + x + 8),
                    }
                };
                const float16x8x2_t ta2 =
                {
                    {
                        vld1q_f16(input2_ptr + x),
                        vld1q_f16(input2_ptr + x + 8),
                    }
                };
                const float16x8_t   scale_vec = vdupq_n_f16(scale);
                const float16x8x2_t result    =
                {
                    {
                        vmulq_f16(vmulq_f16(ta1.val[0], ta2.val[0]), scale_vec),
                        vmulq_f16(vmulq_f16(ta1.val[1], ta2.val[1]), scale_vec),
                    }
                };
                vst1q_f16(output_ptr + x, result.val[0]);
                vst1q_f16(output_ptr + x + 8, result.val[1]);
            }
            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto ta1    = *(input1_ptr + x);
                const auto ta2    = *(input2_ptr + x);
                *(output_ptr + x) = ta1 * ta2 * scale;
            }
        },
        input1, input2, dst);
    }
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template <bool is_scale255, bool is_sat>
void mul_U8_U8_S16(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, int n)
{
    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(src1, input1_win);
    Iterator input2(src2, input2_win);
    Iterator dst(out, win);

    const int  window_step_x  = 16 / sizeof(uint8_t);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    execute_window_loop(
        win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const uint8_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int16_t *>(dst.ptr());

        // Compute window_step_x elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const uint8x16_t bv = wrapper::vloadq(input2_ptr + x);
            const uint8x16_t av = wrapper::vloadq(input1_ptr + x);

            uint16x8_t tmp_low  = vmovl_u8(vget_low_u8(av));
            uint16x8_t tmp_high = vmovl_u8(vget_high_u8(av));
            tmp_low             = vmulq_u16(tmp_low, vmovl_u8(vget_low_u8(bv)));
            tmp_high            = vmulq_u16(tmp_high, vmovl_u8(vget_high_u8(bv)));

            if(is_scale255)
            {
                tmp_low  = scale255_U16_U16(tmp_low);
                tmp_high = scale255_U16_U16(tmp_high);
            }
            else
            {
                const int16x8_t vn = vdupq_n_s16(-n);

                if(is_sat)
                {
                    tmp_low  = vqshlq_u16(tmp_low, vn);
                    tmp_high = vqshlq_u16(tmp_high, vn);
                }
                else
                {
                    tmp_low  = vshlq_u16(tmp_low, vn);
                    tmp_high = vshlq_u16(tmp_high, vn);
                }
            }

            if(is_sat)
            {
                static const uint16x8_t max = vdupq_n_u16(SHRT_MAX);

                tmp_low  = vminq_u16(tmp_low, max);
                tmp_high = vminq_u16(tmp_high, max);
            }

            vst1q_s16(output_ptr + x, vreinterpretq_s16_u16(tmp_low));
            vst1q_s16(output_ptr + x + 8, vreinterpretq_s16_u16(tmp_high));
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            int32_t tmp = static_cast<int32_t>(*(input1_ptr + x)) * static_cast<int32_t>(*(input2_ptr + x));

            if(is_scale255)
            {
                float tmp_f = static_cast<float>(tmp) * scale255_constant;
                tmp         = static_cast<int32_t>(tmp_f + 0.5f);
            }
            else
            {
                tmp >>= n;
            }

            if(is_sat)
            {
                tmp = (tmp > SHRT_MAX) ? SHRT_MAX : tmp;
            }

            *(output_ptr + x) = static_cast<int16_t>(tmp);
        }
    },
    input1, input2, dst);
}

template <bool is_scale255, bool is_sat>
void mul_S16_U8_S16(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, int n)
{
    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(src1, input1_win);
    Iterator input2(src2, input2_win);
    Iterator dst(out, win);

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    execute_window_loop(
        win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const int16_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int16_t *>(dst.ptr());

        // Compute window_step_x elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const int16x8x2_t ta1 =
            {
                {
                    vld1q_s16(input1_ptr + x),
                    vld1q_s16(input1_ptr + x + 8),
                }
            };
            const uint8x8x2_t ta2u =
            {
                {
                    vld1_u8(input2_ptr + x),
                    vld1_u8(input2_ptr + x + 8),
                }
            };
            const int16x8x2_t ta2 =
            {
                {
                    vreinterpretq_s16_u16(vmovl_u8(ta2u.val[0])),
                    vreinterpretq_s16_u16(vmovl_u8(ta2u.val[1]))
                }
            };

            const int16x8x2_t result = mul_S16_S16_S16_n_k<is_scale255, is_sat>(ta1, ta2, n);

            vst1q_s16(output_ptr + x, result.val[0]);
            vst1q_s16(output_ptr + x + 8, result.val[1]);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            int32_t tmp = static_cast<int32_t>(*(input1_ptr + x)) * static_cast<int32_t>(*(input2_ptr + x));

            if(is_scale255)
            {
                float tmp_f = static_cast<float>(tmp) * scale255_constant;

                tmp = static_cast<int32_t>(tmp_f + 0.5f);
            }
            else
            {
                if(tmp >= 0)
                {
                    tmp >>= n;
                }
                else
                {
                    uint32_t mask = (1u << n) - 1;
                    tmp           = (tmp + static_cast<int32_t>(mask)) >> n;
                }
            }
            if(is_sat)
            {
                tmp = (tmp > SHRT_MAX) ? SHRT_MAX : ((tmp < SHRT_MIN) ? SHRT_MIN : tmp);
            }
            *(output_ptr + x) = static_cast<int16_t>(tmp);
        }
    },
    input1, input2, dst);
}

template <bool is_scale255, bool is_sat>
void mul_U8_S16_S16(const ITensor *src1, const ITensor *src2, ITensor *out, const Window &window, int n)
{
    // Simply swap the two input buffers
    mul_S16_U8_S16<is_scale255, is_sat>(src2, src1, out, window, n);
}
} // namespace

void CpuMulKernel::configure(ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy)
{
    ARM_COMPUTE_UNUSED(rounding_policy);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, dst);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src1, src2, dst, scale, overflow_policy, rounding_policy));

    const TensorShape &out_shape = TensorShape::broadcast_shape(src1->tensor_shape(), src2->tensor_shape());

    // Auto initialize dst if not initialized
    set_shape_if_empty(*dst, out_shape);

    _scale          = scale;
    _scale_exponent = 0;
    _func_quantized = nullptr;
    _func_int       = nullptr;
    _func_float     = nullptr;

    bool is_scale_255 = false;
    // Check and validate scaling factor
    if(std::abs(scale - scale255_constant) < 0.00001f)
    {
        is_scale_255 = true;
    }
    else
    {
        int exponent = 0;

        std::frexp(scale, &exponent);

        // Store the positive exponent. We know that we compute 1/2^n
        // Additionally we need to subtract 1 to compensate that frexp used a mantissa of 0.5
        _scale_exponent = std::abs(exponent - 1);
    }

    const DataType dt_input1 = src1->data_type();
    const DataType dt_input2 = src2->data_type();
    const DataType dt_output = dst->data_type();
    const bool     is_sat    = (overflow_policy == ConvertPolicy::SATURATE);

    switch(dt_input1)
    {
        case DataType::QASYMM8:
            if(dt_input2 == DataType::QASYMM8 && dt_output == DataType::QASYMM8)
            {
                if(mul_q8_neon_fixedpoint_possible(src1, src2, dst, scale))
                {
                    _func_quantized = &mul_q8_neon_fixedpoint<uint8_t>;
                }
                else
                {
                    _func_quantized = &mul_saturate_quantized_8<uint8_t>;
                }
            }
            break;
        case DataType::QASYMM8_SIGNED:
            if(dt_input2 == DataType::QASYMM8_SIGNED)
            {
                if(mul_q8_neon_fixedpoint_possible(src1, src2, dst, scale))
                {
                    _func_quantized = &mul_q8_neon_fixedpoint<int8_t>;
                }
                else
                {
                    _func_quantized = &mul_saturate_quantized_8<int8_t>;
                }
            }
            break;
        case DataType::QSYMM16:
            if(dt_input2 == DataType::QSYMM16 && dt_output == DataType::QSYMM16)
            {
                _func_quantized = &mul_saturate_QSYMM16_QSYMM16_QSYMM16;
            }
            else if(dt_input2 == DataType::QSYMM16 && dt_output == DataType::S32)
            {
                _func_int = &mul_QSYMM16_QSYMM16_S32;
            }
            break;
        case DataType::S16:
            if(DataType::U8 == dt_input2 && DataType::S16 == dt_output)
            {
                if(is_scale_255)
                {
                    _func_int = is_sat ? &mul_S16_U8_S16<true, true> : &mul_S16_U8_S16<true, false>;
                }
                else
                {
                    _func_int = is_sat ? &mul_S16_U8_S16<false, true> : &mul_S16_U8_S16<false, false>;
                }
            }
            if(DataType::S16 == dt_input2 && DataType::S16 == dt_output)
            {
                if(is_scale_255)
                {
                    _func_int = is_sat ? &mul_S16_S16_S16<true, true> : &mul_S16_S16_S16<true, false>;
                }
                else
                {
                    _func_int = is_sat ? &mul_S16_S16_S16<false, true> : &mul_S16_S16_S16<false, false>;
                }
            }
            break;
        case DataType::S32:
            if(DataType::S32 == dt_input2 && DataType::S32 == dt_output)
            {
                _func_int = is_sat ? &mul_S32_S32_S32<true> : &mul_S32_S32_S32<false>;
            }
            break;
        case DataType::U8:
            if(DataType::U8 == dt_input2 && DataType::U8 == dt_output)
            {
                if(is_scale_255)
                {
                    _func_int = is_sat ? &mul_U8_U8_U8<true, true> : &mul_U8_U8_U8<true, false>;
                }
                else
                {
                    _func_int = is_sat ? &mul_U8_U8_U8<false, true> : &mul_U8_U8_U8<false, false>;
                }
            }
            else if(DataType::U8 == dt_input2 && DataType::S16 == dt_output)
            {
                if(is_scale_255)
                {
                    _func_int = is_sat ? &mul_U8_U8_S16<true, true> : &mul_U8_U8_S16<true, false>;
                }
                else
                {
                    _func_int = is_sat ? &mul_U8_U8_S16<false, true> : &mul_U8_U8_S16<false, false>;
                }
            }
            else if(DataType::S16 == dt_input2 && DataType::S16 == dt_output)
            {
                if(is_scale_255)
                {
                    _func_int = is_sat ? &mul_U8_S16_S16<true, true> : &mul_U8_S16_S16<true, false>;
                }
                else
                {
                    _func_int = is_sat ? &mul_U8_S16_S16<false, true> : &mul_U8_S16_S16<false, false>;
                }
            }
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func_float = &mul_F16_F16_F16;
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
            _func_float = &mul_F32_F32_F32;
            break;
        default:
            ARM_COMPUTE_ERROR("You called with the wrong img formats");
    }

    // Configure kernel window
    Window win;
    std::tie(win, _split_dimension) = calculate_squashed_or_max_window(*src1, *src2);

    ICpuKernel::configure(win);
}

size_t CpuMulKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);

#if defined(ENABLE_FP32_KERNELS)
    if(this->_func_float == &mul_F32_F32_F32)
    {
        size_t mws = ICPPKernel::default_mws;
        if(platform.get_cpu_model() == CPUModel::N1)
        {
            mws = default_mws_N1_fp32_neon;
        }
        else if(platform.get_cpu_model() == CPUModel::V1)
        {
            mws = default_mws_V1_fp32_neon;
        }
        else
        {
            if(_split_dimension == Window::DimX)
            {
                // Don't split the work load too small if the tensor has been reinterpreted as 1D.
                // This number is loosely chosen as threading overhead in each platform varies wildly.
                return default_mws_other_platforms_1d_tensor;
            }
            return default_mws;
        }

        // tensor is 1D or was re-interpreted as 1D
        if(this->window().shape().num_dimensions() == 1)
        {
            return mws;
        }
        else
        {
            // scale mws down by the number of elements along all the dimensions (x, z, w, etc) except the one
            // that we parallelize along (the y dimension). This allows for parallelization when the Y_SIZE is small
            // but the other sizes are large, which boosts performance.
            mws = static_cast<size_t>(mws / (this->window().num_iterations_total() / this->window().num_iterations(1)));
            return std::max(static_cast<size_t>(1), mws);
        }
    }
#else /* ENABLE_FP32_KERNELS */
    ARM_COMPUTE_UNUSED(platform);
#endif /* ENABLE_FP32_KERNELS */
    if(_split_dimension == Window::DimX)
    {
        // Don't split the work load too small if the tensor has been reinterpreted as 1D.
        // This number is loosely chosen as threading overhead in each platform varies wildly.
        return default_mws_other_platforms_1d_tensor;
    }
    return default_mws;
}

Status CpuMulKernel::validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float scale, ConvertPolicy overflow_policy,
                              RoundingPolicy rounding_policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src1, src2, dst, scale, overflow_policy, rounding_policy));

    return Status{};
}

void CpuMulKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src1 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto src2 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto dst  = tensors.get_tensor(TensorType::ACL_DST);

    if(_func_quantized != nullptr)
    {
        (*_func_quantized)(src1, src2, dst, window, _scale);
    }
    else if(_func_int != nullptr)
    {
        (*_func_int)(src1, src2, dst, window, _scale_exponent);
    }
    else
    {
        ARM_COMPUTE_ERROR_ON(_func_float == nullptr);
        (*_func_float)(src1, src2, dst, window, _scale);
    }
}

const char *CpuMulKernel::name() const
{
    return "CpuMulKernel";
}

namespace
{
Status validate_arguments_complex(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src1, 2, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src2, 2, DataType::F32);

    const TensorShape &out_shape = TensorShape::broadcast_shape(src1->tensor_shape(), src2->tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured dst
    if(dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 2, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst->tensor_shape(), 0), "Wrong shape for dst");
    }

    return Status{};
}
} // namespace

void CpuComplexMulKernel::configure(ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_complex(src1, src2, dst));

    const TensorShape &out_shape = TensorShape::broadcast_shape(src1->tensor_shape(), src2->tensor_shape());

    // Auto initialize dst if not initialized
    const TensorInfo out_info(out_shape, src1->num_channels(), src1->data_type());
    auto_init_if_empty(*dst, out_info);

    // Configure kernel window
    Window win = calculate_max_window(out_shape);

    ICpuKernel::configure(win);
}

Status CpuComplexMulKernel::validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_complex(src1, src2, dst));

    return Status{};
}

void CpuComplexMulKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src1 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto src2 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto dst  = tensors.get_tensor(TensorType::ACL_DST);

    c_mul_F32_F32_F32_n(src1, src2, dst, window);
}

const char *CpuComplexMulKernel::name() const
{
    return "CpuComplexMulKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
