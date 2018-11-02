/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEPixelWiseMultiplicationKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include <arm_fp16.h> // needed for float16_t
#endif                /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
const float       scale255_constant      = 1.f / 255.f;
const float32x4_t scale255_constant_f32q = vdupq_n_f32(scale255_constant);
const float32x4_t positive_round_f32q    = vdupq_n_f32(0.5f);

constexpr unsigned int num_elems_processed_per_iteration = 16;

inline Status validate_arguments(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy)
{
    ARM_COMPUTE_UNUSED(overflow_policy);
    ARM_COMPUTE_UNUSED(rounding_policy);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8, DataType::QS8, DataType::QS16, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::U8, DataType::QS8, DataType::QS16, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::QS8, DataType::QS16, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->data_type() == DataType::U8 && (input1->data_type() != DataType::U8 || input2->data_type() != DataType::U8),
                                    "Output can only be U8 if both inputs are U8");

    const TensorShape &out_shape = TensorShape::broadcast_shape(input1->tensor_shape(), input2->tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output->tensor_shape(), 0), "Wrong shape for output");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    if(is_data_type_fixed_point(input1->data_type()) || is_data_type_fixed_point(input2->data_type()) || is_data_type_fixed_point(output->data_type()))
    {
        // Check that all data types are the same and all fixed-point positions are the same
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input1, input2, output);
        // Check if scale is representable in fixed-point with the provided settings
        ARM_COMPUTE_RETURN_ERROR_ON_VALUE_NOT_REPRESENTABLE_IN_FIXED_POINT(scale, input1);
    }

    if(std::abs(scale - scale255_constant) < 0.00001f)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(rounding_policy != RoundingPolicy::TO_NEAREST_UP && rounding_policy != RoundingPolicy::TO_NEAREST_EVEN);
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

inline std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*input1, *input2);
    const ValidRegion &valid_region = broadcast_pair.second;

    // Auto initialize output if not initialized
    {
        set_shape_if_empty(*output, input1->tensor_shape());

        if(input1->data_type() == DataType::S16 || input2->data_type() == DataType::S16)
        {
            set_format_if_unknown(*output, Format::S16);
        }
        else if(input1->data_type() == DataType::F32 || input2->data_type() == DataType::F32)
        {
            set_format_if_unknown(*output, Format::F32);
        }
        else if(input1->data_type() == DataType::F16 || input2->data_type() == DataType::F16)
        {
            set_format_if_unknown(*output, Format::F16);
        }
        else if(input1->data_type() == DataType::QS8 && input2->data_type() == DataType::QS8)
        {
            set_data_type_if_unknown(*output, DataType::QS8);
            set_fixed_point_position_if_zero(*output, input1->fixed_point_position());
        }
    }

    // Configure kernel window
    Window win        = calculate_max_window(valid_region, Steps(num_elems_processed_per_iteration));
    Window win_input1 = win.broadcast_if_dimension_le_one(*input1);
    Window win_input2 = win.broadcast_if_dimension_le_one(*input2);

    AccessWindowHorizontal input1_access(input1, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal input2_access(input2, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win_input1, input1_access)
                          || update_window_and_padding(win_input2, input2_access)
                          || update_window_and_padding(win, output_access);

    output_access.set_valid_region(win, valid_region);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

/* Scales a given vector by 1/255.
 *
 * @note This does not work for all cases. e.g. for float of 0.49999999999999994 and large floats.
 *
 * @param in Input vector to scale.
 * @return   Scaled output rounded to nearest (round half up).
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

template <bool is_scale255, bool is_sat>
void mul_U8_U8_U8_n(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int n)
{
    const auto input1 = static_cast<const uint8_t *__restrict>(input1_ptr);
    const auto input2 = static_cast<const uint8_t *__restrict>(input2_ptr);
    const auto output = static_cast<uint8_t *__restrict>(output_ptr);

    const uint8x16_t ta1 = vld1q_u8(input1);
    const uint8x16_t ta2 = vld1q_u8(input2);

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
        vst1q_u8(output, vcombine_u8(vqmovn_u16(tmp1_low), vqmovn_u16(tmp1_high)));
    }
    else
    {
        vst1q_u8(output, vcombine_u8(vmovn_u16(tmp1_low), vmovn_u16(tmp1_high)));
    }
}

template <bool is_scale255, bool is_sat>
void mul_QS8_QS8_QS8_n(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int n, int fixed_point_position)
{
    const auto output = static_cast<qint8_t *__restrict>(output_ptr);

    const qint8x16_t ta1 = vld1q_qs8(static_cast<const qint8_t *__restrict>(input1_ptr));
    const qint8x16_t ta2 = vld1q_qs8(static_cast<const qint8_t *__restrict>(input2_ptr));

    if(is_scale255)
    {
        qint16x8_t       tmp1_high = vmovl_s8(vget_high_s8(ta1));
        qint16x8_t       tmp1_low  = vmovl_s8(vget_low_s8(ta1));
        const qint16x8_t tmp2_high = vmovl_s8(vget_high_s8(ta2));
        const qint16x8_t tmp2_low  = vmovl_s8(vget_low_s8(ta2));

        const float32x4x2_t scale255_f32 =
        {
            {
                scale255_constant_f32q,
                scale255_constant_f32q
            }
        };
        const qint16x8_t scale255 = vqcvtq_qs16_f32(scale255_f32, fixed_point_position);

        tmp1_high = vmulq_qs16(tmp1_high, tmp2_high, fixed_point_position);
        tmp1_low  = vmulq_qs16(tmp1_low, tmp2_low, fixed_point_position);
        tmp1_high = vmulq_qs16(tmp1_high, scale255, fixed_point_position);
        tmp1_low  = vmulq_qs16(tmp1_low, scale255, fixed_point_position);

        if(is_sat)
        {
            vst1q_qs8(output, vcombine_s8(vqmovn_s16(tmp1_low), vqmovn_s16(tmp1_high)));
        }
        else
        {
            vst1q_qs8(output, vcombine_s8(vmovn_s16(tmp1_low), vmovn_s16(tmp1_high)));
        }
    }
    else
    {
        const qint8x16_t vn  = vdupq_n_s8(-n);
        qint8x16_t       res = ta2;

        if(is_sat)
        {
            res = vqshlq_s8(vqmulq_qs8(ta1, res, fixed_point_position), vn);
        }
        else
        {
            res = vshlq_s8(vmulq_qs8(ta1, res, fixed_point_position), vn);
        }
        vst1q_qs8(output, res);
    }
}

template <bool is_scale255, bool is_sat>
void mul_QS16_QS16_QS16_n(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int n, int fixed_point_position)
{
    const qint16x8x2_t ta1 = vld2q_qs16(static_cast<const qint16_t *__restrict>(input1_ptr));
    qint16x8x2_t       res = vld2q_qs16(static_cast<const qint16_t *__restrict>(input2_ptr));

    if(is_scale255)
    {
        const float32x4x2_t scale255_f32 =
        {
            {
                scale255_constant_f32q,
                scale255_constant_f32q
            }
        };
        const qint16x8_t scale255 = vqcvtq_qs16_f32(scale255_f32, fixed_point_position);
        if(is_sat)
        {
            res.val[0] = vqmulq_qs16(vqmulq_qs16(ta1.val[0], res.val[0], fixed_point_position), scale255, fixed_point_position);
            res.val[1] = vqmulq_qs16(vqmulq_qs16(ta1.val[1], res.val[1], fixed_point_position), scale255, fixed_point_position);
        }
        else
        {
            res.val[0] = vmulq_qs16(vmulq_qs16(ta1.val[0], res.val[0], fixed_point_position), scale255, fixed_point_position);
            res.val[1] = vmulq_qs16(vmulq_qs16(ta1.val[1], res.val[1], fixed_point_position), scale255, fixed_point_position);
        }
    }
    else
    {
        const qint16x8_t vn = vdupq_n_s16(-n);
        if(is_sat)
        {
            res.val[0] = vqshlq_s16(vqmulq_qs16(ta1.val[0], res.val[0], fixed_point_position), vn);
            res.val[1] = vqshlq_s16(vqmulq_qs16(ta1.val[1], res.val[1], fixed_point_position), vn);
        }
        else
        {
            res.val[0] = vshlq_s16(vmulq_qs16(ta1.val[0], res.val[0], fixed_point_position), vn);
            res.val[1] = vshlq_s16(vmulq_qs16(ta1.val[1], res.val[1], fixed_point_position), vn);
        }
    }
    vst2q_s16(static_cast<qint16_t *__restrict>(output_ptr), res);
}

template <bool is_scale255, bool is_sat>
inline int16x8_t mul_S16_S16_S16_n_loop(const int16x8_t &input1, const int16x8_t &input2, int n)
{
    int32x4_t       tmp1_high = vmovl_s16(vget_high_s16(input1));
    const int32x4_t tmp2_high = vmovl_s16(vget_high_s16(input2));
    int32x4_t       tmp1_low  = vmovl_s16(vget_low_s16(input1));
    const int32x4_t tmp2_low  = vmovl_s16(vget_low_s16(input2));

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
inline int16x8x2_t mul_S16_S16_S16_n_k(const int16x8x2_t &input1, const int16x8x2_t &input2, int n)
{
    const int16x8x2_t result =
    {
        {
            // First 8 elements
            mul_S16_S16_S16_n_loop<is_scale255, is_sat>(input1.val[0], input2.val[0], n),
            // Second 8 elements
            mul_S16_S16_S16_n_loop<is_scale255, is_sat>(input1.val[1], input2.val[1], n)
        }
    };

    return result;
}

template <bool is_scale255, bool is_sat>
void mul_S16_S16_S16_n(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int n)
{
    const auto input1 = static_cast<const int16_t *__restrict>(input1_ptr);
    const auto input2 = static_cast<const int16_t *__restrict>(input2_ptr);
    const auto output = static_cast<int16_t *__restrict>(output_ptr);

    const int16x8x2_t ta1    = vld2q_s16(input1);
    const int16x8x2_t ta2    = vld2q_s16(input2);
    const int16x8x2_t result = mul_S16_S16_S16_n_k<is_scale255, is_sat>(ta1, ta2, n);

    vst2q_s16(output, result);
}

template <bool is_scale255, bool is_sat>
void mul_F32_F32_F32_n(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, float scale)
{
    const auto input1 = static_cast<const float *__restrict>(input1_ptr);
    const auto input2 = static_cast<const float *__restrict>(input2_ptr);
    const auto output = static_cast<float *__restrict>(output_ptr);

    const float32x4x4_t ta1       = vld4q_f32(input1);
    const float32x4x4_t ta2       = vld4q_f32(input2);
    const float32x4_t   scale_vec = vdupq_n_f32(scale);
    const float32x4x4_t result =
    {
        {
            vmulq_f32(vmulq_f32(ta1.val[0], ta2.val[0]), scale_vec),
            vmulq_f32(vmulq_f32(ta1.val[1], ta2.val[1]), scale_vec),
            vmulq_f32(vmulq_f32(ta1.val[2], ta2.val[2]), scale_vec),
            vmulq_f32(vmulq_f32(ta1.val[3], ta2.val[3]), scale_vec)
        }
    };
    vst4q_f32(output, result);
}

template <bool is_scale255, bool is_sat>
void mul_F16_F16_F16_n(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, float scale)
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    const auto          input1    = static_cast<const float16_t *__restrict>(input1_ptr);
    const auto          input2    = static_cast<const float16_t *__restrict>(input2_ptr);
    const auto          output    = static_cast<float16_t *__restrict>(output_ptr);
    const float16x8x2_t ta1       = vld2q_f16(input1);
    const float16x8x2_t ta2       = vld2q_f16(input2);
    const float16x8_t   scale_vec = vdupq_n_f16(scale);
    const float16x8x2_t result =
    {
        {
            vmulq_f16(vmulq_f16(ta1.val[0], ta2.val[0]), scale_vec),
            vmulq_f16(vmulq_f16(ta1.val[1], ta2.val[1]), scale_vec),
        }
    };
    vst2q_f16(output, result);
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    ARM_COMPUTE_UNUSED(input1_ptr);
    ARM_COMPUTE_UNUSED(input2_ptr);
    ARM_COMPUTE_UNUSED(output_ptr);
    ARM_COMPUTE_UNUSED(scale);
    ARM_COMPUTE_ERROR("Not supported. Recompile the library with arch=arm64-v8.2-a.");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

template <bool is_scale255, bool is_sat>
void mul_U8_U8_S16_n(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int n)
{
    const auto input1 = static_cast<const uint8_t *__restrict>(input1_ptr);
    const auto input2 = static_cast<const uint8_t *__restrict>(input2_ptr);
    const auto output = static_cast<int16_t *__restrict>(output_ptr);

    const uint8x16_t bv = vld1q_u8(input2);
    const uint8x16_t av = vld1q_u8(input1);

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

    vst1q_s16(output, vreinterpretq_s16_u16(tmp_low));
    vst1q_s16(output + 8, vreinterpretq_s16_u16(tmp_high));
}

template <bool is_scale255, bool is_sat>
void mul_S16_U8_S16_n(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int n)
{
    const auto input1 = static_cast<const int16_t *__restrict>(input1_ptr);
    const auto input2 = static_cast<const uint8_t *__restrict>(input2_ptr);
    const auto output = static_cast<int16_t *__restrict>(output_ptr);

    const int16x8x2_t ta1  = vld2q_s16(input1);
    const uint8x8x2_t ta2u = vld2_u8(input2);
    const int16x8x2_t ta2 =
    {
        {
            vreinterpretq_s16_u16(vmovl_u8(ta2u.val[0])),
            vreinterpretq_s16_u16(vmovl_u8(ta2u.val[1]))
        }
    };

    const int16x8x2_t result = mul_S16_S16_S16_n_k<is_scale255, is_sat>(ta1, ta2, n);

    vst2q_s16(output, result);
}

template <bool is_scale255, bool is_sat>
void mul_U8_S16_S16_n(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int n)
{
    // Simply swap the two input buffers
    mul_S16_U8_S16_n<is_scale255, is_sat>(input2_ptr, input1_ptr, output_ptr, n);
}
} // namespace

NEPixelWiseMultiplicationKernel::NEPixelWiseMultiplicationKernel()
    : _func_float(nullptr), _func_int(nullptr), _func_q_int(nullptr), _input1(nullptr), _input2(nullptr), _output(nullptr), _scale{ 0 }, _scale_exponent{ 0 }
{
}

void NEPixelWiseMultiplicationKernel::configure(const ITensor *input1, const ITensor *input2, ITensor *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy)
{
    ARM_COMPUTE_UNUSED(rounding_policy);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input1->info(), input2->info(), output->info(), scale, overflow_policy, rounding_policy));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input1->info(), input2->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    _input1         = input1;
    _input2         = input2;
    _output         = output;
    _scale          = scale;
    _scale_exponent = 0;
    _func_int       = nullptr;
    _func_q_int     = nullptr;
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

    const DataType dt_input1 = input1->info()->data_type();
    const DataType dt_input2 = input2->info()->data_type();
    const DataType dt_output = output->info()->data_type();
    const bool     is_sat    = (overflow_policy == ConvertPolicy::SATURATE);

    if(DataType::U8 == dt_input1 && DataType::U8 == dt_input2 && DataType::U8 == dt_output)
    {
        if(is_scale_255)
        {
            _func_int = is_sat ? &mul_U8_U8_U8_n<true, true> : &mul_U8_U8_U8_n<true, false>;
        }
        else
        {
            _func_int = is_sat ? &mul_U8_U8_U8_n<false, true> : &mul_U8_U8_U8_n<false, false>;
        }
    }
    else if(DataType::S16 == dt_input1 && DataType::S16 == dt_input2 && DataType::S16 == dt_output)
    {
        if(is_scale_255)
        {
            _func_int = is_sat ? &mul_S16_S16_S16_n<true, true> : &mul_S16_S16_S16_n<true, false>;
        }
        else
        {
            _func_int = is_sat ? &mul_S16_S16_S16_n<false, true> : &mul_S16_S16_S16_n<false, false>;
        }
    }
    else if(DataType::S16 == dt_input1 && DataType::U8 == dt_input2 && DataType::S16 == dt_output)
    {
        if(is_scale_255)
        {
            _func_int = is_sat ? &mul_S16_U8_S16_n<true, true> : &mul_S16_U8_S16_n<true, false>;
        }
        else
        {
            _func_int = is_sat ? &mul_S16_U8_S16_n<false, true> : &mul_S16_U8_S16_n<false, false>;
        }
    }
    else if(DataType::U8 == dt_input1 && DataType::S16 == dt_input2 && DataType::S16 == dt_output)
    {
        if(is_scale_255)
        {
            _func_int = is_sat ? &mul_U8_S16_S16_n<true, true> : &mul_U8_S16_S16_n<true, false>;
        }
        else
        {
            _func_int = is_sat ? &mul_U8_S16_S16_n<false, true> : &mul_U8_S16_S16_n<false, false>;
        }
    }
    else if(DataType::U8 == dt_input1 && DataType::U8 == dt_input2 && DataType::S16 == dt_output)
    {
        if(is_scale_255)
        {
            _func_int = is_sat ? &mul_U8_U8_S16_n<true, true> : &mul_U8_U8_S16_n<true, false>;
        }
        else
        {
            _func_int = is_sat ? &mul_U8_U8_S16_n<false, true> : &mul_U8_U8_S16_n<false, false>;
        }
    }
    else if(DataType::QS8 == dt_input1 && DataType::QS8 == dt_input2 && DataType::QS8 == dt_output)
    {
        if(is_scale_255)
        {
            _func_q_int = is_sat ? &mul_QS8_QS8_QS8_n<true, true> : &mul_QS8_QS8_QS8_n<true, false>;
        }
        else
        {
            _func_q_int = is_sat ? &mul_QS8_QS8_QS8_n<false, true> : &mul_QS8_QS8_QS8_n<false, false>;
        }
    }
    else if(DataType::QS16 == dt_input1 && DataType::QS16 == dt_input2 && DataType::QS16 == dt_output)
    {
        if(is_scale_255)
        {
            _func_q_int = is_sat ? &mul_QS16_QS16_QS16_n<true, true> : &mul_QS16_QS16_QS16_n<true, false>;
        }
        else
        {
            _func_q_int = is_sat ? &mul_QS16_QS16_QS16_n<false, true> : &mul_QS16_QS16_QS16_n<false, false>;
        }
    }
    else if(DataType::F16 == dt_input1 && DataType::F16 == dt_input2 && DataType::F16 == dt_output)
    {
        _func_float = &mul_F16_F16_F16_n<false, false>;
        _func_int   = nullptr;
    }
    else if(DataType::F32 == dt_input1 && DataType::F32 == dt_input2 && DataType::F32 == dt_output)
    {
        _func_float = &mul_F32_F32_F32_n<false, false>;
        _func_int   = nullptr;
    }
    else
    {
        ARM_COMPUTE_ERROR("You called with the wrong img formats");
    }

    INEKernel::configure(win_config.second);
}

Status NEPixelWiseMultiplicationKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale, ConvertPolicy overflow_policy,
                                                 RoundingPolicy rounding_policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input1, input2, output, scale, overflow_policy, rounding_policy));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input1->clone().get(), input2->clone().get(), output->clone().get()).first);

    return Status{};
}

void NEPixelWiseMultiplicationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const TensorShape &in_shape1 = _input1->info()->tensor_shape();
    const TensorShape &in_shape2 = _input2->info()->tensor_shape();
    const TensorShape &out_shape = _output->info()->tensor_shape();

    bool can_collapse = true;
    if(std::min(in_shape1.total_size(), in_shape2.total_size()) > 1)
    {
        can_collapse = (std::min(in_shape1.num_dimensions(), in_shape2.num_dimensions()) > Window::DimZ);
        for(size_t d = Window::DimZ; can_collapse && (d < out_shape.num_dimensions()); ++d)
        {
            can_collapse = (in_shape1[d] == in_shape2[d]);
        }
    }

    bool   has_collapsed = false;
    Window collapsed     = can_collapse ? window.collapse_if_possible(INEKernel::window(), Window::DimZ, &has_collapsed) : window;

    const TensorShape &in_shape1_collapsed = has_collapsed ? in_shape1.collapsed_from(Window::DimZ) : in_shape1;
    const TensorShape &in_shape2_collapsed = has_collapsed ? in_shape2.collapsed_from(Window::DimZ) : in_shape2;

    Window slice        = collapsed.first_slice_window_3D();
    Window slice_input1 = slice.broadcast_if_dimension_le_one(in_shape1_collapsed);
    Window slice_input2 = slice.broadcast_if_dimension_le_one(in_shape2_collapsed);

    Iterator input1(_input1, slice_input1);
    Iterator input2(_input2, slice_input2);
    Iterator output(_output, slice);

    if(_func_int != nullptr)
    {
        execute_window_loop(collapsed, [&](const Coordinates & id)
        {
            (*_func_int)(input1.ptr(), input2.ptr(), output.ptr(), _scale_exponent);
            collapsed.slide_window_slice_3D(slice_input1);
            collapsed.slide_window_slice_3D(slice_input2);
        },
        input1, input2, output);
    }
    else if(_func_q_int != nullptr)
    {
        int fixed_point_position = _input1->info()->fixed_point_position();
        execute_window_loop(collapsed, [&](const Coordinates & id)
        {
            (*_func_q_int)(input1.ptr(), input2.ptr(), output.ptr(), _scale_exponent, fixed_point_position);
            collapsed.slide_window_slice_3D(slice_input1);
            collapsed.slide_window_slice_3D(slice_input2);
        },
        input1, input2, output);
    }
    else
    {
        ARM_COMPUTE_ERROR_ON(_func_float == nullptr);
        execute_window_loop(collapsed, [&](const Coordinates & id)
        {
            (*_func_float)(input1.ptr(), input2.ptr(), output.ptr(), _scale);
            collapsed.slide_window_slice_3D(slice_input1);
            collapsed.slide_window_slice_3D(slice_input2);
        },
        input1, input2, output);
    }
}

BorderSize NEPixelWiseMultiplicationKernel::border_size() const
{
    const unsigned int replicateSize = _output->info()->dimension(0) - std::min(_input1->info()->dimension(0), _input2->info()->dimension(0));
    const unsigned int border        = std::min<unsigned int>(num_elems_processed_per_iteration - 1U, replicateSize);
    return BorderSize(0, border, 0, 0);
}