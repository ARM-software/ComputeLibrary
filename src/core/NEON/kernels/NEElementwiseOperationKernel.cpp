/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "src/core/NEON/kernels/NEElementwiseOperationKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>
#include <map>

namespace arm_compute
{
namespace
{
float32x4x4_t load_quantized(const uint8_t *input1_ptr, const int32x4_t &offset, const float32x4_t &scale)
{
    qasymm8x16_t        x = vld1q_u8(input1_ptr);
    const float32x4x4_t out =
    {
        {
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(x))))), offset)), scale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(x))))), offset)), scale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(x))))), offset)), scale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(x))))), offset)), scale),
        }
    };
    return out;
}

float32x4x4_t load_quantized_signed(const int8_t *input1_ptr, const int32x4_t &offset, const float32x4_t &scale)
{
    qasymm8x16_signed_t x = vld1q_s8(input1_ptr);
    const float32x4x4_t out =
    {
        {
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(x)))), offset)), scale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(x)))), offset)), scale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(x)))), offset)), scale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(x)))), offset)), scale),
        }
    };
    return out;
}

void store_quantized(uint8_t *output_ptr, const uint32x4x4_t &out)
{
    const uint8x8_t pa = vqmovn_u16(vcombine_u16(vqmovn_u32(out.val[0]), vqmovn_u32(out.val[1])));
    const uint8x8_t pb = vqmovn_u16(vcombine_u16(vqmovn_u32(out.val[2]), vqmovn_u32(out.val[3])));
    vst1q_u8(output_ptr, vcombine_u8(pa, pb));
}

void store_quantized(uint8_t *output_ptr, const int32x4x4_t &out)
{
    const uint8x8_t pa = vqmovun_s16(vcombine_s16(vqmovn_s32(out.val[0]), vqmovn_s32(out.val[1])));
    const uint8x8_t pb = vqmovun_s16(vcombine_s16(vqmovn_s32(out.val[2]), vqmovn_s32(out.val[3])));
    vst1q_u8(output_ptr, vcombine_u8(pa, pb));
}

void store_quantized(uint8_t *output_ptr, const float32x4x4_t &rf, const float32x4_t &offset, const float32x4_t &invscale)
{
    int32x4x4_t out =
    {
        {
            vcvtq_s32_f32(vmlaq_f32(offset, rf.val[0], invscale)),
            vcvtq_s32_f32(vmlaq_f32(offset, rf.val[1], invscale)),
            vcvtq_s32_f32(vmlaq_f32(offset, rf.val[2], invscale)),
            vcvtq_s32_f32(vmlaq_f32(offset, rf.val[3], invscale)),
        }
    };
    store_quantized(output_ptr, out);
}

void store_quantized_signed(int8_t *output_ptr, const int32x4x4_t &out)
{
    const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(out.val[0]), vqmovn_s32(out.val[1])));
    const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(out.val[2]), vqmovn_s32(out.val[3])));
    vst1q_s8(output_ptr, vcombine_s8(pa, pb));
}

void store_quantized_signed(int8_t *output_ptr, const float32x4x4_t &rf, const float32x4_t &offset, const float32x4_t &invscale)
{
    int32x4x4_t out =
    {
        {
            vcvtq_s32_f32(vmlaq_f32(offset, rf.val[0], invscale)),
            vcvtq_s32_f32(vmlaq_f32(offset, rf.val[1], invscale)),
            vcvtq_s32_f32(vmlaq_f32(offset, rf.val[2], invscale)),
            vcvtq_s32_f32(vmlaq_f32(offset, rf.val[3], invscale)),
        }
    };
    store_quantized_signed(output_ptr, out);
}

template <ArithmeticOperation op, typename ScalarType>
inline ScalarType elementwise_arithm_op_scalar(const ScalarType &a, const ScalarType &b)
{
    auto res = ScalarType(0);

    switch(op)
    {
        case ArithmeticOperation::MAX:
            res = std::max(a, b);
            break;
        case ArithmeticOperation::MIN:
            res = std::min(a, b);
            break;
        case ArithmeticOperation::SQUARED_DIFF:
        {
            res = (a - b) * (a - b);
            break;
        }
        case ArithmeticOperation::PRELU:
        {
            res = (a > 0 ? a : a * b);
            break;
        }
        case ArithmeticOperation::DIV:
        {
            res = a / b;
            if(std::is_integral<ScalarType>::value)
            {
                res = (b == 0) ? 0 : res;
                if(static_cast<int32_t>(a) % static_cast<int32_t>(b) != 0 && ((a < 0) != (b < 0)))
                {
                    --res;
                }
            }
            break;
        }
        case ArithmeticOperation::POWER:
        {
            res = std::pow(a, b);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return res;
}

template <ArithmeticOperation op>
inline uint8_t elementwise_arithm_op_quantized_scalar(const float &a, const float &b, UniformQuantizationInfo qinfo)
{
    return quantize_qasymm8(elementwise_arithm_op_scalar<op>(a, b), qinfo);
}

template <ArithmeticOperation op>
inline int8_t elementwise_arithm_op_quantized_signed_scalar(const float &a, const float &b, UniformQuantizationInfo qinfo)
{
    return quantize_qasymm8_signed(elementwise_arithm_op_scalar<op>(a, b), qinfo);
}

template <ArithmeticOperation    op, typename VectorType>
inline typename VectorType::type elementwise_arithm_op(const typename VectorType::type &a, const typename VectorType::type &b)
{
    using vec_type    = typename VectorType::type;
    using scalar_type = typename VectorType::scalar_type;
    using tag_type    = typename VectorType::tag_type;

    vec_type res = wrapper::vdup_n(static_cast<scalar_type>(0), tag_type{});

    switch(op)
    {
        case ArithmeticOperation::MAX:
            res = wrapper::vmax(a, b);
            break;
        case ArithmeticOperation::MIN:
            res = wrapper::vmin(a, b);
            break;
        case ArithmeticOperation::SQUARED_DIFF:
        {
            const vec_type tmp = wrapper::vsub(a, b);
            res                = wrapper::vmul(tmp, tmp);
            break;
        }
        case ArithmeticOperation::PRELU:
        {
            const vec_type zero = wrapper::vdup_n(static_cast<scalar_type>(0), tag_type{});
            const vec_type tmp  = wrapper::vmul(a, b);
            const auto     gt   = wrapper::vcgt(a, zero);

            res = wrapper::vbsl(gt, a, tmp);
            break;
        }

        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return res;
}

template <>
inline int32x4_t elementwise_arithm_op<ArithmeticOperation::DIV, typename wrapper::traits::neon_vector<int32_t, 4>>(const int32x4_t &a, const int32x4_t &b)
{
    return vcvtq_s32_f32(vfloorq_f32(wrapper::vdiv(vcvtq_f32_s32(a), vcvtq_f32_s32(b))));
}

template <>
inline float32x4_t elementwise_arithm_op<ArithmeticOperation::DIV, typename wrapper::traits::neon_vector<float, 4>>(const float32x4_t &a, const float32x4_t &b)
{
    return wrapper::vdiv(a, b);
}

template <>
inline float32x4_t elementwise_arithm_op<ArithmeticOperation::POWER, typename wrapper::traits::neon_vector<float, 4>>(const float32x4_t &a, const float32x4_t &b)
{
    return wrapper::vpow(a, b);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline float16x8_t elementwise_arithm_op<ArithmeticOperation::DIV, typename wrapper::traits::neon_vector<float16_t, 8>>(const float16x8_t &a, const float16x8_t &b)
{
    return wrapper::vdiv(a, b);
}

template <>
inline float16x8_t elementwise_arithm_op<ArithmeticOperation::POWER, typename wrapper::traits::neon_vector<float16_t, 8>>(const float16x8_t &a, const float16x8_t &b)
{
    return wrapper::vpow(a, b);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <ArithmeticOperation op>
inline float32x4x4_t elementwise_arithm_op(const float32x4x4_t &a, const float32x4x4_t &b)
{
    using neon_vector_float = wrapper::traits::neon_vector<float, 4>;
    float32x4x4_t out =
    {
        {
            elementwise_arithm_op<op, neon_vector_float>(a.val[0], b.val[0]),
            elementwise_arithm_op<op, neon_vector_float>(a.val[1], b.val[1]),
            elementwise_arithm_op<op, neon_vector_float>(a.val[2], b.val[2]),
            elementwise_arithm_op<op, neon_vector_float>(a.val[3], b.val[3]),
        }
    };
    return out;
}

template <ArithmeticOperation    op, typename ScalarType, typename VectorType>
inline typename VectorType::type elementwise_arithm_op_broadcast(const typename VectorType::type &a, const ScalarType &broadcast_value, const bool reorder)
{
    using tag_type = typename VectorType::tag_type;
    using vec_type = typename VectorType::type;

    vec_type broadcast_vector = wrapper::vdup_n(broadcast_value, tag_type{});
    return elementwise_arithm_op<op, VectorType>(reorder ? broadcast_vector : a, reorder ? a : broadcast_vector);
}

template <ComparisonOperation op, typename InputScalarType>
inline uint8_t elementwise_comp_op_scalar(const InputScalarType &a, const InputScalarType &b)
{
    bool res = false;

    switch(op)
    {
        case ComparisonOperation::Equal:
            res = (a == b);
            break;
        case ComparisonOperation::NotEqual:
            res = (a != b);
            break;
        case ComparisonOperation::Greater:
            res = (a > b);
            break;
        case ComparisonOperation::GreaterEqual:
            res = (a >= b);
            break;
        case ComparisonOperation::Less:
            res = (a < b);
            break;
        case ComparisonOperation::LessEqual:
            res = (a <= b);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return res ? ~static_cast<uint8_t>(0) : static_cast<uint8_t>(0);
}

template <ComparisonOperation op>
inline uint8_t elementwise_comp_op_quantized_scalar(const float &a, const float &b, UniformQuantizationInfo qinfo)
{
    ARM_COMPUTE_UNUSED(qinfo);
    return elementwise_comp_op_scalar<op>(a, b);
}

template <ComparisonOperation op, typename InputVectorType, typename OutputVectorType>
inline OutputVectorType elementwise_comp_op(const InputVectorType &a, const InputVectorType &b)
{
    OutputVectorType res = { 0, 0, 0, 0 };

    switch(op)
    {
        case ComparisonOperation::Equal:
            res = wrapper::vceq(a, b);
            break;
        case ComparisonOperation::NotEqual:
            res = wrapper::vnot(wrapper::vceq(a, b));
            break;
        case ComparisonOperation::Greater:
            res = wrapper::vcgt(a, b);
            break;
        case ComparisonOperation::GreaterEqual:
            res = wrapper::vcge(a, b);
            break;
        case ComparisonOperation::Less:
            res = wrapper::vcgt(b, a);
            break;
        case ComparisonOperation::LessEqual:
            res = wrapper::vcge(b, a);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return res;
}

template <ComparisonOperation op>
inline uint32x4x4_t elementwise_comp_op(const float32x4x4_t &a, const float32x4x4_t &b)
{
    uint32x4x4_t out =
    {
        {
            elementwise_comp_op<op, float32x4_t, uint32x4_t>(a.val[0], b.val[0]),
            elementwise_comp_op<op, float32x4_t, uint32x4_t>(a.val[1], b.val[1]),
            elementwise_comp_op<op, float32x4_t, uint32x4_t>(a.val[2], b.val[2]),
            elementwise_comp_op<op, float32x4_t, uint32x4_t>(a.val[3], b.val[3])
        }
    };
    return out;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType, typename OutputVectorType>
inline OutputVectorType elementwise_comp_op_broadcast(const InputVectorType &a, const InputScalarType &broadcast_value, const bool reorder)
{
    InputVectorType broadcast_vector = wrapper::vdup_n(broadcast_value, wrapper::traits::vector_128_tag());
    return elementwise_comp_op<op, InputVectorType, OutputVectorType>(reorder ? broadcast_vector : a, reorder ? a : broadcast_vector);
}

template <ArithmeticOperation op, typename ScalarType, typename VectorType>
inline int elementwise_arithm_op_loop(int window_start_x, int window_end_x, int window_step_x,
                                      const ScalarType *input1_ptr, const ScalarType *input2_ptr, ScalarType *output_ptr)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a = wrapper::vloadq(input1_ptr + x);
        const auto b = wrapper::vloadq(input2_ptr + x);
        wrapper::vstore(output_ptr + x, elementwise_arithm_op<op, VectorType>(a, b));
    }
    return x;
}

template <ArithmeticOperation op>
inline int elementwise_arithm_op_quantized_loop(int window_start_x, int window_end_x, int window_step_x,
                                                const uint8_t *input1_ptr, const uint8_t *input2_ptr, uint8_t *output_ptr,
                                                int32x4_t voffset1, int32x4_t voffset2, float32x4_t vscale1, float32x4_t vscale2,
                                                float32x4_t voffseto, float32x4_t invvscaleo)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        // Get inputs and compute output
        const float32x4x4_t af = load_quantized(input1_ptr + x, voffset1, vscale1);
        const float32x4x4_t bf = load_quantized(input2_ptr + x, voffset2, vscale2);
        const float32x4x4_t rf = elementwise_arithm_op<op>(af, bf);
        store_quantized(output_ptr + x, rf, voffseto, invvscaleo);
    }
    return x;
}

template <ArithmeticOperation op>
inline int elementwise_arithm_op_quantized_singed_loop(int window_start_x, int window_end_x, int window_step_x,
                                                       const int8_t *input1_ptr, const int8_t *input2_ptr, int8_t *output_ptr,
                                                       int32x4_t voffset1, int32x4_t voffset2, float32x4_t vscale1, float32x4_t vscale2,
                                                       float32x4_t voffseto, float32x4_t invvscaleo)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        // Get inputs and compute output
        const float32x4x4_t af = load_quantized_signed(input1_ptr + x, voffset1, vscale1);
        const float32x4x4_t bf = load_quantized_signed(input2_ptr + x, voffset2, vscale2);
        const float32x4x4_t rf = elementwise_arithm_op<op>(af, bf);
        store_quantized_signed(output_ptr + x, rf, voffseto, invvscaleo);
    }
    return x;
}

template <ArithmeticOperation op, typename ScalarType, typename VectorType>
inline int elementwise_arithm_op_broadcast_loop(int window_start_x, int window_end_x, int window_step_x,
                                                const ScalarType *non_broadcast_input_ptr, const ScalarType &broadcast_value, ScalarType *output_ptr, const bool reorder)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a = wrapper::vloadq((non_broadcast_input_ptr + x));
        wrapper::vstore(output_ptr + x, elementwise_arithm_op_broadcast<op, ScalarType, VectorType>(a, broadcast_value, reorder));
    }
    return x;
}

template <ArithmeticOperation op>
inline int elementwise_arithm_op_quantized_broadcast_loop(int window_start_x, int window_end_x, int window_step_x,
                                                          const uint8_t *non_broadcast_input_ptr, float32x4x4_t broadcast_vector, uint8_t *output_ptr,
                                                          int32x4_t voffset_non_broadcast, float32x4_t vscale_non_broadcast,
                                                          float32x4_t voffseto, float32x4_t invvscaleo, bool reorder)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const float32x4x4_t af = load_quantized(non_broadcast_input_ptr + x, voffset_non_broadcast, vscale_non_broadcast);
        const float32x4x4_t rf = elementwise_arithm_op<op>(reorder ? broadcast_vector : af, reorder ? af : broadcast_vector);
        store_quantized(output_ptr + x, rf, voffseto, invvscaleo);
    }
    return x;
}
template <ArithmeticOperation op>
inline int elementwise_arithm_op_quantized_signed_broadcast_loop(int window_start_x, int window_end_x, int window_step_x,
                                                                 const int8_t *non_broadcast_input_ptr, float32x4x4_t broadcast_vector, int8_t *output_ptr,
                                                                 int32x4_t voffset_non_broadcast, float32x4_t vscale_non_broadcast,
                                                                 float32x4_t voffseto, float32x4_t invvscaleo, bool reorder)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const float32x4x4_t af = load_quantized_signed(non_broadcast_input_ptr + x, voffset_non_broadcast, vscale_non_broadcast);
        const float32x4x4_t rf = elementwise_arithm_op<op>(reorder ? broadcast_vector : af, reorder ? af : broadcast_vector);
        store_quantized_signed(output_ptr + x, rf, voffseto, invvscaleo);
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_8_loop(int window_start_x, int window_end_x, int window_step_x,
                                      const InputScalarType *input1_ptr, const InputScalarType *input2_ptr, uint8_t *output_ptr)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a   = wrapper::vloadq(input1_ptr + x);
        const auto b   = wrapper::vloadq(input2_ptr + x);
        const auto res = elementwise_comp_op<op, InputVectorType, uint8x16_t>(a, b);
        wrapper::vstore(output_ptr + x, res);
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_16_loop(int window_start_x, int window_end_x, int window_step_x,
                                       const InputScalarType *input1_ptr, const InputScalarType *input2_ptr, uint8_t *output_ptr)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a   = wrapper::vloadq(input1_ptr + x);
        const auto b   = wrapper::vloadq(input2_ptr + x);
        const auto res = elementwise_comp_op<op, InputVectorType, uint16x8_t>(a, b);
        wrapper::vstore(output_ptr + x, wrapper::vmovn(res));
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_32_loop(int window_start_x, int window_end_x, int window_step_x,
                                       const InputScalarType *input1_ptr, const InputScalarType *input2_ptr, uint8_t *output_ptr)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        auto       a    = wrapper::vloadq(input1_ptr + x);
        auto       b    = wrapper::vloadq(input2_ptr + x);
        const auto res  = elementwise_comp_op<op, InputVectorType, uint32x4_t>(a, b);
        a               = wrapper::vloadq(input1_ptr + x + 4);
        b               = wrapper::vloadq(input2_ptr + x + 4);
        const auto res2 = elementwise_comp_op<op, InputVectorType, uint32x4_t>(a, b);
        wrapper::vstore(output_ptr + x, wrapper::vmovn(wrapper::vcombine(wrapper::vmovn(res), wrapper::vmovn(res2))));
    }
    if(x <= window_end_x - 4)
    {
        const auto a   = wrapper::vloadq(input1_ptr + x);
        const auto b   = wrapper::vloadq(input2_ptr + x);
        const auto res = elementwise_comp_op<op, InputVectorType, uint32x4_t>(a, b);
        for(int i = 0; i < 4; i++)
        {
            *(output_ptr + x + i) = wrapper::vgetlane(res, i);
        }
        x = +4;
    }
    return x;
}

template <ComparisonOperation op>
inline int elementwise_comp_op_quantized_loop(int window_start_x, int window_end_x, int window_step_x,
                                              const uint8_t *input1_ptr, const uint8_t *input2_ptr, uint8_t *output_ptr,
                                              int32x4_t voffset1, int32x4_t voffset2, float32x4_t vscale1, float32x4_t vscale2,
                                              float32x4_t voffseto, float32x4_t invvscaleo)
{
    ARM_COMPUTE_UNUSED(voffseto, invvscaleo);
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const float32x4x4_t af = load_quantized(input1_ptr + x, voffset1, vscale1);
        const float32x4x4_t bf = load_quantized(input2_ptr + x, voffset2, vscale2);
        const uint32x4x4_t  rf = elementwise_comp_op<op>(af, bf);
        store_quantized(output_ptr + x, rf);
    }
    return x;
}

template <ComparisonOperation op>
inline int elementwise_comp_op_quantized_signed_loop(int window_start_x, int window_end_x, int window_step_x,
                                                     const int8_t *input1_ptr, const int8_t *input2_ptr, uint8_t *output_ptr,
                                                     int32x4_t voffset1, int32x4_t voffset2, float32x4_t vscale1, float32x4_t vscale2,
                                                     float32x4_t voffseto, float32x4_t invvscaleo)
{
    ARM_COMPUTE_UNUSED(voffseto, invvscaleo);
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const float32x4x4_t af = load_quantized_signed(input1_ptr + x, voffset1, vscale1);
        const float32x4x4_t bf = load_quantized_signed(input2_ptr + x, voffset2, vscale2);
        const uint32x4x4_t  rf = elementwise_comp_op<op>(af, bf);
        store_quantized(output_ptr + x, rf);
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_broadcast_8_loop(int window_start_x, int window_end_x, int window_step_x,
                                                const InputScalarType *non_broadcast_input_ptr, const InputScalarType &broadcast_value, uint8_t *output_ptr, const bool reorder)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a = elementwise_comp_op_broadcast<op, InputScalarType, InputVectorType, uint8x16_t>(wrapper::vloadq((non_broadcast_input_ptr + x)), broadcast_value, reorder);
        wrapper::vstore(output_ptr + x, a);
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_broadcast_16_loop(int window_start_x, int window_end_x, int window_step_x,
                                                 const InputScalarType *non_broadcast_input_ptr, const InputScalarType &broadcast_value, uint8_t *output_ptr, const bool reorder)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a = elementwise_comp_op_broadcast<op, InputScalarType, InputVectorType, uint16x8_t>(wrapper::vloadq((non_broadcast_input_ptr + x)), broadcast_value, reorder);
        wrapper::vstore(output_ptr + x, wrapper::vmovn(a));
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_broadcast_32_loop(int window_start_x, int window_end_x, int window_step_x,
                                                 const InputScalarType *non_broadcast_input_ptr, const InputScalarType &broadcast_value, uint8_t *output_ptr, const bool reorder)
{
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a = elementwise_comp_op_broadcast<op, InputScalarType, InputVectorType, uint32x4_t>(wrapper::vloadq(non_broadcast_input_ptr + x), broadcast_value, reorder);
        const auto b = elementwise_comp_op_broadcast<op, InputScalarType, InputVectorType, uint32x4_t>(wrapper::vloadq(non_broadcast_input_ptr + x + 4), broadcast_value, reorder);
        wrapper::vstore(output_ptr + x, wrapper::vmovn(wrapper::vcombine(wrapper::vmovn(a), wrapper::vmovn(b))));
    }
    if(x <= window_end_x - 4)
    {
        const auto a = elementwise_comp_op_broadcast<op, InputScalarType, InputVectorType, uint32x4_t>(wrapper::vloadq((non_broadcast_input_ptr + x)), broadcast_value, reorder);
        for(int i = 0; i < 4; i++)
        {
            *(output_ptr + x + i) = wrapper::vgetlane(a, i);
        }
        x = +4;
    }
    return x;
}

template <ComparisonOperation op>
inline int elementwise_comp_op_quantized_broadcast_loop(int window_start_x, int window_end_x, int window_step_x,
                                                        const uint8_t *non_broadcast_input_ptr, float32x4x4_t broadcast_vector, uint8_t *output_ptr,
                                                        int32x4_t voffset_non_broadcast, float32x4_t vscale_non_broadcast,
                                                        float32x4_t voffseto, float32x4_t invvscaleo, bool reorder)
{
    ARM_COMPUTE_UNUSED(voffseto, invvscaleo);
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const float32x4x4_t af = load_quantized(non_broadcast_input_ptr + x, voffset_non_broadcast, vscale_non_broadcast);
        const uint32x4x4_t  rf = elementwise_comp_op<op>(reorder ? broadcast_vector : af, reorder ? af : broadcast_vector);
        store_quantized(output_ptr + x, rf);
    }
    return x;
}

template <ComparisonOperation op>
inline int elementwise_comp_op_quantized_signed_broadcast_loop(int window_start_x, int window_end_x, int window_step_x,
                                                               const int8_t *non_broadcast_input_ptr, float32x4x4_t broadcast_vector, uint8_t *output_ptr,
                                                               int32x4_t voffset_non_broadcast, float32x4_t vscale_non_broadcast,
                                                               float32x4_t voffseto, float32x4_t invvscaleo, bool reorder)
{
    ARM_COMPUTE_UNUSED(voffseto, invvscaleo);
    int x = window_start_x;
    for(; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const float32x4x4_t af = load_quantized_signed(non_broadcast_input_ptr + x, voffset_non_broadcast, vscale_non_broadcast);
        const uint32x4x4_t  rf = elementwise_comp_op<op>(reorder ? broadcast_vector : af, reorder ? af : broadcast_vector);
        store_quantized(output_ptr + x, rf);
    }
    return x;
}

template <typename InputScalarType, typename OutputScalarType, typename InputVectorType>
void elementwise_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
                    OutputScalarType (*scalar_func)(const InputScalarType &, const InputScalarType &),
                    int (*broadcast_func)(int, int, int, const InputScalarType *, const InputScalarType &, OutputScalarType *, const bool),
                    int (*neon_func)(int, int, int, const InputScalarType *, const InputScalarType *, OutputScalarType *))
{
    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x         = std::min(16 / static_cast<int>(sizeof(OutputScalarType)), 8);
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = in1->info()->tensor_shape().x() != in2->info()->tensor_shape().x();

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
            auto                  output_ptr              = reinterpret_cast<OutputScalarType *>(output.ptr());
            const auto            non_broadcast_input_ptr = reinterpret_cast<const InputScalarType *>(non_broadcast_input.ptr());
            const InputScalarType broadcast_value         = *reinterpret_cast<const InputScalarType *>(broadcast_input.ptr());

            int x = (*broadcast_func)(window_start_x, window_end_x, window_step_x, non_broadcast_input_ptr, broadcast_value, output_ptr, !is_broadcast_input_2);
            for(; x < window_end_x; ++x)
            {
                const auto a      = *(non_broadcast_input_ptr + x);
                *(output_ptr + x) = (*scalar_func)(!is_broadcast_input_2 ? broadcast_value : a, !is_broadcast_input_2 ? a : broadcast_value);
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
            auto       output_ptr = reinterpret_cast<OutputScalarType *>(output.ptr());
            const auto input1_ptr = reinterpret_cast<const InputScalarType *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const InputScalarType *>(input2.ptr());

            int x = (*neon_func)(window_start_x, window_end_x, window_step_x, input1_ptr, input2_ptr, output_ptr);
            for(; x < window_end_x; ++x)
            {
                const auto a      = *(input1_ptr + x);
                const auto b      = *(input2_ptr + x);
                *(output_ptr + x) = (*scalar_func)(a, b);
            }
        },
        input1, input2, output);
    }
}

void elementwise_op_quantized(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
                              uint8_t (*scalar_func)(const float &, const float &, UniformQuantizationInfo),
                              int (*broadcast_func)(int, int, int, const uint8_t *, float32x4x4_t, uint8_t *, int32x4_t, float32x4_t,
                                                    float32x4_t, float32x4_t, const bool),
                              int (*neon_func)(int, int, int, const uint8_t *, const uint8_t *, uint8_t *,
                                               int32x4_t, int32x4_t, float32x4_t, float32x4_t,
                                               float32x4_t, float32x4_t))
{
    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x         = 16;
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = in1->info()->tensor_shape().x() != in2->info()->tensor_shape().x();

    const UniformQuantizationInfo output_qinfo = out->info()->quantization_info().uniform();

    // Output quantization info (add 0.5 to round toward the nearest integer - 0.5 rounds away from zero)
    const float32x4_t voffseto   = vdupq_n_f32(output_qinfo.offset + 0.5f);
    const float32x4_t invvscaleo = vdupq_n_f32(1.f / output_qinfo.scale);

    if(is_broadcast_across_x)
    {
        // Select the broadcast input on the X axis
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? in2 : in1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;

        const UniformQuantizationInfo broadcast_qinfo     = broadcast_tensor->info()->quantization_info().uniform();
        const UniformQuantizationInfo non_broadcast_qinfo = non_broadcast_tensor->info()->quantization_info().uniform();

        const int32x4_t   voffset_non_broadcast = vdupq_n_s32(non_broadcast_qinfo.offset);
        const float32x4_t vscale_non_broadcast  = vdupq_n_f32(non_broadcast_qinfo.scale);

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const uint8_t *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<uint8_t *>(output.ptr());

            const uint8_t       broadcast_value  = *reinterpret_cast<const uint8_t *>(broadcast_input.ptr());
            const float32x4x4_t broadcast_vector = vdequantize(vdupq_n_u8(broadcast_value), broadcast_qinfo);

            int x = (*broadcast_func)(window_start_x, window_end_x, window_step_x, non_broadcast_input_ptr, broadcast_vector, output_ptr,
                                      voffset_non_broadcast, vscale_non_broadcast, voffseto, invvscaleo, !is_broadcast_input_2);
            for(; x < window_end_x; ++x)
            {
                const float afs   = dequantize_qasymm8(*(non_broadcast_input_ptr + x), non_broadcast_qinfo);
                const float bfs   = dequantize_qasymm8(broadcast_value, broadcast_qinfo);
                *(output_ptr + x) = (*scalar_func)(!is_broadcast_input_2 ? bfs : afs, !is_broadcast_input_2 ? afs : bfs, output_qinfo);
            }
        },
        broadcast_input, non_broadcast_input, output);
    }
    else
    {
        const UniformQuantizationInfo input1_qinfo = in1->info()->quantization_info().uniform();
        const UniformQuantizationInfo input2_qinfo = in2->info()->quantization_info().uniform();

        // Input1 quantization info
        const int32x4_t   voffset1 = vdupq_n_s32(input1_qinfo.offset);
        const float32x4_t vscale1  = vdupq_n_f32(input1_qinfo.scale);

        // Input2 quantization info
        const int32x4_t   voffset2 = vdupq_n_s32(input2_qinfo.offset);
        const float32x4_t vscale2  = vdupq_n_f32(input2_qinfo.scale);

        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(in1, input1_win);
        Iterator input2(in2, input2_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const uint8_t *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

            int x = (*neon_func)(window_start_x, window_end_x, window_step_x, input1_ptr, input2_ptr, output_ptr, voffset1, voffset2,
                                 vscale1, vscale2, voffseto, invvscaleo);
            for(; x < window_end_x; ++x)
            {
                const float afs   = dequantize_qasymm8(*(input1_ptr + x), input1_qinfo);
                const float bfs   = dequantize_qasymm8(*(input2_ptr + x), input2_qinfo);
                *(output_ptr + x) = (*scalar_func)(afs, bfs, output_qinfo);
            }
        },
        input1, input2, output);
    }
}

void elementwise_comp_quantized_signed(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
                                       uint8_t (*scalar_func)(const float &, const float &, UniformQuantizationInfo),
                                       int (*broadcast_func)(int, int, int, const int8_t *, float32x4x4_t, uint8_t *, int32x4_t, float32x4_t,
                                                             float32x4_t, float32x4_t, const bool),
                                       int (*neon_func)(int, int, int, const int8_t *, const int8_t *, uint8_t *,
                                                        int32x4_t, int32x4_t, float32x4_t, float32x4_t,
                                                        float32x4_t, float32x4_t))
{
    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x         = 16;
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = in1->info()->tensor_shape().x() != in2->info()->tensor_shape().x();

    const UniformQuantizationInfo output_qinfo = out->info()->quantization_info().uniform();

    const float32x4_t voffseto   = vdupq_n_f32(output_qinfo.offset);
    const float32x4_t invvscaleo = vdupq_n_f32(1.f / output_qinfo.scale);

    if(is_broadcast_across_x)
    {
        // Select the broadcast input on the X axis
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? in2 : in1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;

        const UniformQuantizationInfo broadcast_qinfo     = broadcast_tensor->info()->quantization_info().uniform();
        const UniformQuantizationInfo non_broadcast_qinfo = non_broadcast_tensor->info()->quantization_info().uniform();

        const int32x4_t   voffset_non_broadcast = vdupq_n_s32(non_broadcast_qinfo.offset);
        const float32x4_t vscale_non_broadcast  = vdupq_n_f32(non_broadcast_qinfo.scale);

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const int8_t *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<uint8_t *>(output.ptr());

            const int8_t        broadcast_value  = *reinterpret_cast<const int8_t *>(broadcast_input.ptr());
            const float32x4x4_t broadcast_vector = vdequantize(vdupq_n_s8(broadcast_value), broadcast_qinfo);

            int x = (*broadcast_func)(window_start_x, window_end_x, window_step_x, non_broadcast_input_ptr, broadcast_vector, output_ptr,
                                      voffset_non_broadcast, vscale_non_broadcast, voffseto, invvscaleo, !is_broadcast_input_2);
            for(; x < window_end_x; ++x)
            {
                const float afs   = dequantize_qasymm8_signed(*(non_broadcast_input_ptr + x), non_broadcast_qinfo);
                const float bfs   = dequantize_qasymm8_signed(broadcast_value, broadcast_qinfo);
                *(output_ptr + x) = (*scalar_func)(!is_broadcast_input_2 ? bfs : afs, !is_broadcast_input_2 ? afs : bfs, output_qinfo);
            }
        },
        broadcast_input, non_broadcast_input, output);
    }
    else
    {
        const UniformQuantizationInfo input1_qinfo = in1->info()->quantization_info().uniform();
        const UniformQuantizationInfo input2_qinfo = in2->info()->quantization_info().uniform();

        // Input1 quantization info
        const int32x4_t   voffset1 = vdupq_n_s32(input1_qinfo.offset);
        const float32x4_t vscale1  = vdupq_n_f32(input1_qinfo.scale);

        // Input2 quantization info
        const int32x4_t   voffset2 = vdupq_n_s32(input2_qinfo.offset);
        const float32x4_t vscale2  = vdupq_n_f32(input2_qinfo.scale);

        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(in1, input1_win);
        Iterator input2(in2, input2_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const int8_t *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const int8_t *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

            int x = (*neon_func)(window_start_x, window_end_x, window_step_x, input1_ptr, input2_ptr, output_ptr, voffset1, voffset2,
                                 vscale1, vscale2, voffseto, invvscaleo);
            for(; x < window_end_x; ++x)
            {
                const float afs   = dequantize_qasymm8_signed(*(input1_ptr + x), input1_qinfo);
                const float bfs   = dequantize_qasymm8_signed(*(input2_ptr + x), input2_qinfo);
                *(output_ptr + x) = (*scalar_func)(afs, bfs, output_qinfo);
            }
        },
        input1, input2, output);
    }
}

void elementwise_op_quantized_signed(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
                                     int8_t (*scalar_func)(const float &, const float &, UniformQuantizationInfo),
                                     int (*broadcast_func)(int, int, int, const int8_t *, float32x4x4_t, int8_t *, int32x4_t, float32x4_t,
                                                           float32x4_t, float32x4_t, const bool),
                                     int (*neon_func)(int, int, int, const int8_t *, const int8_t *, int8_t *,
                                                      int32x4_t, int32x4_t, float32x4_t, float32x4_t,
                                                      float32x4_t, float32x4_t))
{
    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x         = 16;
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = in1->info()->tensor_shape().x() != in2->info()->tensor_shape().x();

    const UniformQuantizationInfo output_qinfo = out->info()->quantization_info().uniform();

    const float32x4_t voffseto   = vdupq_n_f32(output_qinfo.offset);
    const float32x4_t invvscaleo = vdupq_n_f32(1.f / output_qinfo.scale);

    if(is_broadcast_across_x)
    {
        // Select the broadcast input on the X axis
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? in2 : in1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;

        const UniformQuantizationInfo broadcast_qinfo     = broadcast_tensor->info()->quantization_info().uniform();
        const UniformQuantizationInfo non_broadcast_qinfo = non_broadcast_tensor->info()->quantization_info().uniform();

        const int32x4_t   voffset_non_broadcast = vdupq_n_s32(non_broadcast_qinfo.offset);
        const float32x4_t vscale_non_broadcast  = vdupq_n_f32(non_broadcast_qinfo.scale);

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const int8_t *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<int8_t *>(output.ptr());

            const int8_t        broadcast_value  = *reinterpret_cast<const int8_t *>(broadcast_input.ptr());
            const float32x4x4_t broadcast_vector = vdequantize(vdupq_n_s8(broadcast_value), broadcast_qinfo);

            int x = (*broadcast_func)(window_start_x, window_end_x, window_step_x, non_broadcast_input_ptr, broadcast_vector, output_ptr,
                                      voffset_non_broadcast, vscale_non_broadcast, voffseto, invvscaleo, !is_broadcast_input_2);
            for(; x < window_end_x; ++x)
            {
                const float afs   = dequantize_qasymm8_signed(*(non_broadcast_input_ptr + x), non_broadcast_qinfo);
                const float bfs   = dequantize_qasymm8_signed(broadcast_value, broadcast_qinfo);
                *(output_ptr + x) = (*scalar_func)(!is_broadcast_input_2 ? bfs : afs, !is_broadcast_input_2 ? afs : bfs, output_qinfo);
            }
        },
        broadcast_input, non_broadcast_input, output);
    }
    else
    {
        const UniformQuantizationInfo input1_qinfo = in1->info()->quantization_info().uniform();
        const UniformQuantizationInfo input2_qinfo = in2->info()->quantization_info().uniform();

        // Input1 quantization info
        const int32x4_t   voffset1 = vdupq_n_s32(input1_qinfo.offset);
        const float32x4_t vscale1  = vdupq_n_f32(input1_qinfo.scale);

        // Input2 quantization info
        const int32x4_t   voffset2 = vdupq_n_s32(input2_qinfo.offset);
        const float32x4_t vscale2  = vdupq_n_f32(input2_qinfo.scale);

        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(in1, input1_win);
        Iterator input2(in2, input2_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const int8_t *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const int8_t *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

            int x = (*neon_func)(window_start_x, window_end_x, window_step_x, input1_ptr, input2_ptr, output_ptr, voffset1, voffset2,
                                 vscale1, vscale2, voffseto, invvscaleo);
            for(; x < window_end_x; ++x)
            {
                const float afs   = dequantize_qasymm8_signed(*(input1_ptr + x), input1_qinfo);
                const float bfs   = dequantize_qasymm8_signed(*(input2_ptr + x), input2_qinfo);
                *(output_ptr + x) = (*scalar_func)(afs, bfs, output_qinfo);
            }
        },
        input1, input2, output);
    }
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
void elementwise_comp_op_8(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op<InputScalarType, uint8_t, InputVectorType>(in1, in2, out, window,
                                                              &elementwise_comp_op_scalar<op, InputScalarType>,
                                                              &elementwise_comp_op_broadcast_8_loop<op, InputScalarType, InputVectorType>,
                                                              &elementwise_comp_op_8_loop<op, InputScalarType, InputVectorType>);
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
void elementwise_comp_op_16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op<InputScalarType, uint8_t, InputVectorType>(in1, in2, out, window,
                                                              &elementwise_comp_op_scalar<op, InputScalarType>,
                                                              &elementwise_comp_op_broadcast_16_loop<op, InputScalarType, InputVectorType>,
                                                              &elementwise_comp_op_16_loop<op, InputScalarType, InputVectorType>);
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
void elementwise_comp_op_32(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op<InputScalarType, uint8_t, InputVectorType>(in1, in2, out, window,
                                                              &elementwise_comp_op_scalar<op, InputScalarType>,
                                                              &elementwise_comp_op_broadcast_32_loop<op, InputScalarType, InputVectorType>,
                                                              &elementwise_comp_op_32_loop<op, InputScalarType, InputVectorType>);
}

template <ArithmeticOperation op, typename VectorType>
void elementwise_arithm_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    using scalar_type = typename VectorType::scalar_type;

    elementwise_op<scalar_type, scalar_type, VectorType>(in1, in2, out, window,
                                                         &elementwise_arithm_op_scalar<op, scalar_type>,
                                                         &elementwise_arithm_op_broadcast_loop<op, scalar_type, VectorType>,
                                                         &elementwise_arithm_op_loop<op, scalar_type, VectorType>);
}

template <ArithmeticOperation op>
void elementwise_arithm_op_quantized(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op_quantized(in1, in2, out, window, &elementwise_arithm_op_quantized_scalar<op>,
                             &elementwise_arithm_op_quantized_broadcast_loop<op>,
                             &elementwise_arithm_op_quantized_loop<op>);
}
template <ArithmeticOperation op>
void elementwise_arithm_op_quantized_signed(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op_quantized_signed(in1, in2, out, window, &elementwise_arithm_op_quantized_signed_scalar<op>,
                                    &elementwise_arithm_op_quantized_signed_broadcast_loop<op>,
                                    &elementwise_arithm_op_quantized_singed_loop<op>);
}

template <ComparisonOperation op>
void elementwise_comp_op_quantized(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op_quantized(in1, in2, out, window, &elementwise_comp_op_quantized_scalar<op>,
                             &elementwise_comp_op_quantized_broadcast_loop<op>,
                             &elementwise_comp_op_quantized_loop<op>);
}

template <ComparisonOperation op>
void elementwise_comp_op_quantized_signed(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_comp_quantized_signed(in1, in2, out, window, &elementwise_comp_op_quantized_scalar<op>,
                                      &elementwise_comp_op_quantized_signed_broadcast_loop<op>,
                                      &elementwise_comp_op_quantized_signed_loop<op>);
}

std::function<void(const ITensor *, const ITensor *, ITensor *, const Window &)>
configure_func(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output,
               std::map<std::string, NEElementwiseOperationKernel::ElementwiseFunction *> map_function)
{
    std::string function_to_call("op_");
    function_to_call += string_from_data_type(input1->data_type()) + "_";
    function_to_call += string_from_data_type(input2->data_type()) + "_";
    function_to_call += string_from_data_type(output->data_type());

    auto it = map_function.find(function_to_call);

    if(it != map_function.end())
    {
        auto func = it->second;
        return [func](const ITensor * input1, const ITensor * input2, ITensor * output, const Window & window)
        {
            func(input1, input2, output, window);
        };
    }
    return nullptr;
}

template <ArithmeticOperation op>
std::function<void(const ITensor *, const ITensor *, ITensor *, const Window &)>
configure_arithm_func(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    static std::map<std::string, NEElementwiseOperationKernel::ElementwiseFunction *> map_function =
    {
        { "op_F32_F32_F32", &elementwise_arithm_op<op, typename wrapper::traits::neon_vector<float, 4>> },
        { "op_S16_S16_S16", &elementwise_arithm_op<op, typename wrapper::traits::neon_vector<int16_t, 8>> },
        { "op_S32_S32_S32", &elementwise_arithm_op<op, typename wrapper::traits::neon_vector<int32_t, 4>> },
        { "op_QASYMM8_QASYMM8_QASYMM8", &elementwise_arithm_op_quantized<op> },
        { "op_QASYMM8_SIGNED_QASYMM8_SIGNED_QASYMM8_SIGNED", &elementwise_arithm_op_quantized_signed<op> }
    };
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    map_function["op_F16_F16_F16"] = &elementwise_arithm_op<op, typename wrapper::traits::neon_vector<float16_t, 8>>;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

    return configure_func(input1, input2, output, map_function);
}

template <ComparisonOperation op>
std::function<void(const ITensor *input1, const ITensor *input2, ITensor *output, const Window &window)>
configure_comp_func(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    static std::map<std::string, NEElementwiseOperationKernel::ElementwiseFunction *> map_function =
    {
        { "op_U8_U8_U8", &elementwise_comp_op_8<op, uint8_t, uint8x16_t> },
        { "op_F32_F32_U8", &elementwise_comp_op_32<op, float, float32x4_t> },
        { "op_S16_S16_U8", &elementwise_comp_op_16<op, int16_t, int16x8_t> },
        { "op_S32_S32_U8", &elementwise_comp_op_32<op, int32_t, int32x4_t> },
        { "op_QASYMM8_SIGNED_QASYMM8_SIGNED_U8", &elementwise_comp_op_quantized_signed<op> },
        { "op_QASYMM8_QASYMM8_U8", &elementwise_comp_op_quantized<op> }
    };
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    map_function["op_F16_F16_U8"] = &elementwise_comp_op_16<op, float16_t, float16x8_t>;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

    return configure_func(input1, input2, output, map_function);
}
} // namespace

NEElementwiseOperationKernel::NEElementwiseOperationKernel()
    : _function(nullptr), _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

Status NEElementwiseOperationKernel::validate_arguments_common(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input1);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &input2);

    const TensorShape out_shape = TensorShape::broadcast_shape(input1.tensor_shape(), input2.tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
                                        "Wrong shape for output");
    }

    return Status{};
}

void NEElementwiseOperationKernel::configure_common(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    // Configure kernel window
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*input1, *input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    // Auto initialize output if not initialized
    auto_init_if_empty(*output, out_shape, 1, input1->data_type());

    Window win = calculate_max_window(valid_region);

    INEKernel::configure(win);
}

void NEElementwiseOperationKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info, window);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_function == nullptr);
    _function(tensors.get_const_tensor(TensorType::ACL_SRC_0),
              tensors.get_const_tensor(TensorType::ACL_SRC_1),
              tensors.get_tensor(TensorType::ACL_DST), window);
}

/** Arithmetic operators (min, max, squared_diff) */
void NEArithmeticOperationKernel::configure(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1, *input2, *output));
    configure_common(input1, input2, output);
    switch(op)
    {
        case ArithmeticOperation::MAX:
            _function = configure_arithm_func<ArithmeticOperation::MAX>(input1, input2, output);
            break;
        case ArithmeticOperation::MIN:
            _function = configure_arithm_func<ArithmeticOperation::MIN>(input1, input2, output);
            break;
        case ArithmeticOperation::SQUARED_DIFF:
            _function = configure_arithm_func<ArithmeticOperation::SQUARED_DIFF>(input1, input2, output);
            break;
        case ArithmeticOperation::PRELU:
            _function = configure_arithm_func<ArithmeticOperation::PRELU>(input1, input2, output);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
}

Status NEArithmeticOperationKernel::validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S16, DataType::F16, DataType::S32, DataType::F32);
    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &output);
    }
    return validate_arguments_common(input1, input2, output);
}

Status NEArithmeticOperationKernel::validate(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(op);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output));
    return Status{};
}

/** The division operator */

void NEDivisionOperationKernel::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1, *input2, *output));
    configure_common(input1, input2, output);
    _function = configure_arithm_func<ArithmeticOperation::DIV>(input1, input2, output);
}

Status NEDivisionOperationKernel::validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::S32, DataType::F16, DataType::F32);
    return NEArithmeticOperationKernel::validate_arguments(input1, input2, output);
}

Status NEDivisionOperationKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output));
    return Status{};
}

/** The power operator */
void NEPowerOperationKernel::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1, *input2, *output));
    configure_common(input1, input2, output);
    _function = configure_arithm_func<ArithmeticOperation::POWER>(input1, input2, output);
}

Status NEPowerOperationKernel::validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::F16, DataType::F32);
    return NEArithmeticOperationKernel::validate_arguments(input1, input2, output);
}

Status NEPowerOperationKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output));
    return Status{};
}

/** Comparison operators (equal, not equal, less than, greater than, less than or equal, greater than or equal) */
void NEComparisonOperationKernel::configure(ComparisonOperation op, const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1, *input2, *output));
    configure_common(input1, input2, output);
    switch(op)
    {
        case ComparisonOperation::Equal:
            _function = configure_comp_func<ComparisonOperation::Equal>(input1, input2, output);
            break;
        case ComparisonOperation::NotEqual:
            _function = configure_comp_func<ComparisonOperation::NotEqual>(input1, input2, output);
            break;
        case ComparisonOperation::Greater:
            _function = configure_comp_func<ComparisonOperation::Greater>(input1, input2, output);
            break;
        case ComparisonOperation::GreaterEqual:
            _function = configure_comp_func<ComparisonOperation::GreaterEqual>(input1, input2, output);
            break;
        case ComparisonOperation::Less:
            _function = configure_comp_func<ComparisonOperation::Less>(input1, input2, output);
            break;
        case ComparisonOperation::LessEqual:
            _function = configure_comp_func<ComparisonOperation::LessEqual>(input1, input2, output);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
}

Status NEComparisonOperationKernel::validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S16, DataType::F16, DataType::S32, DataType::F32);
    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output, 1, DataType::U8);
    }
    return validate_arguments_common(input1, input2, output);
}

Status NEComparisonOperationKernel::validate(ComparisonOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(op);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output));
    return Status{};
}
} // namespace arm_compute
