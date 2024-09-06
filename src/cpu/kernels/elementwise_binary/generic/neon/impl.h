/*
 * Copyright (c) 2021-2022, 2024 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_ELEMENTWISE_BINARY_GENERIC_NEON_IMPL_H
#define ACL_SRC_CPU_KERNELS_ELEMENTWISE_BINARY_GENERIC_NEON_IMPL_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

#include "src/core/NEON/NEAsymm.h"

namespace arm_compute
{
namespace cpu
{
template <ArithmeticOperation op, typename VectorType>
typename VectorType::type elementwise_arithm_op(const typename VectorType::type &a, const typename VectorType::type &b)
{
    using vec_type    = typename VectorType::type;
    using scalar_type = typename VectorType::scalar_type;
    using tag_type    = typename VectorType::tag_type;

    vec_type res = wrapper::vdup_n(static_cast<scalar_type>(0), tag_type{});

    switch (op)
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

template <ArithmeticOperation op, typename ScalarType, typename VectorType>
typename VectorType::type elementwise_arithm_op_broadcast(const typename VectorType::type &a,
                                                          const ScalarType                &broadcast_value,
                                                          const bool                       reorder)
{
    using tag_type = typename VectorType::tag_type;
    using vec_type = typename VectorType::type;

    vec_type broadcast_vector = wrapper::vdup_n(broadcast_value, tag_type{});
    return elementwise_arithm_op<op, VectorType>(reorder ? broadcast_vector : a, reorder ? a : broadcast_vector);
}

template <typename InputScalarType, typename OutputScalarType, typename InputVectorType>
void elementwise_op(
    const ITensor *in1,
    const ITensor *in2,
    ITensor       *out,
    const Window  &window,
    OutputScalarType (*scalar_func)(const InputScalarType &, const InputScalarType &),
    int (*broadcast_func)(
        int, int, int, const InputScalarType *, const InputScalarType &, OutputScalarType *, const bool),
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

    if (is_broadcast_across_x)
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

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                auto       output_ptr = reinterpret_cast<OutputScalarType *>(output.ptr());
                const auto non_broadcast_input_ptr =
                    reinterpret_cast<const InputScalarType *>(non_broadcast_input.ptr());
                const InputScalarType broadcast_value =
                    *reinterpret_cast<const InputScalarType *>(broadcast_input.ptr());

                int x = (*broadcast_func)(window_start_x, window_end_x, window_step_x, non_broadcast_input_ptr,
                                          broadcast_value, output_ptr, !is_broadcast_input_2);
                for (; x < window_end_x; ++x)
                {
                    const auto a      = *(non_broadcast_input_ptr + x);
                    *(output_ptr + x) = (*scalar_func)(!is_broadcast_input_2 ? broadcast_value : a,
                                                       !is_broadcast_input_2 ? a : broadcast_value);
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

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                auto       output_ptr = reinterpret_cast<OutputScalarType *>(output.ptr());
                const auto input1_ptr = reinterpret_cast<const InputScalarType *>(input1.ptr());
                const auto input2_ptr = reinterpret_cast<const InputScalarType *>(input2.ptr());

                int x = (*neon_func)(window_start_x, window_end_x, window_step_x, input1_ptr, input2_ptr, output_ptr);
                for (; x < window_end_x; ++x)
                {
                    const auto a      = *(input1_ptr + x);
                    const auto b      = *(input2_ptr + x);
                    *(output_ptr + x) = (*scalar_func)(a, b);
                }
            },
            input1, input2, output);
    }
}

template <ArithmeticOperation op, typename ScalarType>
inline ScalarType elementwise_arithm_op_scalar(const ScalarType &a, const ScalarType &b)
{
    auto res = ScalarType(0);

    switch (op)
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

template <>
inline int32x4_t
elementwise_arithm_op<ArithmeticOperation::DIV, typename wrapper::traits::neon_vector<int32_t, 4>>(const int32x4_t &a,
                                                                                                   const int32x4_t &b)
{
    int32x4_t result;

    // Neon(TM) does not have vector integer division
    result[0] = a[0] / b[0];
    result[1] = a[1] / b[1];
    result[2] = a[2] / b[2];
    result[3] = a[3] / b[3];

    return result;
}

template <>
inline float32x4_t
elementwise_arithm_op<ArithmeticOperation::DIV, typename wrapper::traits::neon_vector<float, 4>>(const float32x4_t &a,
                                                                                                 const float32x4_t &b)
{
    return wrapper::vdiv(a, b);
}

template <>
inline float32x4_t
elementwise_arithm_op<ArithmeticOperation::POWER, typename wrapper::traits::neon_vector<float, 4>>(const float32x4_t &a,
                                                                                                   const float32x4_t &b)
{
    return wrapper::vpow(a, b);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline float16x8_t elementwise_arithm_op<ArithmeticOperation::DIV, typename wrapper::traits::neon_vector<float16_t, 8>>(
    const float16x8_t &a, const float16x8_t &b)
{
    return wrapper::vdiv(a, b);
}

template <>
inline float16x8_t
elementwise_arithm_op<ArithmeticOperation::POWER, typename wrapper::traits::neon_vector<float16_t, 8>>(
    const float16x8_t &a, const float16x8_t &b)
{
    return wrapper::vpow(a, b);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <ArithmeticOperation op, typename ScalarType, typename VectorType>
inline int elementwise_arithm_op_loop(int               window_start_x,
                                      int               window_end_x,
                                      int               window_step_x,
                                      const ScalarType *input1_ptr,
                                      const ScalarType *input2_ptr,
                                      ScalarType       *output_ptr)
{
    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a = wrapper::vloadq(input1_ptr + x);
        const auto b = wrapper::vloadq(input2_ptr + x);
        wrapper::vstore(output_ptr + x, elementwise_arithm_op<op, VectorType>(a, b));
    }
    return x;
}

template <ArithmeticOperation op, typename ScalarType, typename VectorType>
inline int elementwise_arithm_op_broadcast_loop(int               window_start_x,
                                                int               window_end_x,
                                                int               window_step_x,
                                                const ScalarType *non_broadcast_input_ptr,
                                                const ScalarType &broadcast_value,
                                                ScalarType       *output_ptr,
                                                const bool        reorder)
{
    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a = wrapper::vloadq((non_broadcast_input_ptr + x));
        wrapper::vstore(output_ptr + x,
                        elementwise_arithm_op_broadcast<op, ScalarType, VectorType>(a, broadcast_value, reorder));
    }
    return x;
}

template <ArithmeticOperation op, typename VectorType>
void elementwise_arithm_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    using scalar_type = typename VectorType::scalar_type;

    elementwise_op<scalar_type, scalar_type, VectorType>(
        in1, in2, out, window, &elementwise_arithm_op_scalar<op, scalar_type>,
        &elementwise_arithm_op_broadcast_loop<op, scalar_type, VectorType>,
        &elementwise_arithm_op_loop<op, scalar_type, VectorType>);
}

template <ComparisonOperation op, typename InputScalarType>
inline uint8_t elementwise_comp_op_scalar(const InputScalarType &a, const InputScalarType &b)
{
    bool res = false;

    switch (op)
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

template <ComparisonOperation op, typename InputVectorType, typename OutputVectorType>
inline OutputVectorType elementwise_comp_op(const InputVectorType &a, const InputVectorType &b)
{
    OutputVectorType res = {0, 0, 0, 0};

    switch (op)
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

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType, typename OutputVectorType>
inline OutputVectorType
elementwise_comp_op_broadcast(const InputVectorType &a, const InputScalarType &broadcast_value, const bool reorder)
{
    InputVectorType broadcast_vector = wrapper::vdup_n(broadcast_value, wrapper::traits::vector_128_tag());
    return elementwise_comp_op<op, InputVectorType, OutputVectorType>(reorder ? broadcast_vector : a,
                                                                      reorder ? a : broadcast_vector);
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_broadcast_8_loop(int                    window_start_x,
                                                int                    window_end_x,
                                                int                    window_step_x,
                                                const InputScalarType *non_broadcast_input_ptr,
                                                const InputScalarType &broadcast_value,
                                                uint8_t               *output_ptr,
                                                const bool             reorder)
{
    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a = elementwise_comp_op_broadcast<op, InputScalarType, InputVectorType, uint8x16_t>(
            wrapper::vloadq((non_broadcast_input_ptr + x)), broadcast_value, reorder);
        wrapper::vstore(output_ptr + x, a);
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_broadcast_16_loop(int                    window_start_x,
                                                 int                    window_end_x,
                                                 int                    window_step_x,
                                                 const InputScalarType *non_broadcast_input_ptr,
                                                 const InputScalarType &broadcast_value,
                                                 uint8_t               *output_ptr,
                                                 const bool             reorder)
{
    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a = elementwise_comp_op_broadcast<op, InputScalarType, InputVectorType, uint16x8_t>(
            wrapper::vloadq((non_broadcast_input_ptr + x)), broadcast_value, reorder);
        wrapper::vstore(output_ptr + x, wrapper::vmovn(a));
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_broadcast_32_loop(int                    window_start_x,
                                                 int                    window_end_x,
                                                 int                    window_step_x,
                                                 const InputScalarType *non_broadcast_input_ptr,
                                                 const InputScalarType &broadcast_value,
                                                 uint8_t               *output_ptr,
                                                 const bool             reorder)
{
    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a = elementwise_comp_op_broadcast<op, InputScalarType, InputVectorType, uint32x4_t>(
            wrapper::vloadq(non_broadcast_input_ptr + x), broadcast_value, reorder);
        const auto b = elementwise_comp_op_broadcast<op, InputScalarType, InputVectorType, uint32x4_t>(
            wrapper::vloadq(non_broadcast_input_ptr + x + 4), broadcast_value, reorder);
        wrapper::vstore(output_ptr + x, wrapper::vmovn(wrapper::vcombine(wrapper::vmovn(a), wrapper::vmovn(b))));
    }
    if (x <= window_end_x - 4)
    {
        const auto a = elementwise_comp_op_broadcast<op, InputScalarType, InputVectorType, uint32x4_t>(
            wrapper::vloadq((non_broadcast_input_ptr + x)), broadcast_value, reorder);
        for (int i = 0; i < 4; i++)
        {
            *(output_ptr + x + i) = wrapper::vgetlane(a, i);
        }
        x = +4;
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_8_loop(int                    window_start_x,
                                      int                    window_end_x,
                                      int                    window_step_x,
                                      const InputScalarType *input1_ptr,
                                      const InputScalarType *input2_ptr,
                                      uint8_t               *output_ptr)
{
    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a   = wrapper::vloadq(input1_ptr + x);
        const auto b   = wrapper::vloadq(input2_ptr + x);
        const auto res = elementwise_comp_op<op, InputVectorType, uint8x16_t>(a, b);
        wrapper::vstore(output_ptr + x, res);
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_16_loop(int                    window_start_x,
                                       int                    window_end_x,
                                       int                    window_step_x,
                                       const InputScalarType *input1_ptr,
                                       const InputScalarType *input2_ptr,
                                       uint8_t               *output_ptr)
{
    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const auto a   = wrapper::vloadq(input1_ptr + x);
        const auto b   = wrapper::vloadq(input2_ptr + x);
        const auto res = elementwise_comp_op<op, InputVectorType, uint16x8_t>(a, b);
        wrapper::vstore(output_ptr + x, wrapper::vmovn(res));
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
inline int elementwise_comp_op_32_loop(int                    window_start_x,
                                       int                    window_end_x,
                                       int                    window_step_x,
                                       const InputScalarType *input1_ptr,
                                       const InputScalarType *input2_ptr,
                                       uint8_t               *output_ptr)
{
    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        auto       a    = wrapper::vloadq(input1_ptr + x);
        auto       b    = wrapper::vloadq(input2_ptr + x);
        const auto res  = elementwise_comp_op<op, InputVectorType, uint32x4_t>(a, b);
        a               = wrapper::vloadq(input1_ptr + x + 4);
        b               = wrapper::vloadq(input2_ptr + x + 4);
        const auto res2 = elementwise_comp_op<op, InputVectorType, uint32x4_t>(a, b);
        wrapper::vstore(output_ptr + x, wrapper::vmovn(wrapper::vcombine(wrapper::vmovn(res), wrapper::vmovn(res2))));
    }
    if (x <= window_end_x - 4)
    {
        const auto a   = wrapper::vloadq(input1_ptr + x);
        const auto b   = wrapper::vloadq(input2_ptr + x);
        const auto res = elementwise_comp_op<op, InputVectorType, uint32x4_t>(a, b);
        for (int i = 0; i < 4; i++)
        {
            *(output_ptr + x + i) = wrapper::vgetlane(res, i);
        }
        x = +4;
    }
    return x;
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
void elementwise_comp_op_8(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op<InputScalarType, uint8_t, InputVectorType>(
        in1, in2, out, window, &elementwise_comp_op_scalar<op, InputScalarType>,
        &elementwise_comp_op_broadcast_8_loop<op, InputScalarType, InputVectorType>,
        &elementwise_comp_op_8_loop<op, InputScalarType, InputVectorType>);
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
void elementwise_comp_op_16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op<InputScalarType, uint8_t, InputVectorType>(
        in1, in2, out, window, &elementwise_comp_op_scalar<op, InputScalarType>,
        &elementwise_comp_op_broadcast_16_loop<op, InputScalarType, InputVectorType>,
        &elementwise_comp_op_16_loop<op, InputScalarType, InputVectorType>);
}

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType>
void elementwise_comp_op_32(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op<InputScalarType, uint8_t, InputVectorType>(
        in1, in2, out, window, &elementwise_comp_op_scalar<op, InputScalarType>,
        &elementwise_comp_op_broadcast_32_loop<op, InputScalarType, InputVectorType>,
        &elementwise_comp_op_32_loop<op, InputScalarType, InputVectorType>);
}

inline float32x4x4_t load_quantized(const uint8_t *input1_ptr, const int32x4_t &offset, const float32x4_t &scale)
{
    qasymm8x16_t        x   = vld1q_u8(input1_ptr);
    const float32x4x4_t out = {{
        vmulq_f32(
            vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(x))))), offset)),
            scale),
        vmulq_f32(
            vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(x))))), offset)),
            scale),
        vmulq_f32(
            vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(x))))), offset)),
            scale),
        vmulq_f32(vcvtq_f32_s32(
                      vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(x))))), offset)),
                  scale),
    }};
    return out;
}

inline float32x4x4_t load_quantized(const int8_t *input1_ptr, const int32x4_t &offset, const float32x4_t &scale)
{
    qasymm8x16_signed_t x   = vld1q_s8(input1_ptr);
    const float32x4x4_t out = {{
        vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(x)))), offset)), scale),
        vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(x)))), offset)), scale),
        vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(x)))), offset)), scale),
        vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(x)))), offset)), scale),
    }};
    return out;
}

inline void store_quantized(uint8_t *output_ptr, const uint32x4x4_t &out)
{
    const uint8x8_t pa = vqmovn_u16(vcombine_u16(vqmovn_u32(out.val[0]), vqmovn_u32(out.val[1])));
    const uint8x8_t pb = vqmovn_u16(vcombine_u16(vqmovn_u32(out.val[2]), vqmovn_u32(out.val[3])));
    vst1q_u8(output_ptr, vcombine_u8(pa, pb));
}

inline void store_quantized(uint8_t *output_ptr, const int32x4x4_t &out)
{
    const uint8x8_t pa = vqmovun_s16(vcombine_s16(vqmovn_s32(out.val[0]), vqmovn_s32(out.val[1])));
    const uint8x8_t pb = vqmovun_s16(vcombine_s16(vqmovn_s32(out.val[2]), vqmovn_s32(out.val[3])));
    vst1q_u8(output_ptr, vcombine_u8(pa, pb));
}

inline void
store_quantized(uint8_t *output_ptr, const float32x4x4_t &rf, const float32x4_t &offset, const float32x4x4_t &invscale)
{
    // Adjust offset with 0.5 to round to nearest.
    const float32x4_t adj_offset = vaddq_f32(offset, vdupq_n_f32(0.5f));

    const int32x4x4_t out = {{
        vcvtq_s32_f32(vmlaq_f32(adj_offset, rf.val[0], invscale.val[0])),
        vcvtq_s32_f32(vmlaq_f32(adj_offset, rf.val[1], invscale.val[1])),
        vcvtq_s32_f32(vmlaq_f32(adj_offset, rf.val[2], invscale.val[2])),
        vcvtq_s32_f32(vmlaq_f32(adj_offset, rf.val[3], invscale.val[3])),
    }};
    store_quantized(output_ptr, out);
}

inline void
store_quantized(uint8_t *output_ptr, const float32x4x4_t &rf, const float32x4_t &offset, const float32x4_t &invscale)
{
    return store_quantized(output_ptr, rf, offset,
                           float32x4x4_t{{
                               invscale,
                               invscale,
                               invscale,
                               invscale,
                           }});
}

inline void store_quantized(int8_t *output_ptr, const int32x4x4_t &out)
{
    const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(out.val[0]), vqmovn_s32(out.val[1])));
    const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(out.val[2]), vqmovn_s32(out.val[3])));
    vst1q_s8(output_ptr, vcombine_s8(pa, pb));
}

inline void
store_quantized(int8_t *output_ptr, const float32x4x4_t &rf, const float32x4_t &offset, const float32x4x4_t &invscale)
{
    // Adjust offset to round to nearest.
    const uint32x4x4_t cmp = {{
#ifdef __aarch64__
        vcltzq_f32(rf.val[0]),
        vcltzq_f32(rf.val[1]),
        vcltzq_f32(rf.val[2]),
        vcltzq_f32(rf.val[3]),
#else  // __aarch64__
        vcltq_f32(rf.val[0], vdupq_n_f32(0.0f)),
        vcltq_f32(rf.val[1], vdupq_n_f32(0.0f)),
        vcltq_f32(rf.val[2], vdupq_n_f32(0.0f)),
        vcltq_f32(rf.val[3], vdupq_n_f32(0.0f)),
#endif // __aarch64__
    }};
    const float32x4_t   neg_point_5 = vdupq_n_f32(-0.5f);
    const float32x4_t   pos_point_5 = vdupq_n_f32(0.5f);
    const float32x4x4_t adj_offset  = {{
         vaddq_f32(offset, vbslq_f32(cmp.val[0], neg_point_5, pos_point_5)),
         vaddq_f32(offset, vbslq_f32(cmp.val[1], neg_point_5, pos_point_5)),
         vaddq_f32(offset, vbslq_f32(cmp.val[2], neg_point_5, pos_point_5)),
         vaddq_f32(offset, vbslq_f32(cmp.val[3], neg_point_5, pos_point_5)),
    }};

    const int32x4x4_t out = {{
        vcvtq_s32_f32(vmlaq_f32(adj_offset.val[0], rf.val[0], invscale.val[0])),
        vcvtq_s32_f32(vmlaq_f32(adj_offset.val[1], rf.val[1], invscale.val[1])),
        vcvtq_s32_f32(vmlaq_f32(adj_offset.val[2], rf.val[2], invscale.val[2])),
        vcvtq_s32_f32(vmlaq_f32(adj_offset.val[3], rf.val[3], invscale.val[3])),
    }};
    store_quantized(output_ptr, out);
}

inline void
store_quantized(int8_t *output_ptr, const float32x4x4_t &rf, const float32x4_t &offset, const float32x4_t &invscale)
{
    return store_quantized(output_ptr, rf, offset,
                           float32x4x4_t{{
                               invscale,
                               invscale,
                               invscale,
                               invscale,
                           }});
}

template <typename Input,
          typename = std::enable_if_t<std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value>>
union ElementwiseQuantizedScalarParams
{
    // For ops that expect dequantized inputs.
    struct Generic
    {
        float                   a;
        float                   b;
        UniformQuantizationInfo qinfo; // Unused in comparison operations
    } generic;

    // The prelu implementation expects quantized inputs.
    struct Prelu
    {
        Input   a;
        Input   b;
        float   s1;       // scale_out / scale_a
        float   s2;       // s1 / scale_b
        int32_t a_offset; // Input quantization offset
        int32_t b_offset; // Input quantization offset
        int32_t o_offset; // Output quantization offset
    } prelu;
};

template <ArithmeticOperation op,
          typename Input,
          typename Output,
          typename = std::enable_if_t<(std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value) &&
                                      (std::is_same<Output, int8_t>::value || std::is_same<Output, uint8_t>::value)>>
inline Output elementwise_arithm_op_quantized_scalar(const ElementwiseQuantizedScalarParams<Input> &params)
{
    const auto &_   = params.generic;
    const float res = elementwise_arithm_op_scalar<op>(_.a, _.b);
    return Qasymm8QuantizationHelper<Output>::quantize(res, _.qinfo);
}

// Specialization that optimizes PReLU by fusing quantization logic into the operator logic.
// Turns
//   dequant(a) > 0 ? quant(dequant(a)) : quant(dequant(a) * dequant(b))
// into
//   a > offset_a ? quant_s1(a - offset_a) : quant_s2((a - offset_a) * (b - offset_b))
// where quant_s1 and quant_s2 use the normal output offset, but scales s1 = scale_out / scale_a and s2 = s1 / scale_b respectively.
template <typename Input,
          typename Output,
          typename = std::enable_if_t<(std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value) &&
                                      (std::is_same<Output, int8_t>::value || std::is_same<Output, uint8_t>::value)>>
inline Output elementwise_prelu_quantized_scalar(
    Input a, Input b, float s1, float s2, int32_t a_offset, int32_t b_offset, int32_t o_offset)
{
    int a_minus_offset = static_cast<int>(a) - a_offset;
    if (a_minus_offset > 0)
    {
        return Qasymm8QuantizationHelper<Output>::quantize(static_cast<float>(a_minus_offset),
                                                           UniformQuantizationInfo{s1, o_offset});
    }
    else
    {
        int b_minus_offset = static_cast<int>(b) - b_offset;
        return Qasymm8QuantizationHelper<Output>::quantize(static_cast<float>(a_minus_offset) * b_minus_offset,
                                                           UniformQuantizationInfo{s2, o_offset});
    }
}

template <>
inline uint8_t elementwise_arithm_op_quantized_scalar<ArithmeticOperation::PRELU, uint8_t, uint8_t>(
    const ElementwiseQuantizedScalarParams<uint8_t> &params)
{
    const auto &_ = params.prelu;
    return elementwise_prelu_quantized_scalar<uint8_t, uint8_t>(_.a, _.b, _.s1, _.s2, _.a_offset, _.b_offset,
                                                                _.o_offset);
}

template <>
inline int8_t elementwise_arithm_op_quantized_scalar<ArithmeticOperation::PRELU, int8_t, int8_t>(
    const ElementwiseQuantizedScalarParams<int8_t> &params)
{
    const auto &_ = params.prelu;
    return elementwise_prelu_quantized_scalar<int8_t, int8_t>(_.a, _.b, _.s1, _.s2, _.a_offset, _.b_offset, _.o_offset);
}

template <ArithmeticOperation op>
float32x4x4_t elementwise_arithm_op(const float32x4x4_t &a, const float32x4x4_t &b)
{
    using neon_vector_float = wrapper::traits::neon_vector<float, 4>;
    float32x4x4_t out       = {{
              elementwise_arithm_op<op, neon_vector_float>(a.val[0], b.val[0]),
              elementwise_arithm_op<op, neon_vector_float>(a.val[1], b.val[1]),
              elementwise_arithm_op<op, neon_vector_float>(a.val[2], b.val[2]),
              elementwise_arithm_op<op, neon_vector_float>(a.val[3], b.val[3]),
    }};
    return out;
}

template <ComparisonOperation op,
          typename Input,
          typename = std::enable_if_t<std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value>>
inline uint8_t elementwise_comp_op_quantized_scalar(const ElementwiseQuantizedScalarParams<Input> &params)
{
    const auto &_ = params.generic;
    return elementwise_comp_op_scalar<op>(_.a, _.b);
}

template <ComparisonOperation op>
inline uint32x4x4_t elementwise_comp_op(const float32x4x4_t &a, const float32x4x4_t &b)
{
    uint32x4x4_t out = {{elementwise_comp_op<op, float32x4_t, uint32x4_t>(a.val[0], b.val[0]),
                         elementwise_comp_op<op, float32x4_t, uint32x4_t>(a.val[1], b.val[1]),
                         elementwise_comp_op<op, float32x4_t, uint32x4_t>(a.val[2], b.val[2]),
                         elementwise_comp_op<op, float32x4_t, uint32x4_t>(a.val[3], b.val[3])}};
    return out;
}

template <ArithmeticOperation op,
          typename Input,
          typename Output,
          typename = std::enable_if_t<(std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value) &&
                                      (std::is_same<Output, int8_t>::value || std::is_same<Output, uint8_t>::value)>>
inline int elementwise_arithm_op_quantized_loop(int          window_start_x,
                                                int          window_end_x,
                                                int          window_step_x,
                                                const Input *input1_ptr,
                                                const Input *input2_ptr,
                                                Output      *output_ptr,
                                                int32x4_t    voffset1,
                                                int32x4_t    voffset2,
                                                float32x4_t  vscale1,
                                                float32x4_t  vscale2,
                                                float32x4_t  voffseto,
                                                float32x4_t  invvscaleo)
{
    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        // Get inputs and compute output
        const float32x4x4_t af = load_quantized(input1_ptr + x, voffset1, vscale1);
        const float32x4x4_t bf = load_quantized(input2_ptr + x, voffset2, vscale2);
        const float32x4x4_t rf = elementwise_arithm_op<op>(af, bf);
        store_quantized(output_ptr + x, rf, voffseto, invvscaleo);
    }
    return x;
}

inline int32x4x4_t widen_to_i32_and_offset(uint8x16_t q, int32x4_t offset)
{
    const int16x8_t low16x8  = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(q)));
    const int16x8_t high16x8 = vreinterpretq_s16_u16(wrapper::vmovl_high(q));
    return {{
        vaddw_s16(offset, vget_low_s16(low16x8)),
        wrapper::vaddw_high(offset, low16x8),
        vaddw_s16(offset, vget_low_s16(high16x8)),
        wrapper::vaddw_high(offset, high16x8),
    }};
}

inline int32x4x4_t widen_to_i32_and_offset(int8x16_t q, int32x4_t offset)
{
    const int16x8_t low16x8  = vmovl_s8(vget_low_s8(q));
    const int16x8_t high16x8 = wrapper::vmovl_high(q);
    return {{
        vaddw_s16(offset, vget_low_s16(low16x8)),
        wrapper::vaddw_high(offset, low16x8),
        vaddw_s16(offset, vget_low_s16(high16x8)),
        wrapper::vaddw_high(offset, high16x8),
    }};
}

// Specialization that optimizes PReLU by fusing quantization logic into the operator logic.
// Turns
//   dequant(a) > 0 ? quant(dequant(a)) : quant(dequant(a) * dequant(b))
// into
//   a > offset_a ? quant_s1(a - offset_a) : quant_s2((a - offset_a) * (b - offset_b))
// where quant_s1 and quant_s2 use the normal output offset, but scales s1 = scale_out / scale_a and s2 = s1 / scale_b respectively.
template <typename Input,
          typename Output,
          typename = std::enable_if_t<(std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value) &&
                                      (std::is_same<Output, int8_t>::value || std::is_same<Output, uint8_t>::value)>>
inline int elementwise_prelu_quantized_loop(int          window_start_x,
                                            int          window_end_x,
                                            int          window_step_x,
                                            const Input *input1_ptr,
                                            const Input *input2_ptr,
                                            Output      *output_ptr,
                                            int32x4_t    v_neg_offset1,
                                            int32x4_t    v_neg_offset2,
                                            float32x4_t  vinv_s1,
                                            float32x4_t  vinv_s2,
                                            float32x4_t  voffseto,
                                            float32x4_t  invvscaleo)
{
    ARM_COMPUTE_UNUSED(invvscaleo);

    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const int32x4x4_t a_minus_offset = widen_to_i32_and_offset(wrapper::vloadq(input1_ptr + x), v_neg_offset1);
        const int32x4x4_t b_minus_offset = widen_to_i32_and_offset(wrapper::vloadq(input2_ptr + x), v_neg_offset2);

        // prelu: a > offset_a ? (a - offset_a) : ((a - offset_a) * (b - offset_b))
        const uint32x4x4_t cmp = {{
#ifdef __aarch64__
            vcgtzq_s32(a_minus_offset.val[0]),
            vcgtzq_s32(a_minus_offset.val[1]),
            vcgtzq_s32(a_minus_offset.val[2]),
            vcgtzq_s32(a_minus_offset.val[3]),
#else  // __aarch64__
            vcgtq_s32(a_minus_offset.val[0], vdupq_n_s32(0)),
            vcgtq_s32(a_minus_offset.val[1], vdupq_n_s32(0)),
            vcgtq_s32(a_minus_offset.val[2], vdupq_n_s32(0)),
            vcgtq_s32(a_minus_offset.val[3], vdupq_n_s32(0)),
#endif // __aarch64__
        }};
        const int32x4x4_t   prelu_false = {{
              vmulq_s32(a_minus_offset.val[0], b_minus_offset.val[0]),
              vmulq_s32(a_minus_offset.val[1], b_minus_offset.val[1]),
              vmulq_s32(a_minus_offset.val[2], b_minus_offset.val[2]),
              vmulq_s32(a_minus_offset.val[3], b_minus_offset.val[3]),
        }};
        const int32x4x4_t   prelui      = {{
                   vbslq_s32(cmp.val[0], a_minus_offset.val[0], prelu_false.val[0]),
                   vbslq_s32(cmp.val[1], a_minus_offset.val[1], prelu_false.val[1]),
                   vbslq_s32(cmp.val[2], a_minus_offset.val[2], prelu_false.val[2]),
                   vbslq_s32(cmp.val[3], a_minus_offset.val[3], prelu_false.val[3]),
        }};
        const float32x4x4_t preluf      = {{
                 vcvtq_f32_s32(prelui.val[0]),
                 vcvtq_f32_s32(prelui.val[1]),
                 vcvtq_f32_s32(prelui.val[2]),
                 vcvtq_f32_s32(prelui.val[3]),
        }};

        // quant(prelu)
        const float32x4x4_t vinv_s = {{
            vbslq_f32(cmp.val[0], vinv_s1, vinv_s2),
            vbslq_f32(cmp.val[1], vinv_s1, vinv_s2),
            vbslq_f32(cmp.val[2], vinv_s1, vinv_s2),
            vbslq_f32(cmp.val[3], vinv_s1, vinv_s2),
        }};
        store_quantized(output_ptr + x, preluf, voffseto, vinv_s);
    }
    return x;
}

// Note: v_neg_offset1 and v_neg_offset2 are negated compared to the generic template.
template <>
inline int elementwise_arithm_op_quantized_loop<ArithmeticOperation::PRELU, uint8_t, uint8_t>(int window_start_x,
                                                                                              int window_end_x,
                                                                                              int window_step_x,
                                                                                              const uint8_t *input1_ptr,
                                                                                              const uint8_t *input2_ptr,
                                                                                              uint8_t       *output_ptr,
                                                                                              int32x4_t   v_neg_offset1,
                                                                                              int32x4_t   v_neg_offset2,
                                                                                              float32x4_t vinv_s1,
                                                                                              float32x4_t vinv_s2,
                                                                                              float32x4_t voffseto,
                                                                                              float32x4_t invvscaleo)
{
    return elementwise_prelu_quantized_loop(window_start_x, window_end_x, window_step_x, input1_ptr, input2_ptr,
                                            output_ptr, v_neg_offset1, v_neg_offset2, vinv_s1, vinv_s2, voffseto,
                                            invvscaleo);
}

// Note: v_neg_offset1 and v_neg_offset2 are negated compared to the generic template.
template <>
inline int elementwise_arithm_op_quantized_loop<ArithmeticOperation::PRELU, int8_t, int8_t>(int window_start_x,
                                                                                            int window_end_x,
                                                                                            int window_step_x,
                                                                                            const int8_t *input1_ptr,
                                                                                            const int8_t *input2_ptr,
                                                                                            int8_t       *output_ptr,
                                                                                            int32x4_t     v_neg_offset1,
                                                                                            int32x4_t     v_neg_offset2,
                                                                                            float32x4_t   vinv_s1,
                                                                                            float32x4_t   vinv_s2,
                                                                                            float32x4_t   voffseto,
                                                                                            float32x4_t   invvscaleo)
{
    return elementwise_prelu_quantized_loop(window_start_x, window_end_x, window_step_x, input1_ptr, input2_ptr,
                                            output_ptr, v_neg_offset1, v_neg_offset2, vinv_s1, vinv_s2, voffseto,
                                            invvscaleo);
}

template <typename Input,
          typename = std::enable_if_t<std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value>>
union ElementwiseQuantizedBroadcastParams
{
    // For ops that expect dequantized inputs.
    struct Generic
    {
        float32x4x4_t broadcast_vector;
        float32x4_t   vscale_non_broadcast;
        float32x4_t   invvscaleo; // Not used by comparisons
    } generic;

    // The prelu implementation expects quantized inputs.
    struct Prelu
    {
        Input       broadcast_value;
        int32_t     offset_broadcast;
        float       s1;      // scale_out / scale_a
        float32x4_t vinv_s1; // scale_a / scale_out
        float32x4_t vinv_s2; // vinv_s1 * scale_b
        int32_t     o_offset;
    } prelu;
};

template <ArithmeticOperation op,
          typename Input,
          typename Output,
          typename = std::enable_if_t<(std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value) &&
                                      (std::is_same<Output, int8_t>::value || std::is_same<Output, uint8_t>::value)>>
inline int elementwise_arithm_op_quantized_broadcast_loop(int          window_start_x,
                                                          int          window_end_x,
                                                          int          window_step_x,
                                                          const Input *non_broadcast_input_ptr,
                                                          Output      *output_ptr,
                                                          int32x4_t    voffset_non_broadcast,
                                                          float32x4_t  voffseto,
                                                          bool         reorder,
                                                          const ElementwiseQuantizedBroadcastParams<Input> &params)
{
    const float32x4x4_t &broadcast_vector     = params.generic.broadcast_vector;
    const float32x4_t   &vscale_non_broadcast = params.generic.vscale_non_broadcast;
    const float32x4_t   &invvscaleo           = params.generic.invvscaleo;

    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const float32x4x4_t af =
            load_quantized(non_broadcast_input_ptr + x, voffset_non_broadcast, vscale_non_broadcast);
        const float32x4x4_t rf =
            elementwise_arithm_op<op>(reorder ? broadcast_vector : af, reorder ? af : broadcast_vector);
        store_quantized(output_ptr + x, rf, voffseto, invvscaleo);
    }
    return x;
}

// Implements one vector worth of PReLU on quantized data where the first operand is broadcast.
template <typename Input,
          typename Output,
          typename = std::enable_if_t<(std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value) &&
                                      (std::is_same<Output, int8_t>::value || std::is_same<Output, uint8_t>::value)>>
inline void elementwise_prelu_quantized_broadcast_vector(int32_t      a_minus_offset,
                                                         const Input *b_input_ptr,
                                                         int32x4_t    b_neg_voffset,
                                                         Output      *output_ptr,
                                                         float        s1,
                                                         float32x4_t  vinv_s2,
                                                         float32x4_t  o_voffset,
                                                         int32_t      o_offset)
{
    // a > offset_a ? (a - offset_a) : ((a - offset_a) * (b - offset_b))
    if (a_minus_offset > 0)
    {
        const Output res   = Qasymm8QuantizationHelper<Output>::quantize(static_cast<float>(a_minus_offset),
                                                                         UniformQuantizationInfo{s1, o_offset});
        const auto   res_v = wrapper::vdup_n(res, wrapper::traits::vector_128_tag{});
        wrapper::vstore(output_ptr, res_v);
    }
    else
    {
        const int32x4x4_t   b_minus_offset = widen_to_i32_and_offset(wrapper::vloadq(b_input_ptr), b_neg_voffset);
        const int32x4x4_t   prelu_false    = {{
                 vmulq_n_s32(b_minus_offset.val[0], a_minus_offset),
                 vmulq_n_s32(b_minus_offset.val[1], a_minus_offset),
                 vmulq_n_s32(b_minus_offset.val[2], a_minus_offset),
                 vmulq_n_s32(b_minus_offset.val[3], a_minus_offset),
        }};
        const float32x4x4_t preluf         = {{
                    vcvtq_f32_s32(prelu_false.val[0]),
                    vcvtq_f32_s32(prelu_false.val[1]),
                    vcvtq_f32_s32(prelu_false.val[2]),
                    vcvtq_f32_s32(prelu_false.val[3]),
        }};

        store_quantized(output_ptr, preluf, o_voffset, vinv_s2);
    }
}

// Implements one vector worth of PReLU on quantized data where the second operand is broadcast.
template <typename Input,
          typename Output,
          typename = std::enable_if_t<(std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value) &&
                                      (std::is_same<Output, int8_t>::value || std::is_same<Output, uint8_t>::value)>>
inline void elementwise_prelu_quantized_broadcast_vector(const Input *a_input_ptr,
                                                         int32x4_t    a_neg_voffset,
                                                         int32_t      b_minus_offset,
                                                         Output      *output_ptr,
                                                         float32x4_t  vinv_s1,
                                                         float32x4_t  vinv_s2,
                                                         float32x4_t  voffseto)
{
    const int32x4x4_t a_minus_offset = widen_to_i32_and_offset(wrapper::vloadq(a_input_ptr), a_neg_voffset);

    // prelu: a > offset_a ? (a - offset_a) : ((a - offset_a) * (b - offset_b))
    const uint32x4x4_t cmp = {{
#ifdef __aarch64__
        vcgtzq_s32(a_minus_offset.val[0]),
        vcgtzq_s32(a_minus_offset.val[1]),
        vcgtzq_s32(a_minus_offset.val[2]),
        vcgtzq_s32(a_minus_offset.val[3]),
#else  // __aarch64__
        vcgtq_s32(a_minus_offset.val[0], vdupq_n_s32(0)),
        vcgtq_s32(a_minus_offset.val[1], vdupq_n_s32(0)),
        vcgtq_s32(a_minus_offset.val[2], vdupq_n_s32(0)),
        vcgtq_s32(a_minus_offset.val[3], vdupq_n_s32(0)),
#endif // __aarch64__
    }};
    const int32x4x4_t   prelu_false = {{
          vmulq_n_s32(a_minus_offset.val[0], b_minus_offset),
          vmulq_n_s32(a_minus_offset.val[1], b_minus_offset),
          vmulq_n_s32(a_minus_offset.val[2], b_minus_offset),
          vmulq_n_s32(a_minus_offset.val[3], b_minus_offset),
    }};
    const int32x4x4_t   prelui      = {{
               vbslq_s32(cmp.val[0], a_minus_offset.val[0], prelu_false.val[0]),
               vbslq_s32(cmp.val[1], a_minus_offset.val[1], prelu_false.val[1]),
               vbslq_s32(cmp.val[2], a_minus_offset.val[2], prelu_false.val[2]),
               vbslq_s32(cmp.val[3], a_minus_offset.val[3], prelu_false.val[3]),
    }};
    const float32x4x4_t preluf      = {{
             vcvtq_f32_s32(prelui.val[0]),
             vcvtq_f32_s32(prelui.val[1]),
             vcvtq_f32_s32(prelui.val[2]),
             vcvtq_f32_s32(prelui.val[3]),
    }};

    // quant(prelu)
    const float32x4x4_t vinv_s = {{
        vbslq_f32(cmp.val[0], vinv_s1, vinv_s2),
        vbslq_f32(cmp.val[1], vinv_s1, vinv_s2),
        vbslq_f32(cmp.val[2], vinv_s1, vinv_s2),
        vbslq_f32(cmp.val[3], vinv_s1, vinv_s2),
    }};
    store_quantized(output_ptr, preluf, voffseto, vinv_s);
}

// Specialization that optimizes PReLU by fusing quantization logic into the operator logic.
// Turns
//   dequant(a) > 0 ? quant(dequant(a)) : quant(dequant(a) * dequant(b))
// into
//   a > offset_a ? quant_s1(a - offset_a) : quant_s2((a - offset_a) * (b - offset_b))
// where quant_s1 and quant_s2 use the normal output offset, but scales s1 = scale_out / scale_a and s2 = s1 / scale_b respectively.
template <typename Input,
          typename Output,
          typename = std::enable_if_t<(std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value) &&
                                      (std::is_same<Output, int8_t>::value || std::is_same<Output, uint8_t>::value)>>
inline int elementwise_prelu_quantized_broadcast_loop(int          window_start_x,
                                                      int          window_end_x,
                                                      int          window_step_x,
                                                      const Input *non_broadcast_input_ptr,
                                                      Output      *output_ptr,
                                                      int32x4_t    vnegoffset_non_broadcast,
                                                      float32x4_t  voffseto,
                                                      bool         reorder,
                                                      const ElementwiseQuantizedBroadcastParams<Input> &params)
{
    const auto   &_                      = params.prelu;
    const int32_t broadcast_q            = static_cast<int32_t>(_.broadcast_value);
    const int32_t broadcast_minus_offset = broadcast_q - _.offset_broadcast;

    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        if (reorder)
        {
            // With reorder, (a) is broadcast and (b) is non-broadcast.
            elementwise_prelu_quantized_broadcast_vector(broadcast_minus_offset, non_broadcast_input_ptr + x,
                                                         vnegoffset_non_broadcast, output_ptr + x, _.s1, _.vinv_s2,
                                                         voffseto, _.o_offset);
        }
        else
        {
            // Without reorder, (a) is non-broadcast and (b) is broadcast.
            elementwise_prelu_quantized_broadcast_vector(non_broadcast_input_ptr + x, vnegoffset_non_broadcast,
                                                         broadcast_minus_offset, output_ptr + x, _.vinv_s1, _.vinv_s2,
                                                         voffseto);
        }
    }
    return x;
}

// Note: vnegoffset_non_broadcast is negated compared to the generic template.
template <>
inline int elementwise_arithm_op_quantized_broadcast_loop<ArithmeticOperation::PRELU, uint8_t, uint8_t>(
    int                                                 window_start_x,
    int                                                 window_end_x,
    int                                                 window_step_x,
    const uint8_t                                      *non_broadcast_input_ptr,
    uint8_t                                            *output_ptr,
    int32x4_t                                           vnegoffset_non_broadcast,
    float32x4_t                                         voffseto,
    bool                                                reorder,
    const ElementwiseQuantizedBroadcastParams<uint8_t> &params)
{
    return elementwise_prelu_quantized_broadcast_loop(window_start_x, window_end_x, window_step_x,
                                                      non_broadcast_input_ptr, output_ptr, vnegoffset_non_broadcast,
                                                      voffseto, reorder, params);
}

// Note: vnegoffset_non_broadcast is negated compared to the generic template.
template <>
inline int elementwise_arithm_op_quantized_broadcast_loop<ArithmeticOperation::PRELU, int8_t, int8_t>(
    int                                                window_start_x,
    int                                                window_end_x,
    int                                                window_step_x,
    const int8_t                                      *non_broadcast_input_ptr,
    int8_t                                            *output_ptr,
    int32x4_t                                          vnegoffset_non_broadcast,
    float32x4_t                                        voffseto,
    bool                                               reorder,
    const ElementwiseQuantizedBroadcastParams<int8_t> &params)
{
    return elementwise_prelu_quantized_broadcast_loop(window_start_x, window_end_x, window_step_x,
                                                      non_broadcast_input_ptr, output_ptr, vnegoffset_non_broadcast,
                                                      voffseto, reorder, params);
}

template <ComparisonOperation op,
          typename Input,
          typename = std::enable_if_t<std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value>>
inline int elementwise_comp_op_quantized_loop(int          window_start_x,
                                              int          window_end_x,
                                              int          window_step_x,
                                              const Input *input1_ptr,
                                              const Input *input2_ptr,
                                              uint8_t     *output_ptr,
                                              int32x4_t    voffset1,
                                              int32x4_t    voffset2,
                                              float32x4_t  vscale1,
                                              float32x4_t  vscale2,
                                              float32x4_t  voffseto,
                                              float32x4_t  invvscaleo)
{
    ARM_COMPUTE_UNUSED(voffseto, invvscaleo);
    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const float32x4x4_t af = load_quantized(input1_ptr + x, voffset1, vscale1);
        const float32x4x4_t bf = load_quantized(input2_ptr + x, voffset2, vscale2);
        const uint32x4x4_t  rf = elementwise_comp_op<op>(af, bf);
        store_quantized(output_ptr + x, rf);
    }
    return x;
}

template <ComparisonOperation op,
          typename Input,
          typename = std::enable_if_t<std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value>>
inline int elementwise_comp_op_quantized_broadcast_loop(int          window_start_x,
                                                        int          window_end_x,
                                                        int          window_step_x,
                                                        const Input *non_broadcast_input_ptr,
                                                        uint8_t     *output_ptr,
                                                        int32x4_t    voffset_non_broadcast,
                                                        float32x4_t  voffseto,
                                                        bool         reorder,
                                                        const ElementwiseQuantizedBroadcastParams<Input> &params)
{
    ARM_COMPUTE_UNUSED(voffseto);

    const float32x4_t   &vscale_non_broadcast = params.generic.vscale_non_broadcast;
    const float32x4x4_t &broadcast_vector     = params.generic.broadcast_vector;

    int x = window_start_x;
    for (; x <= (window_end_x - window_step_x); x += window_step_x)
    {
        const float32x4x4_t af =
            load_quantized(non_broadcast_input_ptr + x, voffset_non_broadcast, vscale_non_broadcast);
        const uint32x4x4_t rf =
            elementwise_comp_op<op>(reorder ? broadcast_vector : af, reorder ? af : broadcast_vector);
        store_quantized(output_ptr + x, rf);
    }
    return x;
}

template <typename Input,
          typename Output,
          typename = std::enable_if_t<(std::is_same<Input, int8_t>::value || std::is_same<Input, uint8_t>::value) &&
                                      (std::is_same<Output, int8_t>::value || std::is_same<Output, uint8_t>::value)>>
inline void elementwise_op_quantized(const ITensor *in1,
                                     const ITensor *in2,
                                     ITensor       *out,
                                     const Window  &window,
                                     Output (*scalar_func)(const ElementwiseQuantizedScalarParams<Input> &),
                                     int (*broadcast_func)(int,
                                                           int,
                                                           int,
                                                           const Input *,
                                                           Output *,
                                                           int32x4_t,
                                                           float32x4_t,
                                                           const bool,
                                                           const ElementwiseQuantizedBroadcastParams<Input> &),
                                     int (*neon_func)(int,
                                                      int,
                                                      int,
                                                      const Input *,
                                                      const Input *,
                                                      Output *,
                                                      int32x4_t,
                                                      int32x4_t,
                                                      float32x4_t,
                                                      float32x4_t,
                                                      float32x4_t,
                                                      float32x4_t))
{
    bool is_prelu = scalar_func == &elementwise_arithm_op_quantized_scalar<ArithmeticOperation::PRELU, Input, Output>;
    if (is_prelu)
    {
        ARM_COMPUTE_ERROR_ON(
            broadcast_func !=
            (&elementwise_arithm_op_quantized_broadcast_loop<ArithmeticOperation::PRELU, Input, Output>));
        ARM_COMPUTE_ERROR_ON(neon_func !=
                             (&elementwise_arithm_op_quantized_loop<ArithmeticOperation::PRELU, Input, Output>));
    }

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
    const UniformQuantizationInfo input1_qinfo = in1->info()->quantization_info().uniform();
    const UniformQuantizationInfo input2_qinfo = in2->info()->quantization_info().uniform();

    const float       prelu_s1      = is_prelu ? output_qinfo.scale / input1_qinfo.scale : 1.0f;
    const float       prelu_s2      = is_prelu ? prelu_s1 / input2_qinfo.scale : 1.0f;
    const float32x4_t prelu_vinv_s1 = is_prelu ? vdupq_n_f32(input1_qinfo.scale / output_qinfo.scale) : float32x4_t{};
    const float32x4_t prelu_vinv_s2 =
        is_prelu ? vdupq_n_f32(input1_qinfo.scale * input2_qinfo.scale / output_qinfo.scale) : float32x4_t{};

    const float32x4_t voffseto   = vdupq_n_f32(output_qinfo.offset);
    const float32x4_t invvscaleo = vdupq_n_f32(1.f / output_qinfo.scale);

    if (is_broadcast_across_x)
    {
        // Select the broadcast input on the X axis
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? in2 : in1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;

        const UniformQuantizationInfo broadcast_qinfo     = broadcast_tensor->info()->quantization_info().uniform();
        const UniformQuantizationInfo non_broadcast_qinfo = non_broadcast_tensor->info()->quantization_info().uniform();

        const int32x4_t voffset_non_broadcast =
            vdupq_n_s32(is_prelu ? -non_broadcast_qinfo.offset : non_broadcast_qinfo.offset);
        const float32x4_t vscale_non_broadcast = !is_prelu ? vdupq_n_f32(non_broadcast_qinfo.scale) : float32x4_t{};

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(out, win);

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                const auto non_broadcast_input_ptr = reinterpret_cast<const Input *>(non_broadcast_input.ptr());
                const auto output_ptr              = reinterpret_cast<Output *>(output.ptr());

                const Input broadcast_value = *reinterpret_cast<const Input *>(broadcast_input.ptr());
                const float broadcast_value_deq =
                    is_prelu ? 0.0f : Qasymm8QuantizationHelper<Input>::dequantize(broadcast_value, broadcast_qinfo);

                ElementwiseQuantizedBroadcastParams<Input> params{};
                if (is_prelu)
                {
                    params.prelu = {
                        broadcast_value, broadcast_qinfo.offset, prelu_s1,
                        prelu_vinv_s1,   prelu_vinv_s2,          output_qinfo.offset,
                    };
                }
                else
                {
                    const float32x4x4_t broadcast_vector = {{
                        vdupq_n_f32(broadcast_value_deq),
                        vdupq_n_f32(broadcast_value_deq),
                        vdupq_n_f32(broadcast_value_deq),
                        vdupq_n_f32(broadcast_value_deq),
                    }};
                    params.generic                       = {
                                              broadcast_vector,
                                              vscale_non_broadcast,
                                              invvscaleo,
                    };
                }

                int x = (*broadcast_func)(window_start_x, window_end_x, window_step_x, non_broadcast_input_ptr,
                                          output_ptr, voffset_non_broadcast, voffseto, !is_broadcast_input_2, params);
                for (; x < window_end_x; ++x)
                {
                    const Input non_broadcast_value = *(non_broadcast_input_ptr + x);

                    ElementwiseQuantizedScalarParams<Input> params{};
                    if (is_prelu)
                    {
                        params.prelu = {
                            !is_broadcast_input_2 ? broadcast_value : non_broadcast_value,
                            !is_broadcast_input_2 ? non_broadcast_value : broadcast_value,
                            prelu_s1,
                            prelu_s2,
                            input1_qinfo.offset,
                            input2_qinfo.offset,
                            output_qinfo.offset,
                        };
                    }
                    else
                    {
                        const float non_broadcast_value_deq =
                            Qasymm8QuantizationHelper<Input>::dequantize(non_broadcast_value, non_broadcast_qinfo);
                        params.generic = {
                            !is_broadcast_input_2 ? broadcast_value_deq : non_broadcast_value_deq,
                            !is_broadcast_input_2 ? non_broadcast_value_deq : broadcast_value_deq,
                            output_qinfo,
                        };
                    }

                    *(output_ptr + x) = (*scalar_func)(params);
                }
            },
            broadcast_input, non_broadcast_input, output);
    }
    else
    {
        // Input1 quantization info
        const int32x4_t   voffset1 = vdupq_n_s32(is_prelu ? -input1_qinfo.offset : input1_qinfo.offset);
        const float32x4_t vscale1  = is_prelu ? prelu_vinv_s1 : vdupq_n_f32(input1_qinfo.scale);

        // Input2 quantization info
        const int32x4_t   voffset2 = vdupq_n_s32(is_prelu ? -input2_qinfo.offset : input2_qinfo.offset);
        const float32x4_t vscale2  = is_prelu ? prelu_vinv_s2 : vdupq_n_f32(input2_qinfo.scale);

        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(in1, input1_win);
        Iterator input2(in2, input2_win);
        Iterator output(out, win);

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                const auto input1_ptr = reinterpret_cast<const Input *>(input1.ptr());
                const auto input2_ptr = reinterpret_cast<const Input *>(input2.ptr());
                const auto output_ptr = reinterpret_cast<Output *>(output.ptr());

                int x = (*neon_func)(window_start_x, window_end_x, window_step_x, input1_ptr, input2_ptr, output_ptr,
                                     voffset1, voffset2, vscale1, vscale2, voffseto, invvscaleo);
                for (; x < window_end_x; ++x)
                {
                    const Input input1_value = *(input1_ptr + x);
                    const Input input2_value = *(input2_ptr + x);

                    ElementwiseQuantizedScalarParams<Input> params{};
                    if (is_prelu)
                    {
                        params.prelu = {
                            input1_value,        input2_value,        prelu_s1, prelu_s2, input1_qinfo.offset,
                            input2_qinfo.offset, output_qinfo.offset,
                        };
                    }
                    else
                    {
                        params.generic = {
                            Qasymm8QuantizationHelper<Input>::dequantize(input1_value, input1_qinfo),
                            Qasymm8QuantizationHelper<Input>::dequantize(input2_value, input2_qinfo),
                            output_qinfo,
                        };
                    }

                    *(output_ptr + x) = (*scalar_func)(params);
                }
            },
            input1, input2, output);
    }
}

template <ArithmeticOperation op>
void elementwise_arithm_op_quantized(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op_quantized<uint8_t, uint8_t>(in1, in2, out, window,
                                               &elementwise_arithm_op_quantized_scalar<op, uint8_t, uint8_t>,
                                               &elementwise_arithm_op_quantized_broadcast_loop<op, uint8_t, uint8_t>,
                                               &elementwise_arithm_op_quantized_loop<op, uint8_t, uint8_t>);
}

template <ArithmeticOperation op>
void elementwise_arithm_op_quantized_signed(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op_quantized<int8_t, int8_t>(in1, in2, out, window,
                                             &elementwise_arithm_op_quantized_scalar<op, int8_t, int8_t>,
                                             &elementwise_arithm_op_quantized_broadcast_loop<op, int8_t, int8_t>,
                                             &elementwise_arithm_op_quantized_loop<op, int8_t, int8_t>);
}

template <ComparisonOperation op>
void elementwise_comp_op_quantized(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op_quantized<uint8_t, uint8_t>(
        in1, in2, out, window, &elementwise_comp_op_quantized_scalar<op, uint8_t>,
        &elementwise_comp_op_quantized_broadcast_loop<op, uint8_t>, &elementwise_comp_op_quantized_loop<op, uint8_t>);
}

template <ComparisonOperation op>
void elementwise_comp_op_quantized_signed(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    elementwise_op_quantized<int8_t, uint8_t>(in1, in2, out, window, &elementwise_comp_op_quantized_scalar<op, int8_t>,
                                              &elementwise_comp_op_quantized_broadcast_loop<op, int8_t>,
                                              &elementwise_comp_op_quantized_loop<op, int8_t>);
}
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_ELEMENTWISE_BINARY_GENERIC_NEON_IMPL_H
