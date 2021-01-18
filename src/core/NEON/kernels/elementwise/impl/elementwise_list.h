/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_ELEMENTWISE_LIST_H
#define SRC_CORE_NEON_KERNELS_ELEMENTWISE_LIST_H

#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
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

template <ArithmeticOperation    op, typename ScalarType, typename VectorType>
inline typename VectorType::type elementwise_arithm_op_broadcast(const typename VectorType::type &a, const ScalarType &broadcast_value, const bool reorder)
{
    using tag_type = typename VectorType::tag_type;
    using vec_type = typename VectorType::type;

    vec_type broadcast_vector = wrapper::vdup_n(broadcast_value, tag_type{});
    return elementwise_arithm_op<op, VectorType>(reorder ? broadcast_vector : a, reorder ? a : broadcast_vector);
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

template <ArithmeticOperation op, typename VectorType>
void elementwise_arithm_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    using scalar_type = typename VectorType::scalar_type;

    elementwise_op<scalar_type, scalar_type, VectorType>(in1, in2, out, window,
                                                         &elementwise_arithm_op_scalar<op, scalar_type>,
                                                         &elementwise_arithm_op_broadcast_loop<op, scalar_type, VectorType>,
                                                         &elementwise_arithm_op_loop<op, scalar_type, VectorType>);
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

template <ComparisonOperation op, typename InputScalarType, typename InputVectorType, typename OutputVectorType>
inline OutputVectorType elementwise_comp_op_broadcast(const InputVectorType &a, const InputScalarType &broadcast_value, const bool reorder)
{
    InputVectorType broadcast_vector = wrapper::vdup_n(broadcast_value, wrapper::traits::vector_128_tag());
    return elementwise_comp_op<op, InputVectorType, OutputVectorType>(reorder ? broadcast_vector : a, reorder ? a : broadcast_vector);
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
} // namesapce cpu
} // namespace arm_compute

#endif /* SRC_CORE_NEON_KERNELS_ELEMENTWISE_LIST_H */