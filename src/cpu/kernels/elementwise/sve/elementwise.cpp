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
#if defined(__ARM_FEATURE_SVE)
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "src/cpu/kernels/elementwise/sve/elementwise_list.h"
#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
using namespace arm_compute::wrapper;

template <typename InputScalarType, typename OutputScalarType, typename OperatorType>
struct LoopArguments
{
    OperatorType           op;
    const InputScalarType *input1_ptr;
    const InputScalarType *input2_ptr;
    OutputScalarType      *output_ptr;
};

template <typename InputScalarType, typename OutputScalarType, typename OperatorType>
struct BroadcastLoopArguments
{
    OperatorType           op;
    const InputScalarType *input1_ptr;
    InputScalarType        broadcast_value;
    OutputScalarType      *output_ptr;
    bool                   reorder;
};

template <typename InputScalarType, typename OutputScalarType>
void arithmetic_op_loop(svbool_t pg, const LoopArguments<InputScalarType, OutputScalarType, ArithmeticOperation> &args)
{
    const auto in1 = svld1(pg, args.input1_ptr);
    const auto in2 = svld1(pg, args.input2_ptr);
    const auto res = elementwise_arithmetic_op<typename sve_vector<InputScalarType>::type>(pg, in1, in2, args.op);
    svst1(pg, args.output_ptr, res);
}

template <typename InputScalarType, typename OutputScalarType>
void arithmetic_op_broadcast_loop(svbool_t pg, const BroadcastLoopArguments<InputScalarType, OutputScalarType, ArithmeticOperation> &args)
{
    const auto non_broadcast_vector = svld1(pg, args.input1_ptr);
    const auto broadcast_vector     = svdup_n(args.broadcast_value);
    const auto in1                  = args.reorder ? broadcast_vector : non_broadcast_vector;
    const auto in2                  = args.reorder ? non_broadcast_vector : broadcast_vector;
    const auto res                  = elementwise_arithmetic_op<typename sve_vector<InputScalarType>::type>(pg, in1, in2, args.op);
    svst1(pg, args.output_ptr, res);
}

template <typename InputScalarType, typename OutputScalarType>
void comparison_op_loop(svbool_t pg, const LoopArguments<InputScalarType, OutputScalarType, ComparisonOperation> &args)
{
    const auto     in1       = svld1(pg, args.input1_ptr);
    const auto     in2       = svld1(pg, args.input2_ptr);
    const auto     res       = elementwise_comparison_op<typename sve_vector<InputScalarType>::type, typename sve_vector<OutputScalarType>::type>(pg, in1, in2, args.op);
    const svbool_t output_pg = narrow_to_byte_predicate<sizeof(InputScalarType)>(pg);
    svst1(output_pg, args.output_ptr, res);
}

template <typename InputScalarType, typename OutputScalarType>
void comparison_op_broadcast_loop(svbool_t pg, const BroadcastLoopArguments<InputScalarType, OutputScalarType, ComparisonOperation> &args)
{
    const auto     non_broadcast_vector = svld1(pg, args.input1_ptr);
    const auto     broadcast_vector     = svdup_n(args.broadcast_value);
    const auto     in1                  = args.reorder ? broadcast_vector : non_broadcast_vector;
    const auto     in2                  = args.reorder ? non_broadcast_vector : broadcast_vector;
    const auto     res                  = elementwise_comparison_op<typename sve_vector<InputScalarType>::type, typename sve_vector<OutputScalarType>::type>(pg, in1, in2, args.op);
    const svbool_t output_pg            = narrow_to_byte_predicate<sizeof(InputScalarType)>(pg);
    svst1(output_pg, args.output_ptr, res);
}

template <typename InputScalarType, typename OutputScalarType, typename OperatorType>
using LoopFuncType = void (*)(svbool_t, const LoopArguments<InputScalarType, OutputScalarType, OperatorType> &);

template <typename InputScalarType, typename OutputScalarType, typename OperatorType>
using BroadcastLoopFuncType = void (*)(svbool_t, const BroadcastLoopArguments<InputScalarType, OutputScalarType, OperatorType> &);

template <typename InputVectorType, typename OutputVectorType, typename OperatorType,
          typename InputScalarType  = typename sve_scalar<InputVectorType>::type,
          typename OutputScalarType = typename sve_scalar<OutputVectorType>::type>
void elementwise_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
                    OperatorType op,
                    LoopFuncType<InputScalarType, OutputScalarType, OperatorType>          func,
                    BroadcastLoopFuncType<InputScalarType, OutputScalarType, OperatorType> broadcast_func)
{
    const auto all_true_pg = svptrue<InputScalarType>();

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

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

            int x = window_start_x;

            svbool_t pg = svwhilelt<InputScalarType>(x, window_end_x);
            do
            {
                broadcast_func(pg,
                {
                    op,
                    non_broadcast_input_ptr + x,
                    broadcast_value,
                    output_ptr + x,
                    !is_broadcast_input_2
                });
                x += svcnt<InputScalarType>();
                pg = svwhilelt<InputScalarType>(x, window_end_x);
            }
            while(svptest_any(all_true_pg, pg));
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

            int x = window_start_x;

            svbool_t pg = svwhilelt<InputScalarType>(x, window_end_x);
            do
            {
                func(pg,
                {
                    op,
                    input1_ptr + x,
                    input2_ptr + x,
                    output_ptr + x
                });
                x += svcnt<InputScalarType>();
                pg = svwhilelt<InputScalarType>(x, window_end_x);
            }
            while(svptest_any(all_true_pg, pg));
        },
        input1, input2, output);
    }
}

template <ArithmeticOperation op, typename ScalarType>
void elementwise_arithmetic_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    using VectorType = typename sve_vector<ScalarType>::type;

    elementwise_op<VectorType, VectorType, ArithmeticOperation>(in1, in2, out, window, op,
                                                                &arithmetic_op_loop<ScalarType, ScalarType>,
                                                                &arithmetic_op_broadcast_loop<ScalarType, ScalarType>);
}

template <ComparisonOperation op, typename InputScalarType, typename OutputScalarType = uint8_t>
void elementwise_comparison_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    static_assert(sizeof(InputScalarType) >= sizeof(OutputScalarType), "input data type's width should be equal to or greater than output data type's width");
    using InputVectorType  = typename sve_vector<InputScalarType>::type;
    using OutputVectorType = typename sve_vector<OutputScalarType>::type;

    elementwise_op<InputVectorType, OutputVectorType, ComparisonOperation>(in1, in2, out, window, op,
                                                                           &comparison_op_loop<InputScalarType, OutputScalarType>,
                                                                           &comparison_op_broadcast_loop<InputScalarType, OutputScalarType>);
}

template <>
svint32_t elementwise_pow<svint32_t>(svbool_t &pg, const svint32_t &a, const svint32_t &b)
{
    return svcvt_s32_z(pg, svpow_z(pg, svcvt_f32_z(pg, a), svcvt_f32_z(pg, b)));
}

template <>
svint32_t elementwise_div<svint32_t>(svbool_t &pg, const svint32_t &a, const svint32_t &b)
{
    return svcvt_s32_z(pg, svdiv_z(pg, svcvt_f32_z(pg, a), svcvt_f32_z(pg, b)));
}

template <>
svint16_t elementwise_div<svint16_t>(svbool_t &pg, const svint16_t &a, const svint16_t &b)
{
    ARM_COMPUTE_UNUSED(pg, a, b);
    ARM_COMPUTE_ERROR("Not supported");
}

template void elementwise_arithmetic_op<ArithmeticOperation::MAX, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::MAX, float32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::MAX, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::MAX, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_arithmetic_op<ArithmeticOperation::MIN, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::MIN, float32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::MIN, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::MIN, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_arithmetic_op<ArithmeticOperation::SQUARED_DIFF, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::SQUARED_DIFF, float32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::SQUARED_DIFF, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::SQUARED_DIFF, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_arithmetic_op<ArithmeticOperation::PRELU, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::PRELU, float32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::PRELU, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::PRELU, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_arithmetic_op<ArithmeticOperation::DIV, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::DIV, float32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::DIV, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::DIV, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_arithmetic_op<ArithmeticOperation::POWER, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::POWER, float32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::POWER, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_arithmetic_op<ArithmeticOperation::POWER, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_comparison_op<ComparisonOperation::Equal, float>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Equal, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Equal, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Equal, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Equal, uint8_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_comparison_op<ComparisonOperation::NotEqual, float>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::NotEqual, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::NotEqual, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::NotEqual, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::NotEqual, uint8_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_comparison_op<ComparisonOperation::Greater, float>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Greater, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Greater, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Greater, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Greater, uint8_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_comparison_op<ComparisonOperation::GreaterEqual, float>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::GreaterEqual, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::GreaterEqual, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::GreaterEqual, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::GreaterEqual, uint8_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_comparison_op<ComparisonOperation::Less, float>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Less, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Less, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Less, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::Less, uint8_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template void elementwise_comparison_op<ComparisonOperation::LessEqual, float>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::LessEqual, int32_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::LessEqual, float16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::LessEqual, int16_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void elementwise_comparison_op<ComparisonOperation::LessEqual, uint8_t>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_SVE) */