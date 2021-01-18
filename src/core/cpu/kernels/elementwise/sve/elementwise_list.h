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
#ifndef SRC_CORE_SVE_KERNELS_ELEMENTWISE_LIST_H
#define SRC_CORE_SVE_KERNELS_ELEMENTWISE_LIST_H
#if defined(__ARM_FEATURE_SVE)
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/SVEMath.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include "src/core/NEON/wrapper/svtraits.h"
#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
namespace sve
{
using namespace arm_compute::wrapper;

template <typename VectorType>
inline VectorType elementwise_pow(svbool_t &pg, const VectorType &a, const VectorType &b)
{
    return svpow_z(pg, a, b);
}

template <>
inline svint32_t elementwise_pow<svint32_t>(svbool_t &pg, const svint32_t &a, const svint32_t &b)
{
    return svcvt_s32_z(pg, svpow_z(pg, svcvt_f32_z(pg, a), svcvt_f32_z(pg, b)));
}

template <typename VectorType>
inline VectorType elementwise_div(svbool_t &pg, const VectorType &a, const VectorType &b)
{
    return svdiv_z(pg, a, b);
}

template <>
inline svint32_t elementwise_div<svint32_t>(svbool_t &pg, const svint32_t &a, const svint32_t &b)
{
    return svcvt_s32_z(pg, svdiv_z(pg, svcvt_f32_z(pg, a), svcvt_f32_z(pg, b)));
}

template <typename VectorType>
inline VectorType elementwise_arithmetic_op(svbool_t &pg, const VectorType &a, const VectorType &b, ArithmeticOperation op)
{
    using ScalarType = typename sve_scalar<VectorType>::type;
    VectorType res{};

    switch(op)
    {
        case ArithmeticOperation::MAX:
            res = svmax_z(pg, a, b);
            break;
        case ArithmeticOperation::MIN:
            res = svmin_z(pg, a, b);
            break;
        case ArithmeticOperation::SQUARED_DIFF:
        {
            const auto tmp = svsub_z(pg, a, b);
            res            = svmul_z(pg, tmp, tmp);
            break;
        }
        case ArithmeticOperation::PRELU:
        {
            const auto zero = svdup_n(ScalarType(0));
            const auto tmp  = svmul_z(pg, a, b);
            const auto gt   = svcmpgt(pg, a, zero);
            res             = svsel(gt, a, tmp);
            break;
        }
        case ArithmeticOperation::DIV:
        {
            res = elementwise_div(pg, a, b);
            break;
        }
        case ArithmeticOperation::POWER:
        {
            res = elementwise_pow(pg, a, b);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return res;
}

template <uint32_t bytewidth>
inline svbool_t narrow_to_byte_predicate(svbool_t pg)
{
    const auto all_false = svpfalse();

    switch(bytewidth)
    {
        case 8:
            pg = svuzp1_b32(pg, all_false);
        /* fall through */
        case 4:
            pg = svuzp1_b16(pg, all_false);
        /* fall through */
        case 2:
            pg = svuzp1_b8(pg, all_false);
        /* fall through */
        default:
            break;
    }
    return pg;
}

template <typename InputVectorType, typename OutputVectorType>
inline OutputVectorType elementwise_comparison_op(svbool_t &pg, const InputVectorType &a, const InputVectorType &b, ComparisonOperation op)
{
    svbool_t selection_vector{};

    switch(op)
    {
        case ComparisonOperation::Equal:
            selection_vector = svcmpeq(pg, a, b);
            break;
        case ComparisonOperation::NotEqual:
            selection_vector = svcmpne(pg, a, b);
            break;
        case ComparisonOperation::Greater:
            selection_vector = svcmpgt(pg, a, b);
            break;
        case ComparisonOperation::GreaterEqual:
            selection_vector = svcmpge(pg, a, b);
            break;
        case ComparisonOperation::Less:
            selection_vector = svcmplt(pg, a, b);
            break;
        case ComparisonOperation::LessEqual:
            selection_vector = svcmple(pg, a, b);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    using InputScalarType = typename sve_scalar<InputVectorType>::type;
    selection_vector      = narrow_to_byte_predicate<sizeof(InputScalarType)>(selection_vector);

    using OutputScalarType  = typename sve_scalar<OutputVectorType>::type;
    const auto false_vector = svdup_n(static_cast<OutputScalarType>((uint32_t)0));
    const auto true_vector  = svdup_n(static_cast<OutputScalarType>(~(uint32_t)0));
    auto       ret          = svsel(selection_vector, true_vector, false_vector);

    return ret;
}

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
inline void arithmetic_op_loop(svbool_t pg, const LoopArguments<InputScalarType, OutputScalarType, ArithmeticOperation> &args)
{
    const auto in1 = svld1(pg, args.input1_ptr);
    const auto in2 = svld1(pg, args.input2_ptr);
    const auto res = elementwise_arithmetic_op<typename sve_vector<InputScalarType>::type>(pg, in1, in2, args.op);
    svst1(pg, args.output_ptr, res);
}

template <typename InputScalarType, typename OutputScalarType>
inline void arithmetic_op_broadcast_loop(svbool_t pg, const BroadcastLoopArguments<InputScalarType, OutputScalarType, ArithmeticOperation> &args)
{
    const auto non_broadcast_vector = svld1(pg, args.input1_ptr);
    const auto broadcast_vector     = svdup_n(args.broadcast_value);
    const auto in1                  = args.reorder ? broadcast_vector : non_broadcast_vector;
    const auto in2                  = args.reorder ? non_broadcast_vector : broadcast_vector;
    const auto res                  = elementwise_arithmetic_op<typename sve_vector<InputScalarType>::type>(pg, in1, in2, args.op);
    svst1(pg, args.output_ptr, res);
}

template <typename InputScalarType, typename OutputScalarType>
inline void comparison_op_loop(svbool_t pg, const LoopArguments<InputScalarType, OutputScalarType, ComparisonOperation> &args)
{
    const auto     in1       = svld1(pg, args.input1_ptr);
    const auto     in2       = svld1(pg, args.input2_ptr);
    const auto     res       = elementwise_comparison_op<typename sve_vector<InputScalarType>::type, typename sve_vector<OutputScalarType>::type>(pg, in1, in2, args.op);
    const svbool_t output_pg = narrow_to_byte_predicate<sizeof(InputScalarType)>(pg);
    svst1(output_pg, args.output_ptr, res);
}

template <typename InputScalarType, typename OutputScalarType>
inline void comparison_op_broadcast_loop(svbool_t pg, const BroadcastLoopArguments<InputScalarType, OutputScalarType, ComparisonOperation> &args)
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

} // namespace sve
} // namespace cpu
} // namespace arm_compute
#endif // defined(__ARM_FEATURE_SVE)
#endif /* SRC_CORE_SVE_KERNELS_ELEMENTWISE_LIST_H */
