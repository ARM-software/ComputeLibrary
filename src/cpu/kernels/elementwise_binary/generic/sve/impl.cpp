/*
 * Copyright (c) 2021-2022 Arm Limited.
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

#include "src/cpu/kernels/elementwise_binary/generic/sve/impl.h"

#include "src/core/NEON/SVEMath.h"

#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
using namespace arm_compute::wrapper;

template <typename ScalarType>
void elementwise_arithmetic_op(
    const ITensor *in1, const ITensor *in2, ITensor *out, ArithmeticOperation op, const Window &window)
{
    using VectorType = typename sve_vector<ScalarType>::type;

    const auto all_true_pg = svptrue<ScalarType>();

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

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
                auto       output_ptr              = reinterpret_cast<ScalarType *>(output.ptr());
                const auto non_broadcast_input_ptr = reinterpret_cast<const ScalarType *>(non_broadcast_input.ptr());
                const ScalarType broadcast_value   = *reinterpret_cast<const ScalarType *>(broadcast_input.ptr());
                const auto       broadcast_vector  = svdup_n(broadcast_value);

                int x = window_start_x;

                svbool_t pg = svwhilelt<ScalarType>(x, window_end_x);
                do
                {
                    const auto non_broadcast_vector = svld1(pg, non_broadcast_input_ptr + x);
                    VectorType res{};

                    if (is_broadcast_input_2)
                    {
                        res = elementwise_arithmetic_op<typename sve_vector<ScalarType>::type>(pg, non_broadcast_vector,
                                                                                               broadcast_vector, op);
                    }
                    else
                    {
                        res = elementwise_arithmetic_op<typename sve_vector<ScalarType>::type>(
                            pg, broadcast_vector, non_broadcast_vector, op);
                    }
                    svst1(pg, output_ptr + x, res);

                    x += svcnt<ScalarType>();
                    pg = svwhilelt<ScalarType>(x, window_end_x);
                } while (svptest_any(all_true_pg, pg));
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
                auto       output_ptr = reinterpret_cast<ScalarType *>(output.ptr());
                const auto input1_ptr = reinterpret_cast<const ScalarType *>(input1.ptr());
                const auto input2_ptr = reinterpret_cast<const ScalarType *>(input2.ptr());

                int x = window_start_x;

                svbool_t pg = svwhilelt<ScalarType>(x, window_end_x);
                do
                {
                    const auto in1 = svld1(pg, input1_ptr + x);
                    const auto in2 = svld1(pg, input2_ptr + x);
                    const auto res = elementwise_arithmetic_op<typename sve_vector<ScalarType>::type>(pg, in1, in2, op);
                    svst1(pg, output_ptr + x, res);

                    x += svcnt<ScalarType>();
                    pg = svwhilelt<ScalarType>(x, window_end_x);
                } while (svptest_any(all_true_pg, pg));
            },
            input1, input2, output);
    }
}
template void elementwise_arithmetic_op<float32_t>(
    const ITensor *in1, const ITensor *in2, ITensor *out, const ArithmeticOperation op, const Window &window);
template void elementwise_arithmetic_op<float16_t>(
    const ITensor *in1, const ITensor *in2, ITensor *out, const ArithmeticOperation op, const Window &window);
template void elementwise_arithmetic_op<int16_t>(
    const ITensor *in1, const ITensor *in2, ITensor *out, const ArithmeticOperation op, const Window &window);
template void elementwise_arithmetic_op<int32_t>(
    const ITensor *in1, const ITensor *in2, ITensor *out, const ArithmeticOperation op, const Window &window);

template <typename InputScalarType, typename OutputScalarType>
void elementwise_comparison_op(
    const ITensor *in1, const ITensor *in2, ITensor *out, ComparisonOperation op, const Window &window)
{
    static_assert(sizeof(InputScalarType) >= sizeof(OutputScalarType),
                  "input data type's width should be equal to or greater than output data type's width");

    using OutputVectorType = typename sve_vector<OutputScalarType>::type;
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
                const auto broadcast_vector = svdup_n(broadcast_value);

                int x = window_start_x;

                svbool_t pg = svwhilelt<InputScalarType>(x, window_end_x);
                do
                {
                    const auto       non_broadcast_vector = svld1(pg, non_broadcast_input_ptr + x);
                    const svbool_t   output_pg            = narrow_to_byte_predicate<sizeof(InputScalarType)>(pg);
                    OutputVectorType res{};
                    if (is_broadcast_input_2)
                    {
                        res = elementwise_comparison_op<typename sve_vector<InputScalarType>::type,
                                                        typename sve_vector<OutputScalarType>::type>(
                            pg, non_broadcast_vector, broadcast_vector, op);
                    }
                    else
                    {
                        res = elementwise_comparison_op<typename sve_vector<InputScalarType>::type,
                                                        typename sve_vector<OutputScalarType>::type>(
                            pg, broadcast_vector, non_broadcast_vector, op);
                    }
                    svst1(output_pg, output_ptr + x, res);

                    x += svcnt<InputScalarType>();
                    pg = svwhilelt<InputScalarType>(x, window_end_x);
                } while (svptest_any(all_true_pg, pg));
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

                int x = window_start_x;

                svbool_t pg = svwhilelt<InputScalarType>(x, window_end_x);
                do
                {
                    const auto in1 = svld1(pg, input1_ptr + x);
                    const auto in2 = svld1(pg, input2_ptr + x);
                    const auto res =
                        elementwise_comparison_op<typename sve_vector<InputScalarType>::type,
                                                  typename sve_vector<OutputScalarType>::type>(pg, in1, in2, op);
                    const svbool_t output_pg = narrow_to_byte_predicate<sizeof(InputScalarType)>(pg);
                    svst1(output_pg, output_ptr + x, res);

                    x += svcnt<InputScalarType>();
                    pg = svwhilelt<InputScalarType>(x, window_end_x);
                } while (svptest_any(all_true_pg, pg));
            },
            input1, input2, output);
    }
}

template void elementwise_comparison_op<float32_t>(
    const ITensor *in1, const ITensor *in2, ITensor *out, const ComparisonOperation op, const Window &window);
template void elementwise_comparison_op<float16_t>(
    const ITensor *in1, const ITensor *in2, ITensor *out, const ComparisonOperation op, const Window &window);
template void elementwise_comparison_op<uint8_t>(
    const ITensor *in1, const ITensor *in2, ITensor *out, const ComparisonOperation op, const Window &window);
template void elementwise_comparison_op<int16_t>(
    const ITensor *in1, const ITensor *in2, ITensor *out, const ComparisonOperation op, const Window &window);
template void elementwise_comparison_op<int32_t>(
    const ITensor *in1, const ITensor *in2, ITensor *out, const ComparisonOperation op, const Window &window);

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

} // namespace cpu
} // namespace arm_compute
