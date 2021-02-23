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
#ifndef SRC_CORE_SVE_KERNELS_ELEMENTWISE_QUANTIZED_LIST_H
#define SRC_CORE_SVE_KERNELS_ELEMENTWISE_QUANTIZED_LIST_H

#if defined(__ARM_FEATURE_SVE2)

#include "src/core/cpu/kernels/elementwise/sve/elementwise_list.h"

namespace arm_compute
{
namespace cpu
{
namespace sve
{
using namespace arm_compute::wrapper;

template <typename InputScalarType, typename OutputScalarType, typename OperatorType>
struct QuantizedLoopArguments
{
    OperatorType           op;
    const InputScalarType *input1_ptr;
    const InputScalarType *input2_ptr;
    OutputScalarType      *output_ptr;

    const svint32_t   &in1_offset;
    const svint32_t   &in2_offset;
    const svint32_t   &out_offset;
    const svfloat32_t &in1_scale;
    const svfloat32_t &in2_scale;
    const svfloat32_t &out_scale;
};

template <typename InputScalarType, typename OutputScalarType, typename OperatorType>
struct BroadcastQuantizedLoopArguments
{
    OperatorType           op;
    const InputScalarType *input1_ptr;
    float                  broadcast_value;
    OutputScalarType      *output_ptr;
    bool                   reorder;

    const svint32_t   &in1_offset;
    const svint32_t   &out_offset;
    const svfloat32_t &in1_scale;
    const svfloat32_t &out_scale;
};

svfloat32x4_t load_quantized(const int8_t *ptr, svbool_t pg, const svint32_t &offset, const svfloat32_t &scale)
{
    auto x = svld1(pg, ptr);

    const auto widened = svcreate4(
                             svmovlb(svmovlb(x)),
                             svmovlt(svmovlb(x)),
                             svmovlb(svmovlt(x)),
                             svmovlt(svmovlt(x)));

    pg = svptrue_b8();

    return svcreate4(
               svmul_z(pg, svcvt_f32_z(pg, svsub_z(pg, svget4(widened, 0), offset)), scale),
               svmul_z(pg, svcvt_f32_z(pg, svsub_z(pg, svget4(widened, 1), offset)), scale),
               svmul_z(pg, svcvt_f32_z(pg, svsub_z(pg, svget4(widened, 2), offset)), scale),
               svmul_z(pg, svcvt_f32_z(pg, svsub_z(pg, svget4(widened, 3), offset)), scale));
}

svfloat32x4_t load_quantized(const uint8_t *ptr, svbool_t pg, const svint32_t &offset, const svfloat32_t &scale)
{
    auto x = svld1(pg, ptr);

    //vprint(x);

    const auto widened = svcreate4(
                             svmovlb(svmovlb(x)),
                             svmovlt(svmovlb(x)),
                             svmovlb(svmovlt(x)),
                             svmovlt(svmovlt(x)));

    pg = svptrue_b8();

    return svcreate4(
               svmul_z(pg, svcvt_f32_z(pg, svsub_z(pg, svreinterpret_s32(svget4(widened, 0)), offset)), scale),
               svmul_z(pg, svcvt_f32_z(pg, svsub_z(pg, svreinterpret_s32(svget4(widened, 1)), offset)), scale),
               svmul_z(pg, svcvt_f32_z(pg, svsub_z(pg, svreinterpret_s32(svget4(widened, 2)), offset)), scale),
               svmul_z(pg, svcvt_f32_z(pg, svsub_z(pg, svreinterpret_s32(svget4(widened, 3)), offset)), scale));
}

void store_quantized(uint8_t *ptr, svbool_t pg, svfloat32x4_t data, const svint32_t &offset, const svfloat32_t &inv_scale)
{
    const auto quantized = svcreate4(
                               svadd_z(pg, svcvt_s32_z(pg, svrinta_z(pg, svmul_z(pg, svget4(data, 0), inv_scale))), offset),
                               svadd_z(pg, svcvt_s32_z(pg, svrinta_z(pg, svmul_z(pg, svget4(data, 1), inv_scale))), offset),
                               svadd_z(pg, svcvt_s32_z(pg, svrinta_z(pg, svmul_z(pg, svget4(data, 2), inv_scale))), offset),
                               svadd_z(pg, svcvt_s32_z(pg, svrinta_z(pg, svmul_z(pg, svget4(data, 3), inv_scale))), offset));

    const auto narrowed_bottom = svqxtunt(svqxtunb(svget4(quantized, 0)), svget4(quantized, 1));
    const auto narrowed_top    = svqxtunt(svqxtunb(svget4(quantized, 2)), svget4(quantized, 3));
    const auto narrowed        = svqxtnt(svqxtnb(narrowed_bottom), narrowed_top);
    svst1(pg, ptr, narrowed);
}

void store_quantized(int8_t *ptr, svbool_t pg, svfloat32x4_t data, const svint32_t &offset, const svfloat32_t &inv_scale)
{
    const auto quantized = svcreate4(
                               svadd_z(pg, svcvt_s32_z(pg, svrinta_z(pg, svmul_z(pg, svget4(data, 0), inv_scale))), offset),
                               svadd_z(pg, svcvt_s32_z(pg, svrinta_z(pg, svmul_z(pg, svget4(data, 1), inv_scale))), offset),
                               svadd_z(pg, svcvt_s32_z(pg, svrinta_z(pg, svmul_z(pg, svget4(data, 2), inv_scale))), offset),
                               svadd_z(pg, svcvt_s32_z(pg, svrinta_z(pg, svmul_z(pg, svget4(data, 3), inv_scale))), offset));

    const auto narrowed_bottom = svqxtnt(svqxtnb(svget4(quantized, 0)), svget4(quantized, 1));
    const auto narrowed_top    = svqxtnt(svqxtnb(svget4(quantized, 2)), svget4(quantized, 3));
    const auto narrowed        = svqxtnt(svqxtnb(narrowed_bottom), narrowed_top);

    svst1(pg, ptr, narrowed);
}

template <typename InputScalarType, typename OutputScalarType>
inline void arithmetic_op_quantized_loop(svbool_t pg, const QuantizedLoopArguments<InputScalarType, OutputScalarType, ArithmeticOperation> &args)
{
    const auto in1 = load_quantized(args.input1_ptr, pg, args.in1_offset, args.in1_scale);
    const auto in2 = load_quantized(args.input2_ptr, pg, args.in2_offset, args.in2_scale);

    const auto result = svcreate4(
                            elementwise_arithmetic_op<svfloat32_t>(pg, svget4(in1, 0), svget4(in2, 0), args.op),
                            elementwise_arithmetic_op<svfloat32_t>(pg, svget4(in1, 1), svget4(in2, 1), args.op),
                            elementwise_arithmetic_op<svfloat32_t>(pg, svget4(in1, 2), svget4(in2, 2), args.op),
                            elementwise_arithmetic_op<svfloat32_t>(pg, svget4(in1, 3), svget4(in2, 3), args.op));

    store_quantized(args.output_ptr, pg, result, args.out_offset, args.out_scale);
}

template <typename InputScalarType, typename OutputScalarType>
inline void arithmetic_op_broadcast_quantized_loop(svbool_t pg, const BroadcastQuantizedLoopArguments<InputScalarType, OutputScalarType, ArithmeticOperation> &args)
{
    const auto in1 = load_quantized(args.input1_ptr, pg, args.in1_offset, args.in1_scale);
    const auto in2 = svcreate4(
                         svdup_n(args.broadcast_value), svdup_n(args.broadcast_value), svdup_n(args.broadcast_value), svdup_n(args.broadcast_value));

    const auto &af = args.reorder ? in2 : in1;
    const auto &bf = args.reorder ? in1 : in2;

    const auto result = svcreate4(
                            elementwise_arithmetic_op<svfloat32_t>(pg, svget4(af, 0), svget4(bf, 0), args.op),
                            elementwise_arithmetic_op<svfloat32_t>(pg, svget4(af, 1), svget4(bf, 1), args.op),
                            elementwise_arithmetic_op<svfloat32_t>(pg, svget4(af, 2), svget4(bf, 2), args.op),
                            elementwise_arithmetic_op<svfloat32_t>(pg, svget4(af, 3), svget4(bf, 3), args.op));

    store_quantized(args.output_ptr, pg, result, args.out_offset, args.out_scale);
}

template <typename InputScalarType, typename OutputScalarType>
inline void comparison_op_quantized_loop(svbool_t pg, const QuantizedLoopArguments<InputScalarType, OutputScalarType, ComparisonOperation> &args)
{
    const auto in1 = load_quantized(args.input1_ptr, pg, args.in1_offset, args.in1_scale);
    const auto in2 = load_quantized(args.input2_ptr, pg, args.in2_offset, args.in2_scale);

    using OutputVectorType = typename sve_vector<OutputScalarType>::type;

    const auto result = svcreate4(
                            elementwise_comparison_op<svfloat32_t, OutputVectorType>(pg, svget4(in1, 0), svget4(in2, 0), args.op),
                            elementwise_comparison_op<svfloat32_t, OutputVectorType>(pg, svget4(in1, 1), svget4(in2, 1), args.op),
                            elementwise_comparison_op<svfloat32_t, OutputVectorType>(pg, svget4(in1, 2), svget4(in2, 2), args.op),
                            elementwise_comparison_op<svfloat32_t, OutputVectorType>(pg, svget4(in1, 3), svget4(in2, 3), args.op));

    const auto zipped_bottom = svzip1(svget4(result, 0), svget4(result, 1));
    const auto zipped_top    = svzip1(svget4(result, 2), svget4(result, 3));
    const auto zipped        = svzip1(zipped_bottom, zipped_top);
    svst1(pg, args.output_ptr, zipped);
}

template <typename InputScalarType, typename OutputScalarType>
inline void comparison_op_broadcast_quantized_loop(svbool_t pg, const BroadcastQuantizedLoopArguments<InputScalarType, OutputScalarType, ComparisonOperation> &args)
{
    const auto in1 = load_quantized(args.input1_ptr, pg, args.in1_offset, args.in1_scale);
    const auto in2 = svcreate4(
                         svdup_n(args.broadcast_value), svdup_n(args.broadcast_value), svdup_n(args.broadcast_value), svdup_n(args.broadcast_value));

    const auto &af = args.reorder ? in2 : in1;
    const auto &bf = args.reorder ? in1 : in2;

    using OutputVectorType = typename sve_vector<OutputScalarType>::type;

    const auto result = svcreate4(
                            elementwise_comparison_op<svfloat32_t, OutputVectorType>(pg, svget4(af, 0), svget4(bf, 0), args.op),
                            elementwise_comparison_op<svfloat32_t, OutputVectorType>(pg, svget4(af, 1), svget4(bf, 1), args.op),
                            elementwise_comparison_op<svfloat32_t, OutputVectorType>(pg, svget4(af, 2), svget4(bf, 2), args.op),
                            elementwise_comparison_op<svfloat32_t, OutputVectorType>(pg, svget4(af, 3), svget4(bf, 3), args.op));

    const auto zipped_bottom = svzip1(svget4(result, 0), svget4(result, 1));
    const auto zipped_top    = svzip1(svget4(result, 2), svget4(result, 3));
    const auto zipped        = svzip1(zipped_bottom, zipped_top);
    svst1(pg, args.output_ptr, zipped);
}

template <typename InputScalarType, typename OutputScalarType, typename OperatorType>
using LoopQuantizedFuncType = void (*)(svbool_t, const QuantizedLoopArguments<InputScalarType, OutputScalarType, OperatorType> &);

template <typename InputScalarType, typename OutputScalarType, typename OperatorType>
using BroadcastQuantizedLoopFuncType = void (*)(svbool_t, const BroadcastQuantizedLoopArguments<InputScalarType, OutputScalarType, OperatorType> &);

template <typename InputVectorType, typename OutputVectorType, typename OperatorType,
          typename InputScalarType  = typename sve_scalar<InputVectorType>::type,
          typename OutputScalarType = typename sve_scalar<OutputVectorType>::type>
void elementwise_quantized_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
                              OperatorType op,
                              LoopQuantizedFuncType<InputScalarType, OutputScalarType, OperatorType>          func,
                              BroadcastQuantizedLoopFuncType<InputScalarType, OutputScalarType, OperatorType> broadcast_func)
{
    const auto all_true_pg = wrapper::svptrue<InputScalarType>();

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = in1->info()->tensor_shape().x() != in2->info()->tensor_shape().x();

    const auto output_voffset = svdup_n(out->info()->quantization_info().uniform().offset);
    const auto output_vscale  = svdup_n(1.f / out->info()->quantization_info().uniform().scale);

    if(is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? in2 : in1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;

        const auto non_broadcast_qinfo = is_broadcast_input_2 ? in1->info()->quantization_info() : in2->info()->quantization_info();
        const auto broadcast_qinfo     = is_broadcast_input_2 ? in2->info()->quantization_info() : in1->info()->quantization_info();

        const auto non_broadcast_voffset = svdup_n(non_broadcast_qinfo.uniform().offset);
        const auto non_broadcast_vscale  = svdup_n(non_broadcast_qinfo.uniform().scale);

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

            svbool_t pg = wrapper::svwhilelt<InputScalarType>(x, window_end_x);
            do
            {
                const auto args = BroadcastQuantizedLoopArguments<InputScalarType, OutputScalarType, OperatorType>
                {
                    op,
                    non_broadcast_input_ptr + x,
                    Qasymm8QuantizationHelper<InputScalarType>::dequantize(broadcast_value, broadcast_qinfo),
                    output_ptr + x,
                    !is_broadcast_input_2,
                    non_broadcast_voffset, output_voffset,
                    non_broadcast_vscale, output_vscale
                };
                broadcast_func(pg, args);
                x += wrapper::svcnt<InputScalarType>();
                pg = wrapper::svwhilelt<InputScalarType>(x, window_end_x);
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

        const auto in1_voffset = svdup_n(in1->info()->quantization_info().uniform().offset);
        const auto in1_vscale  = svdup_n(in1->info()->quantization_info().uniform().scale);

        const auto in2_voffset = svdup_n(in2->info()->quantization_info().uniform().offset);
        const auto in2_vscale  = svdup_n(in2->info()->quantization_info().uniform().scale);

        execute_window_loop(win, [&](const Coordinates &)
        {
            auto       output_ptr = reinterpret_cast<OutputScalarType *>(output.ptr());
            const auto input1_ptr = reinterpret_cast<const InputScalarType *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const InputScalarType *>(input2.ptr());

            int x = window_start_x;

            svbool_t pg = wrapper::svwhilelt<InputScalarType>(x, window_end_x);
            do
            {
                const auto args = QuantizedLoopArguments<InputScalarType, OutputScalarType, OperatorType>
                {
                    op,
                    input1_ptr + x,
                    input2_ptr + x,
                    output_ptr + x,
                    in1_voffset, in2_voffset, output_voffset,
                    in1_vscale, in2_vscale, output_vscale
                };
                func(pg, args);
                x += wrapper::svcnt<InputScalarType>();
                pg = wrapper::svwhilelt<InputScalarType>(x, window_end_x);
            }
            while(svptest_any(all_true_pg, pg));
        },
        input1, input2, output);
    }
}

template <ArithmeticOperation op, typename ScalarType>
void elementwise_arithmetic_quantized_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    using VectorType = typename sve_vector<ScalarType>::type;
    elementwise_quantized_op<VectorType, VectorType, ArithmeticOperation>(in1, in2, out, window, op,
                                                                          &arithmetic_op_quantized_loop<ScalarType, ScalarType>,
                                                                          &arithmetic_op_broadcast_quantized_loop<ScalarType, ScalarType>);
}

template <ComparisonOperation op, typename InputScalarType, typename OutputScalarType = uint8_t>
void elementwise_comparison_quantized_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    static_assert(sizeof(InputScalarType) >= sizeof(OutputScalarType), "input data type's width should be equal to or greater than output data type's width");
    using InputVectorType  = typename sve_vector<InputScalarType>::type;
    using OutputVectorType = typename sve_vector<OutputScalarType>::type;
    elementwise_quantized_op<InputVectorType, OutputVectorType, ComparisonOperation>(in1, in2, out, window, op,
                                                                                     &comparison_op_quantized_loop<InputScalarType, OutputScalarType>,
                                                                                     &comparison_op_broadcast_quantized_loop<InputScalarType, OutputScalarType>);
}

} // namespace sve
} // namespace cpu
} // namespace arm_compute

#endif /* defined(__ARM_FEATURE_SVE2) */
#endif /* SRC_CORE_SVE_KERNELS_ELEMENTWISE_QUANTIZED_LIST_H */