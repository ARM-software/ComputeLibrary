/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEReductionOperationKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace
{
template <class F>
class Reducer
{
public:
    static void reduceX(const Window &window, const ITensor *input, ITensor *output, F f)
    {
        // Set out window
        Window out_window(window);
        out_window.set(Window::DimX, Window::Dimension(0, 0, 0));

        // Get first input and output slices
        Window in_slice  = window.first_slice_window_1D();
        Window out_slice = out_window.first_slice_window_1D();

        do
        {
            Iterator in(input, in_slice);
            Iterator out(output, out_slice);

            f(in, out, in_slice, out_slice, *input->info());
        }
        while(window.slide_window_slice_1D(in_slice) && out_window.slide_window_slice_1D(out_slice));
    }
    static void reduceY(const Window &window, const ITensor *input, ITensor *output, F f)
    {
        // Set in window
        Window in_window(window);

        in_window.set(Window::DimY, Window::Dimension(0, 1, 1));

        // Get first input and output slices
        Window in_slice  = in_window.first_slice_window_2D();
        Window out_slice = window.first_slice_window_2D();

        do
        {
            Iterator in(input, in_slice);
            Iterator out(output, out_slice);

            f(in, out, in_slice, out_slice, *input->info(), 1);
        }
        while(in_window.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(out_slice));
    }
    static void reduceZ(const Window &window, const ITensor *input, ITensor *output, F f)
    {
        // Set in window
        Window in_window(window);

        in_window.set(Window::DimZ, Window::Dimension(0, 1, 1));

        // Get first input and output slices
        Window in_slice  = in_window.first_slice_window_3D();
        Window out_slice = window.first_slice_window_3D();

        do
        {
            Iterator in(input, in_slice);
            Iterator out(output, out_slice);

            f(in, out, in_slice, out_slice, *input->info(), 2);
        }
        while(in_window.slide_window_slice_3D(in_slice) && window.slide_window_slice_3D(out_slice));
    }
    static void reduceW(const Window &window, const ITensor *input, ITensor *output, F f)
    {
        // Set in/out window
        Window in_window(window);
        Window out_window(window);

        in_window.set(3, Window::Dimension(0, 1, 1));
        out_window.set(3, Window::Dimension(0, 1, 1));

        // Get first input and output slices
        Window in_slice  = in_window.first_slice_window_4D();
        Window out_slice = out_window.first_slice_window_4D();

        do
        {
            Iterator in(input, in_slice);
            Iterator out(output, out_slice);

            f(in, out, in_slice, out_slice, *input->info(), 3);
        }
        while(in_window.slide_window_slice_4D(in_slice) && out_window.slide_window_slice_4D(out_slice));
    }
};

template <typename T, int S, ReductionOperation op>
struct RedOpX
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice, const TensorInfo &in_info)
    {
        ARM_COMPUTE_UNUSED(out_slice);
        auto vec_sum_value = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto in_ptr       = reinterpret_cast<const T *>(input.ptr());
            const auto vec_elements = wrapper::vloadq(in_ptr);

            if(op == ReductionOperation::SUM_SQUARE)
            {
                vec_sum_value = wrapper::vadd(wrapper::vmul(vec_elements, vec_elements), vec_sum_value);
            }
            else
            {
                vec_sum_value = wrapper::vadd(vec_elements, vec_sum_value);
            }
        },
        input);

        auto carry_addition = wrapper::vpadd(wrapper::vgethigh(vec_sum_value), wrapper::vgetlow(vec_sum_value));
        for(int i = 0; i < S / 4; ++i)
        {
            carry_addition = wrapper::vpadd(carry_addition, carry_addition);
        }

        auto res = wrapper::vgetlane(carry_addition, 0);
        if(op == ReductionOperation::MEAN_SUM)
        {
            res /= in_info.dimension(0);
        }

        *(reinterpret_cast<T *>(output.ptr())) = res;
    }
};

template <ReductionOperation op>
struct RedOpX_qasymm8
{
    inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice, const TensorInfo &in_info)
    {
        ARM_COMPUTE_UNUSED(out_slice);
        auto vec_sum_value1 = vdupq_n_u32(static_cast<uint32_t>(0.f));
        auto vec_sum_value2 = vdupq_n_u32(static_cast<uint32_t>(0.f));
        auto vec_sum_value3 = vdupq_n_u32(static_cast<uint32_t>(0.f));
        auto vec_sum_value4 = vdupq_n_u32(static_cast<uint32_t>(0.f));

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto vec_elements = wrapper::vloadq(input.ptr());

            const auto temp16x8t_1 = wrapper::vmovl(wrapper::vgetlow(vec_elements));
            const auto temp16x8t_2 = wrapper::vmovl(wrapper::vgethigh(vec_elements));

            const auto temp32x4t_1 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_1));
            const auto temp32x4t_2 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_1));
            const auto temp32x4t_3 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_2));
            const auto temp32x4t_4 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_2));

            vec_sum_value1 = wrapper::vadd(temp32x4t_1, vec_sum_value1);
            vec_sum_value2 = wrapper::vadd(temp32x4t_2, vec_sum_value2);
            vec_sum_value3 = wrapper::vadd(temp32x4t_3, vec_sum_value3);
            vec_sum_value4 = wrapper::vadd(temp32x4t_4, vec_sum_value4);
        },
        input);

        auto carry_addition = wrapper::vadd(vec_sum_value1, vec_sum_value2);
        carry_addition      = wrapper::vadd(carry_addition, vec_sum_value3);
        carry_addition      = wrapper::vadd(carry_addition, vec_sum_value4);

        auto carry_paddition = wrapper::vpadd(wrapper::vgethigh(carry_addition), wrapper::vgetlow(carry_addition));
        carry_paddition      = wrapper::vpadd(carry_paddition, carry_paddition);
        auto res             = wrapper::vgetlane(carry_paddition, 0);

        if(op == ReductionOperation::MEAN_SUM)
        {
            res /= in_info.dimension(0);
        }

        *(output.ptr()) = static_cast<uint8_t>(res);
    }
};

template <typename T, int S, ReductionOperation op>
struct RedOpYZW
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice, const TensorInfo &in_info, int axis)
    {
        ARM_COMPUTE_UNUSED(out_slice);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            auto vec_sum_value = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
            for(unsigned int dim = 0; dim < in_info.dimension(axis); ++dim)
            {
                T *in_ptr;
                switch(axis)
                {
                    case 1:
                        in_ptr = reinterpret_cast<T *>(input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, dim)));
                        break;
                    case 2:
                        in_ptr = reinterpret_cast<T *>(input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, 0, dim)));
                        break;
                    case 3:
                        in_ptr = reinterpret_cast<T *>(input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, 0, 0, dim)));
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Not supported");
                }
                const auto vec_elements = wrapper::vloadq(in_ptr);

                if(op == ReductionOperation::SUM_SQUARE)
                {
                    vec_sum_value = wrapper::vadd(wrapper::vmul(vec_elements, vec_elements), vec_sum_value);
                }
                else
                {
                    vec_sum_value = wrapper::vadd(vec_elements, vec_sum_value);
                }
            }

            if(op == ReductionOperation::MEAN_SUM)
            {
                auto vec_width_inv = wrapper::vinv(wrapper::vdup_n(static_cast<T>(in_info.dimension(axis)), ExactTagType{}));
                vec_sum_value      = wrapper::vmul(vec_sum_value, vec_width_inv);
            }

            wrapper::vstore(reinterpret_cast<T *>(output.ptr()), vec_sum_value);
        },
        input, output);
    }
};

template <ReductionOperation op>
struct RedOpYZW_qasymm8
{
    inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice, const TensorInfo &in_info, int axis)
    {
        ARM_COMPUTE_UNUSED(out_slice);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            auto vec_sum_value1 = vdupq_n_u32(static_cast<uint32_t>(0.f));
            auto vec_sum_value2 = vdupq_n_u32(static_cast<uint32_t>(0.f));
            auto vec_sum_value3 = vdupq_n_u32(static_cast<uint32_t>(0.f));
            auto vec_sum_value4 = vdupq_n_u32(static_cast<uint32_t>(0.f));
            for(unsigned int dim = 0; dim < in_info.dimension(axis); ++dim)
            {
                uint8_t *in_ptr;
                switch(axis)
                {
                    case 1:
                        in_ptr = input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, dim));
                        break;
                    case 2:
                        in_ptr = input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, 0, dim));
                        break;
                    case 3:
                        in_ptr = input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, 0, 0, dim));
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Not supported");
                }
                const auto vec_elements = wrapper::vloadq(in_ptr);

                const auto temp16x8t_1 = wrapper::vmovl(wrapper::vgetlow(vec_elements));
                const auto temp16x8t_2 = wrapper::vmovl(wrapper::vgethigh(vec_elements));

                const auto temp32x4t_1 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_1));
                const auto temp32x4t_2 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_1));
                const auto temp32x4t_3 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_2));
                const auto temp32x4t_4 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_2));

                vec_sum_value1 = wrapper::vadd(temp32x4t_1, vec_sum_value1);
                vec_sum_value2 = wrapper::vadd(temp32x4t_2, vec_sum_value2);
                vec_sum_value3 = wrapper::vadd(temp32x4t_3, vec_sum_value3);
                vec_sum_value4 = wrapper::vadd(temp32x4t_4, vec_sum_value4);
            }

            if(op == ReductionOperation::MEAN_SUM)
            {
                const auto vec_width_inv    = wrapper::vinv(vdupq_n_f32(in_info.dimension(axis)));
                const auto vec_sum_value1_f = wrapper::vmul(vcvtq_f32_u32(vec_sum_value1), vec_width_inv);
                const auto vec_sum_value2_f = wrapper::vmul(vcvtq_f32_u32(vec_sum_value2), vec_width_inv);
                const auto vec_sum_value3_f = wrapper::vmul(vcvtq_f32_u32(vec_sum_value3), vec_width_inv);
                const auto vec_sum_value4_f = wrapper::vmul(vcvtq_f32_u32(vec_sum_value4), vec_width_inv);

                vec_sum_value1 = vcvtq_u32_f32(vec_sum_value1_f);
                vec_sum_value2 = vcvtq_u32_f32(vec_sum_value2_f);
                vec_sum_value3 = vcvtq_u32_f32(vec_sum_value3_f);
                vec_sum_value4 = vcvtq_u32_f32(vec_sum_value4_f);
            }

            const auto temp16x8t_1 = vcombine_u16(wrapper::vqmovn(vec_sum_value1), wrapper::vqmovn(vec_sum_value2));
            const auto temp16x8t_2 = vcombine_u16(wrapper::vqmovn(vec_sum_value3), wrapper::vqmovn(vec_sum_value4));
            auto       res         = vcombine_u8(wrapper::vqmovn(temp16x8t_1), wrapper::vqmovn(temp16x8t_2));
            wrapper::vstore(output.ptr(), res);
        },
        input, output);
    }
};

void reduce_sumsq(const Window &window, const ITensor *input, ITensor *output, unsigned int axis)
{
    switch(axis)
    {
        case 0:
            switch(input->info()->data_type())
            {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpX<float16_t, 8, ReductionOperation::SUM_SQUARE>>::reduceX(window, input, output, RedOpX<float16_t, 8, ReductionOperation::SUM_SQUARE>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpX<float, 4, ReductionOperation::SUM_SQUARE>>::reduceX(window, input, output, RedOpX<float, 4, ReductionOperation::SUM_SQUARE>());
                case DataType::QASYMM8:
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 1:
            switch(input->info()->data_type())
            {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8, ReductionOperation::SUM_SQUARE>>::reduceY(window, input, output, RedOpYZW<float16_t, 8, ReductionOperation::SUM_SQUARE>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4, ReductionOperation::SUM_SQUARE>>::reduceY(window, input, output, RedOpYZW<float, 4, ReductionOperation::SUM_SQUARE>());
                case DataType::QASYMM8:
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 2:
            switch(input->info()->data_type())
            {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8, ReductionOperation::SUM_SQUARE>>::reduceZ(window, input, output, RedOpYZW<float16_t, 8, ReductionOperation::SUM_SQUARE>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4, ReductionOperation::SUM_SQUARE>>::reduceZ(window, input, output, RedOpYZW<float, 4, ReductionOperation::SUM_SQUARE>());
                case DataType::QASYMM8:
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 3:
            switch(input->info()->data_type())
            {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8, ReductionOperation::SUM_SQUARE>>::reduceW(window, input, output, RedOpYZW<float16_t, 8, ReductionOperation::SUM_SQUARE>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4, ReductionOperation::SUM_SQUARE>>::reduceW(window, input, output, RedOpYZW<float, 4, ReductionOperation::SUM_SQUARE>());
                case DataType::QASYMM8:
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction axis");
    }
}

void reduce_sum(const Window &window, const ITensor *input, ITensor *output, unsigned int axis)
{
    switch(axis)
    {
        case 0:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpX_qasymm8<ReductionOperation::SUM>>::reduceX(window, input, output, RedOpX_qasymm8<ReductionOperation::SUM>());
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpX<float16_t, 8, ReductionOperation::SUM>>::reduceX(window, input, output, RedOpX<float16_t, 8, ReductionOperation::SUM>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpX<float, 4, ReductionOperation::SUM>>::reduceX(window, input, output, RedOpX<float, 4, ReductionOperation::SUM>());
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 1:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpYZW_qasymm8<ReductionOperation::SUM>>::reduceY(window, input, output, RedOpYZW_qasymm8<ReductionOperation::SUM>());
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8, ReductionOperation::SUM>>::reduceY(window, input, output, RedOpYZW<float16_t, 8, ReductionOperation::SUM>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4, ReductionOperation::SUM>>::reduceY(window, input, output, RedOpYZW<float, 4, ReductionOperation::SUM>());
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 2:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpYZW_qasymm8<ReductionOperation::SUM>>::reduceZ(window, input, output, RedOpYZW_qasymm8<ReductionOperation::SUM>());
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8, ReductionOperation::SUM>>::reduceZ(window, input, output, RedOpYZW<float16_t, 8, ReductionOperation::SUM>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4, ReductionOperation::SUM>>::reduceZ(window, input, output, RedOpYZW<float, 4, ReductionOperation::SUM>());
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 3:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpYZW_qasymm8<ReductionOperation::SUM>>::reduceW(window, input, output, RedOpYZW_qasymm8<ReductionOperation::SUM>());
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8, ReductionOperation::SUM>>::reduceW(window, input, output, RedOpYZW<float16_t, 8, ReductionOperation::SUM>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4, ReductionOperation::SUM>>::reduceW(window, input, output, RedOpYZW<float, 4, ReductionOperation::SUM>());
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction axis");
    }
}
void reduce_mean_sum(const Window &window, const ITensor *input, ITensor *output, unsigned int axis)
{
    switch(axis)
    {
        case 0:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpX_qasymm8<ReductionOperation::MEAN_SUM>>::reduceX(window, input, output, RedOpX_qasymm8<ReductionOperation::MEAN_SUM>());
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpX<float16_t, 8, ReductionOperation::MEAN_SUM>>::reduceX(window, input, output, RedOpX<float16_t, 8, ReductionOperation::MEAN_SUM>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpX<float, 4, ReductionOperation::MEAN_SUM>>::reduceX(window, input, output, RedOpX<float, 4, ReductionOperation::MEAN_SUM>());
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 1:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpYZW_qasymm8<ReductionOperation::MEAN_SUM>>::reduceY(window, input, output, RedOpYZW_qasymm8<ReductionOperation::MEAN_SUM>());
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8, ReductionOperation::MEAN_SUM>>::reduceY(window, input, output, RedOpYZW<float16_t, 8, ReductionOperation::MEAN_SUM>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4, ReductionOperation::MEAN_SUM>>::reduceY(window, input, output, RedOpYZW<float, 4, ReductionOperation::MEAN_SUM>());
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 2:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpYZW_qasymm8<ReductionOperation::MEAN_SUM>>::reduceZ(window, input, output, RedOpYZW_qasymm8<ReductionOperation::MEAN_SUM>());
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8, ReductionOperation::MEAN_SUM>>::reduceZ(window, input, output, RedOpYZW<float16_t, 8, ReductionOperation::MEAN_SUM>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4, ReductionOperation::MEAN_SUM>>::reduceZ(window, input, output, RedOpYZW<float, 4, ReductionOperation::MEAN_SUM>());
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 3:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpYZW_qasymm8<ReductionOperation::MEAN_SUM>>::reduceW(window, input, output, RedOpYZW_qasymm8<ReductionOperation::MEAN_SUM>());
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8, ReductionOperation::MEAN_SUM>>::reduceW(window, input, output, RedOpYZW<float16_t, 8, ReductionOperation::MEAN_SUM>());
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4, ReductionOperation::MEAN_SUM>>::reduceW(window, input, output, RedOpYZW<float, 4, ReductionOperation::MEAN_SUM>());
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction axis");
    }
}

TensorShape calculate_output_shape(const TensorShape &input_shape, unsigned int axis)
{
    TensorShape output_shape{ input_shape };
    output_shape.set(axis, 1);

    return output_shape;
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_UNUSED(op);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);

        const TensorShape output_shape         = calculate_output_shape(input->tensor_shape(), axis);
        const TensorInfo  tensor_info_reshaped = input->clone()->set_tensor_shape(output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_reshaped);
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, unsigned int axis)
{
    // Calculate output shape and set if empty
    const TensorShape output_shape = calculate_output_shape(input->tensor_shape(), axis);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output, output_shape, 1, input->data_type());

    unsigned int num_elems_processed_per_iteration = 16 / data_size_from_type(input->data_type());

    // Configure kernel window
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};

    return std::make_tuple(err, win);
}
} // namespace

NEReductionOperationKernel::NEReductionOperationKernel()
    : _input(nullptr), _output(nullptr), _reduction_axis(0), _op(ReductionOperation::SUM_SQUARE), _border_size()
{
}

BorderSize NEReductionOperationKernel::border_size() const
{
    return _border_size;
}

void NEReductionOperationKernel::configure(const ITensor *input, ITensor *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis, op));

    unsigned int num_elems_processed_per_iteration = 16 / data_size_from_type(input->info()->data_type());

    _input          = input;
    _output         = output;
    _border_size    = (axis == 0) ? BorderSize(0, num_elems_processed_per_iteration - (input->info()->dimension(0) % num_elems_processed_per_iteration), 0, 0) : BorderSize();
    _op             = op;
    _reduction_axis = axis;

    // Configure kernel window
    auto win_config = validate_and_configure_window(_input->info(), _output->info(), axis);

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));
}

Status NEReductionOperationKernel::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis, op));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get(), axis)));

    return Status{};
}

void NEReductionOperationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_op)
    {
        case ReductionOperation::SUM_SQUARE:
            reduce_sumsq(window, _input, _output, _reduction_axis);
            break;
        case ReductionOperation::MEAN_SUM:
            reduce_mean_sum(window, _input, _output, _reduction_axis);
            break;
        case ReductionOperation::SUM:
            reduce_sum(window, _input, _output, _reduction_axis);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction operation.");
    }
}
} // namespace arm_compute
