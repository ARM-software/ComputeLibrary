/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace
{
uint32x4x4_t calculate_index(uint32_t idx, float32x4_t a, float32x4_t b, uint32x4x4_t c, ReductionOperation op, int axis)
{
    uint32x4_t mask{ 0 };
    if(op == ReductionOperation::ARG_IDX_MIN)
    {
        mask = wrapper::vcgt(b, a);
    }
    else
    {
        mask = wrapper::vclt(b, a);
    }

    uint32x4_t vec_idx = { idx, idx + 1, idx + 2, idx + 3 };
    if(axis != 0)
    {
        vec_idx = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
    }
    uint32x4x4_t res = { { wrapper::vbsl(mask, vec_idx, c.val[0]), 0, 0, 0 } };

    return res;
}

uint32x4x4_t calculate_index(uint32_t idx, uint8x16_t a, uint8x16_t b, uint32x4x4_t c, ReductionOperation op, int axis)
{
    uint32x4x4_t mask{ { 0 } };
    uint8x16_t   mask_u8{ 0 };
    if(op == ReductionOperation::ARG_IDX_MIN)
    {
        mask_u8 = wrapper::vcgt(b, a);
    }
    else
    {
        mask_u8 = wrapper::vclt(b, a);
    }
    auto wide_u16_1 = wrapper::vorr(vshll_n_u8(wrapper::vgetlow(mask_u8), 8), wrapper::vmovl(wrapper::vgetlow(mask_u8)));
    auto wide_u16_2 = wrapper::vorr(vshll_n_u8(wrapper::vgethigh(mask_u8), 8), wrapper::vmovl(wrapper::vgethigh(mask_u8)));
    mask.val[0]     = wrapper::vorr(vshll_n_u16(wrapper::vgetlow(wide_u16_1), 16), wrapper::vmovl(wrapper::vgetlow(wide_u16_1)));
    mask.val[1]     = wrapper::vorr(vshll_n_u16(wrapper::vgethigh(wide_u16_1), 16), wrapper::vmovl(wrapper::vgethigh(wide_u16_1)));
    mask.val[2]     = wrapper::vorr(vshll_n_u16(wrapper::vgetlow(wide_u16_2), 16), wrapper::vmovl(wrapper::vgetlow(wide_u16_2)));
    mask.val[3]     = wrapper::vorr(vshll_n_u16(wrapper::vgethigh(wide_u16_2), 16), wrapper::vmovl(wrapper::vgethigh(wide_u16_2)));

    uint32x4x4_t vec_idx = { { { idx + 0, idx + 1, idx + 2, idx + 3 },
            { idx + 4, idx + 5, idx + 6, idx + 7 },
            { idx + 8, idx + 9, idx + 10, idx + 11 },
            { idx + 12, idx + 13, idx + 14, idx + 15 }
        }
    };
    if(axis != 0)
    {
        vec_idx.val[0] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
        vec_idx.val[1] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
        vec_idx.val[2] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
        vec_idx.val[3] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
    }
    uint32x4x4_t res =
    {
        {
            vbslq_u32(mask.val[0], vec_idx.val[0], c.val[0]),
            vbslq_u32(mask.val[1], vec_idx.val[1], c.val[1]),
            vbslq_u32(mask.val[2], vec_idx.val[2], c.val[2]),
            vbslq_u32(mask.val[3], vec_idx.val[3], c.val[3])
        }
    };

    return res;
}

uint32_t calculate_vector_index(uint32x4x4_t vec_res_idx, float32x4_t vec_res_value, ReductionOperation op)
{
    uint32x4_t res_idx_mask{ 0 };
    uint32x4_t mask_ones = vdupq_n_u32(0xFFFFFFFF);

    if(op == ReductionOperation::ARG_IDX_MIN)
    {
        auto pmin    = wrapper::vpmin(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
        pmin         = wrapper::vpmin(pmin, pmin);
        auto mask    = wrapper::vceq(vec_res_value, wrapper::vcombine(pmin, pmin));
        res_idx_mask = wrapper::vand(vec_res_idx.val[0], mask);
    }
    else
    {
        auto pmax    = wrapper::vpmax(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
        pmax         = wrapper::vpmax(pmax, pmax);
        auto mask    = vceqq_f32(vec_res_value, wrapper::vcombine(pmax, pmax));
        res_idx_mask = wrapper::vand(vec_res_idx.val[0], mask);
    }

    res_idx_mask = wrapper::vadd(res_idx_mask, mask_ones);
    auto pmin    = wrapper::vpmin(wrapper::vgethigh(res_idx_mask), wrapper::vgetlow(res_idx_mask));
    pmin         = wrapper::vpmin(pmin, pmin);
    uint32_t res = wrapper::vgetlane(pmin, 0);

    return (res - 0xFFFFFFFF);
}

uint32_t calculate_vector_index(uint32x4x4_t vec_res_idx, uint8x16_t vec_res_value, ReductionOperation op)
{
    uint32x4x4_t res_idx_mask{ { 0 } };
    uint32x4_t   mask_ones = vdupq_n_u32(0xFFFFFFFF);
    uint8x16_t   mask_u8{ 0 };
    if(op == ReductionOperation::ARG_IDX_MIN)
    {
        auto pmin = wrapper::vpmin(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
        pmin      = wrapper::vpmin(pmin, pmin);
        pmin      = wrapper::vpmin(pmin, pmin);
        pmin      = wrapper::vpmin(pmin, pmin);
        mask_u8   = wrapper::vceq(vec_res_value, wrapper::vcombine(pmin, pmin));
    }
    else
    {
        auto pmax = wrapper::vpmax(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
        pmax      = wrapper::vpmax(pmax, pmax);
        pmax      = wrapper::vpmax(pmax, pmax);
        pmax      = wrapper::vpmax(pmax, pmax);
        mask_u8   = wrapper::vceq(vec_res_value, wrapper::vcombine(pmax, pmax));
    }

    // Widen vectors
    auto wide_u16_1     = wrapper::vorr(vshll_n_u8(wrapper::vgetlow(mask_u8), 8), wrapper::vmovl(wrapper::vgetlow(mask_u8)));
    auto wide_u16_2     = wrapper::vorr(vshll_n_u8(wrapper::vgethigh(mask_u8), 8), wrapper::vmovl(wrapper::vgethigh(mask_u8)));
    auto wide_u32_1     = wrapper::vorr(vshll_n_u16(wrapper::vgetlow(wide_u16_1), 16), wrapper::vmovl(wrapper::vgetlow(wide_u16_1)));
    auto wide_u32_2     = wrapper::vorr(vshll_n_u16(wrapper::vgethigh(wide_u16_1), 16), wrapper::vmovl(wrapper::vgethigh(wide_u16_1)));
    auto wide_u32_3     = wrapper::vorr(vshll_n_u16(wrapper::vgetlow(wide_u16_2), 16), wrapper::vmovl(wrapper::vgetlow(wide_u16_2)));
    auto wide_u32_4     = wrapper::vorr(vshll_n_u16(wrapper::vgethigh(wide_u16_2), 16), wrapper::vmovl(wrapper::vgethigh(wide_u16_2)));
    res_idx_mask.val[0] = wrapper::vand(vec_res_idx.val[0], wide_u32_1);
    res_idx_mask.val[1] = wrapper::vand(vec_res_idx.val[1], wide_u32_2);
    res_idx_mask.val[2] = wrapper::vand(vec_res_idx.val[2], wide_u32_3);
    res_idx_mask.val[3] = wrapper::vand(vec_res_idx.val[3], wide_u32_4);
    res_idx_mask.val[0] = wrapper::vadd(res_idx_mask.val[0], mask_ones);
    res_idx_mask.val[1] = wrapper::vadd(res_idx_mask.val[1], mask_ones);
    res_idx_mask.val[2] = wrapper::vadd(res_idx_mask.val[2], mask_ones);
    res_idx_mask.val[3] = wrapper::vadd(res_idx_mask.val[3], mask_ones);

    uint32_t res  = 0xFFFFFFFF;
    int      iter = 0;
    do
    {
        auto pmin = wrapper::vpmin(wrapper::vgethigh(res_idx_mask.val[iter]), wrapper::vgetlow(res_idx_mask.val[iter]));
        pmin      = wrapper::vpmin(pmin, pmin);
        res       = std::min(wrapper::vgetlane(pmin, 0), res);
        iter++;
    }
    while(iter < 4);

    return (res - 0xFFFFFFFF);
}
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
uint32x4x4_t calculate_index(uint32_t idx, float16x8_t a, float16x8_t b, uint32x4x4_t c, ReductionOperation op, int axis)
{
    uint32x4x2_t mask{ 0 };
    uint16x8_t   mask_u16{ 0 };
    if(op == ReductionOperation::ARG_IDX_MIN)
    {
        mask_u16 = wrapper::vcgt(b, a);
    }
    else
    {
        mask_u16 = wrapper::vclt(b, a);
    }
    mask.val[0]          = wrapper::vmovl(wrapper::vgetlow(mask_u16));
    mask.val[1]          = wrapper::vmovl(wrapper::vgethigh(mask_u16));
    uint32x4x2_t vec_idx = { { { idx + 0, idx + 1, idx + 2, idx + 3 },
            { idx + 4, idx + 5, idx + 6, idx + 7 }
        }
    };
    if(axis != 0)
    {
        vec_idx.val[0] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
        vec_idx.val[1] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
    }
    uint32x4x4_t res = { wrapper::vbsl(mask.val[0], vec_idx.val[0], c.val[0]),
                         wrapper::vbsl(mask.val[1], vec_idx.val[1], c.val[1]),
                         0, 0
                       };

    return res;
}

uint32_t calculate_vector_index(uint32x4x4_t vec_res_idx, float16x8_t vec_res_value, ReductionOperation op)
{
    uint32x4x2_t res_idx_mask{ 0 };
    uint32x4_t   mask_ones = vdupq_n_u32(0xFFFFFFFF);
    uint16x8_t   mask_u16;
    if(op == ReductionOperation::ARG_IDX_MIN)
    {
        auto pmin = wrapper::vpmin(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
        pmin      = wrapper::vpmin(pmin, pmin);
        pmin      = wrapper::vpmin(pmin, pmin);
        mask_u16  = wrapper::vceq(vec_res_value, wrapper::vcombine(pmin, pmin));
    }
    else
    {
        auto pmax = wrapper::vpmax(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
        pmax      = wrapper::vpmax(pmax, pmax);
        pmax      = wrapper::vpmax(pmax, pmax);
        mask_u16  = wrapper::vceq(vec_res_value, wrapper::vcombine(pmax, pmax));
    }

    // Widen vectors
    auto wide_u32_1     = wrapper::vorr(vshll_n_u16(wrapper::vgetlow(mask_u16), 8), wrapper::vmovl(wrapper::vgetlow(mask_u16)));
    auto wide_u32_2     = wrapper::vorr(vshll_n_u16(wrapper::vgethigh(mask_u16), 8), wrapper::vmovl(wrapper::vgethigh(mask_u16)));
    res_idx_mask.val[0] = wrapper::vand(vec_res_idx.val[0], wide_u32_1);
    res_idx_mask.val[1] = wrapper::vand(vec_res_idx.val[1], wide_u32_2);
    res_idx_mask.val[0] = wrapper::vadd(res_idx_mask.val[0], mask_ones);
    res_idx_mask.val[1] = wrapper::vadd(res_idx_mask.val[1], mask_ones);

    uint32_t res  = 0xFFFFFFFF;
    int      iter = 0;
    do
    {
        auto pmin = wrapper::vpmin(wrapper::vgethigh(res_idx_mask.val[iter]), wrapper::vgetlow(res_idx_mask.val[iter]));
        pmin      = wrapper::vpmin(pmin, pmin);
        res       = std::min(wrapper::vgetlane(pmin, 0), res);
        iter++;
    }
    while(iter < 2);

    return (res - 0xFFFFFFFF);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <class F>
class Reducer
{
public:
    static void reduceX(const Window &window, const ITensor *input, ITensor *output, F f, const ReductionOperation op)
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

            f(in, out, in_slice, out_slice, *input->info(), op);
        }
        while(window.slide_window_slice_1D(in_slice) && out_window.slide_window_slice_1D(out_slice));
    }
    static void reduceY(const Window &window, const ITensor *input, ITensor *output, F f, const ReductionOperation op)
    {
        // Set in window
        Window in_window(window);
        Window out_window(window);

        in_window.set(Window::DimY, Window::Dimension(0, 1, 1));
        out_window.set(Window::DimY, Window::Dimension(0, output->info()->dimension(1), output->info()->dimension(1)));

        // Get first input and output slices
        Window in_slice  = in_window.first_slice_window_2D();
        Window out_slice = out_window.first_slice_window_2D();

        do
        {
            Iterator in(input, in_slice);
            Iterator out(output, out_slice);

            f(in, out, in_slice, out_slice, *input->info(), 1, op);
        }
        while(in_window.slide_window_slice_2D(in_slice) && out_window.slide_window_slice_2D(out_slice));
    }
    static void reduceZ(const Window &window, const ITensor *input, ITensor *output, F f, const ReductionOperation op)
    {
        // Set in window
        Window in_window(window);
        Window out_window(window);

        in_window.set(Window::DimZ, Window::Dimension(0, 1, 1));
        out_window.set(Window::DimZ, Window::Dimension(0, output->info()->dimension(2), output->info()->dimension(2)));

        // Get first input and output slices
        Window in_slice  = in_window.first_slice_window_3D();
        Window out_slice = out_window.first_slice_window_3D();

        do
        {
            Iterator in(input, in_slice);
            Iterator out(output, out_slice);

            f(in, out, in_slice, out_slice, *input->info(), 2, op);
        }
        while(in_window.slide_window_slice_3D(in_slice) && out_window.slide_window_slice_3D(out_slice));
    }
    static void reduceW(const Window &window, const ITensor *input, ITensor *output, F f, const ReductionOperation op)
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

            f(in, out, in_slice, out_slice, *input->info(), 3, op);
        }
        while(in_window.slide_window_slice_4D(in_slice) && out_window.slide_window_slice_4D(out_slice));
    }
};

template <typename T, int S>
struct RedOpX
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice, const TensorInfo &in_info, const ReductionOperation op)
    {
        ARM_COMPUTE_UNUSED(out_slice);
        auto init_res_value = static_cast<T>(0.f);
        if(op == ReductionOperation::ARG_IDX_MAX || op == ReductionOperation::ARG_IDX_MIN)
        {
            init_res_value = *reinterpret_cast<T *>(input.ptr());
        }
        else if(op == ReductionOperation::PROD)
        {
            init_res_value = static_cast<T>(1.f);
        }
        auto         vec_res_value = wrapper::vdup_n(init_res_value, ExactTagType{});
        uint32x4x4_t vec_res_idx{ { 0 } };

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto in_ptr       = reinterpret_cast<const T *>(input.ptr());
            const auto vec_elements = wrapper::vloadq(in_ptr);

            switch(op)
            {
                case ReductionOperation::SUM_SQUARE:
                    vec_res_value = wrapper::vadd(wrapper::vmul(vec_elements, vec_elements), vec_res_value);
                    break;
                case ReductionOperation::MEAN_SUM:
                case ReductionOperation::SUM:
                    vec_res_value = wrapper::vadd(vec_elements, vec_res_value);
                    break;
                case ReductionOperation::PROD:
                    vec_res_value = wrapper::vmul(vec_elements, vec_res_value);
                    break;
                case ReductionOperation::ARG_IDX_MIN:
                {
                    auto temp_vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                    vec_res_idx             = calculate_index(id.x(), temp_vec_res_value, vec_res_value, vec_res_idx, op, 0);
                    vec_res_value           = temp_vec_res_value;
                    break;
                }
                case ReductionOperation::ARG_IDX_MAX:
                {
                    auto temp_vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                    vec_res_idx             = calculate_index(id.x(), temp_vec_res_value, vec_res_value, vec_res_idx, op, 0);
                    vec_res_value           = temp_vec_res_value;
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        },
        input);

        switch(op)
        {
            case ReductionOperation::SUM:
            case ReductionOperation::SUM_SQUARE:
            case ReductionOperation::MEAN_SUM:
            {
                auto carry_res = wrapper::vpadd(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
                for(int i = 0; i < S / 4; ++i)
                {
                    carry_res = wrapper::vpadd(carry_res, carry_res);
                }
                auto res = wrapper::vgetlane(carry_res, 0);

                if(op == ReductionOperation::MEAN_SUM)
                {
                    res /= in_info.dimension(0);
                }

                *(reinterpret_cast<T *>(output.ptr())) = res;
                break;
            }
            case ReductionOperation::PROD:
            {
                auto carry_res = wrapper::vmul(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
                T    res       = 1;
                for(int i = 0; i < S / 2; ++i)
                {
                    res *= wrapper::vgetlane(carry_res, i);
                }
                *(reinterpret_cast<T *>(output.ptr())) = res;
                break;
            }
            case ReductionOperation::ARG_IDX_MIN:
            case ReductionOperation::ARG_IDX_MAX:
            {
                auto res                                      = calculate_vector_index(vec_res_idx, vec_res_value, op);
                *(reinterpret_cast<uint32_t *>(output.ptr())) = res;
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Not supported");
        }
    }
};

struct RedOpX_qasymm8
{
    inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice, const TensorInfo &in_info, const ReductionOperation op)
    {
        ARM_COMPUTE_UNUSED(out_slice);
        auto vec_res_value1 = vdupq_n_u32(static_cast<uint32_t>(0.f));
        auto vec_res_value2 = vdupq_n_u32(static_cast<uint32_t>(0.f));
        auto vec_res_value3 = vdupq_n_u32(static_cast<uint32_t>(0.f));
        auto vec_res_value4 = vdupq_n_u32(static_cast<uint32_t>(0.f));

        auto vec_res_value1_f = vdupq_n_f32(static_cast<float>(1.f));
        auto vec_res_value2_f = vdupq_n_f32(static_cast<float>(1.f));
        auto vec_res_value3_f = vdupq_n_f32(static_cast<float>(1.f));
        auto vec_res_value4_f = vdupq_n_f32(static_cast<float>(1.f));

        uint8x16_t vec_res_value = { 0 };

        if(op == ReductionOperation::ARG_IDX_MAX || op == ReductionOperation::ARG_IDX_MIN)
        {
            vec_res_value = wrapper::vdup_n(*input.ptr(), wrapper::traits::vector_128_tag{});
        }

        uint32x4x4_t vec_res_idx{ { 0 } };
        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto vec_elements = wrapper::vloadq(input.ptr());
            switch(op)
            {
                case ReductionOperation::SUM:
                case ReductionOperation::MEAN_SUM:
                {
                    const auto temp16x8t_1 = wrapper::vmovl(wrapper::vgetlow(vec_elements));
                    const auto temp16x8t_2 = wrapper::vmovl(wrapper::vgethigh(vec_elements));

                    const auto temp32x4t_1 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_1));
                    const auto temp32x4t_2 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_1));
                    const auto temp32x4t_3 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_2));
                    const auto temp32x4t_4 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_2));

                    vec_res_value1 = wrapper::vadd(temp32x4t_1, vec_res_value1);
                    vec_res_value2 = wrapper::vadd(temp32x4t_2, vec_res_value2);
                    vec_res_value3 = wrapper::vadd(temp32x4t_3, vec_res_value3);
                    vec_res_value4 = wrapper::vadd(temp32x4t_4, vec_res_value4);
                    break;
                }
                case ReductionOperation::PROD:
                {
                    const auto offset32x4f_4 = vdupq_n_f32(in_info.quantization_info().offset);
                    const auto scale32x4f_4  = vdupq_n_f32(in_info.quantization_info().scale);

                    const auto temp16x8t_1 = vmovl_u8(vget_low_u8(vec_elements));
                    const auto temp16x8t_2 = vmovl_u8(vget_high_u8(vec_elements));

                    const auto temp32x4t_1 = vmovl_u16(vget_low_u16(temp16x8t_1));
                    const auto temp32x4t_2 = vmovl_u16(vget_high_u16(temp16x8t_1));
                    const auto temp32x4t_3 = vmovl_u16(vget_low_u16(temp16x8t_2));
                    const auto temp32x4t_4 = vmovl_u16(vget_high_u16(temp16x8t_2));

                    auto temp32x4f_1 = vcvtq_f32_u32(temp32x4t_1);
                    auto temp32x4f_2 = vcvtq_f32_u32(temp32x4t_2);
                    auto temp32x4f_3 = vcvtq_f32_u32(temp32x4t_3);
                    auto temp32x4f_4 = vcvtq_f32_u32(temp32x4t_4);

                    //de-quantize vec_elements
                    temp32x4f_1 = vmulq_f32(vsubq_f32(temp32x4f_1, offset32x4f_4), scale32x4f_4);
                    temp32x4f_2 = vmulq_f32(vsubq_f32(temp32x4f_2, offset32x4f_4), scale32x4f_4);
                    temp32x4f_3 = vmulq_f32(vsubq_f32(temp32x4f_3, offset32x4f_4), scale32x4f_4);
                    temp32x4f_4 = vmulq_f32(vsubq_f32(temp32x4f_4, offset32x4f_4), scale32x4f_4);

                    vec_res_value1_f = vmulq_f32(temp32x4f_1, vec_res_value1_f);
                    vec_res_value2_f = vmulq_f32(temp32x4f_2, vec_res_value2_f);
                    vec_res_value3_f = vmulq_f32(temp32x4f_3, vec_res_value3_f);
                    vec_res_value4_f = vmulq_f32(temp32x4f_4, vec_res_value4_f);
                    break;
                }
                case ReductionOperation::ARG_IDX_MIN:
                {
                    auto temp_vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                    vec_res_idx             = calculate_index(id.x(), temp_vec_res_value, vec_res_value, vec_res_idx, op, 0);
                    vec_res_value           = temp_vec_res_value;
                    break;
                }
                case ReductionOperation::ARG_IDX_MAX:
                {
                    auto temp_vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                    vec_res_idx             = calculate_index(id.x(), temp_vec_res_value, vec_res_value, vec_res_idx, op, 0);
                    vec_res_value           = temp_vec_res_value;
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        },
        input);

        if(op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::ARG_IDX_MAX)
        {
            auto res                                      = calculate_vector_index(vec_res_idx, vec_res_value, op);
            *(reinterpret_cast<uint32_t *>(output.ptr())) = res;
        }
        else if(op == ReductionOperation::PROD)
        {
            auto carry_res = wrapper::vmul(vec_res_value1_f, vec_res_value2_f);
            carry_res      = wrapper::vmul(carry_res, vec_res_value3_f);
            carry_res      = wrapper::vmul(carry_res, vec_res_value4_f);

            float res = wrapper::vgetlane(carry_res, 0);
            res *= wrapper::vgetlane(carry_res, 1);
            res *= wrapper::vgetlane(carry_res, 2);
            res *= wrapper::vgetlane(carry_res, 3);

            //re-quantize result
            res             = sqcvt_qasymm8_f32(res, in_info.quantization_info().scale, in_info.quantization_info().offset);
            *(output.ptr()) = static_cast<uint8_t>(res);
        }
        else
        {
            auto carry_res = wrapper::vadd(vec_res_value1, vec_res_value2);
            carry_res      = wrapper::vadd(carry_res, vec_res_value3);
            carry_res      = wrapper::vadd(carry_res, vec_res_value4);

            auto carry_paddition = wrapper::vpadd(wrapper::vgethigh(carry_res), wrapper::vgetlow(carry_res));
            carry_paddition      = wrapper::vpadd(carry_paddition, carry_paddition);
            auto res             = wrapper::vgetlane(carry_paddition, 0);

            if(op == ReductionOperation::MEAN_SUM)
            {
                res /= in_info.dimension(0);
            }

            *(output.ptr()) = static_cast<uint8_t>(res);
        }
    }
};

template <typename T, int S>
struct RedOpYZW
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;
    using neon_vector  = typename wrapper::traits::neon_vector<T, S>::type;

    inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice, const TensorInfo &in_info, int axis, const ReductionOperation op)
    {
        ARM_COMPUTE_UNUSED(out_slice);

        execute_window_loop(in_slice, [&](const Coordinates &)
        {
            neon_vector vec_res_value = { 0 };
            if(op == ReductionOperation::ARG_IDX_MAX || op == ReductionOperation::ARG_IDX_MIN)
            {
                vec_res_value = wrapper::vloadq(reinterpret_cast<T *>(input.ptr()));
            }
            else if(op == ReductionOperation::PROD)
            {
                vec_res_value = wrapper::vdup_n(static_cast<T>(1.f), ExactTagType{});
            }
            else
            {
                vec_res_value = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
            }
            uint32x4x4_t vec_res_idx{ { 0 } };

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

                switch(op)
                {
                    case ReductionOperation::SUM:
                    case ReductionOperation::MEAN_SUM:
                        vec_res_value = wrapper::vadd(vec_elements, vec_res_value);
                        break;
                    case ReductionOperation::SUM_SQUARE:
                        vec_res_value = wrapper::vadd(wrapper::vmul(vec_elements, vec_elements), vec_res_value);
                        break;
                    case ReductionOperation::PROD:
                        vec_res_value = wrapper::vmul(vec_elements, vec_res_value);
                        break;
                    case ReductionOperation::ARG_IDX_MIN:
                    {
                        auto temp_vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                        vec_res_idx             = calculate_index(dim, temp_vec_res_value, vec_res_value, vec_res_idx, op, axis);
                        vec_res_value           = temp_vec_res_value;
                        break;
                    }
                    case ReductionOperation::ARG_IDX_MAX:
                    {
                        auto temp_vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                        vec_res_idx             = calculate_index(dim, temp_vec_res_value, vec_res_value, vec_res_idx, op, axis);
                        vec_res_value           = temp_vec_res_value;
                        break;
                    }
                    default:
                        ARM_COMPUTE_ERROR("Not supported");
                }
            }

            if(op == ReductionOperation::MEAN_SUM)
            {
                auto vec_width_inv = wrapper::vinv(wrapper::vdup_n(static_cast<T>(in_info.dimension(axis)), ExactTagType{}));
                vec_res_value      = wrapper::vmul(vec_res_value, vec_width_inv);
            }

            if(op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::ARG_IDX_MAX)
            {
                wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr()), vec_res_idx.val[0]);
            }
            else
            {
                wrapper::vstore(reinterpret_cast<T *>(output.ptr()), vec_res_value);
            }
        },
        input, output);
    }
};

template <typename T, int S, int axis, ReductionOperation op>
struct RedOpYZW_complex
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;
    using neon_vector  = typename wrapper::traits::neon_vector<T, S>::type;

    inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice, const TensorInfo &in_info, int, const ReductionOperation)
    {
        ARM_COMPUTE_UNUSED(out_slice);
        ARM_COMPUTE_ERROR_ON(axis != 2);

        const size_t stride_z = in_info.strides_in_bytes()[axis];

        execute_window_loop(in_slice, [&](const Coordinates &)
        {
            neon_vector vec_res_value_0 = { 0 };
            neon_vector vec_res_value_1 = { 0 };

            vec_res_value_0 = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
            vec_res_value_1 = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});

            for(unsigned int dim = 0; dim < in_info.dimension(axis); ++dim)
            {
                T *in_ptr_0;
                T *in_ptr_1;
                switch(axis)
                {
                    case 2:
                        in_ptr_0 = reinterpret_cast<T *>(input.ptr() + stride_z * dim);
                        in_ptr_1 = reinterpret_cast<T *>(input.ptr() + 16 + stride_z * dim);
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Not supported");
                }
                const auto vec_elements_0 = wrapper::vloadq(in_ptr_0);
                const auto vec_elements_1 = wrapper::vloadq(in_ptr_1);

                switch(op)
                {
                    case ReductionOperation::SUM:
                        vec_res_value_0 = wrapper::vadd(vec_elements_0, vec_res_value_0);
                        vec_res_value_1 = wrapper::vadd(vec_elements_1, vec_res_value_1);
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Not supported");
                }
            }

            wrapper::vstore(reinterpret_cast<T *>(output.ptr()), vec_res_value_0);
            wrapper::vstore(reinterpret_cast<T *>(output.ptr() + 16), vec_res_value_1);

        },
        input, output);
    }
};

struct RedOpYZW_qasymm8
{
    inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice, const TensorInfo &in_info, int axis, const ReductionOperation op)
    {
        ARM_COMPUTE_UNUSED(out_slice);

        execute_window_loop(in_slice, [&](const Coordinates &)
        {
            uint32x4x4_t vec_res_idx{ { 0 } };
            auto         vec_res_value1 = vdupq_n_u32(0);
            auto         vec_res_value2 = vdupq_n_u32(0);
            auto         vec_res_value3 = vdupq_n_u32(0);
            auto         vec_res_value4 = vdupq_n_u32(0);

            auto vec_res_value1_f = vdupq_n_f32(1);
            auto vec_res_value2_f = vdupq_n_f32(1);
            auto vec_res_value3_f = vdupq_n_f32(1);
            auto vec_res_value4_f = vdupq_n_f32(1);

            auto vec_res_value = wrapper::vloadq(input.ptr());

            for(unsigned int index_dim = 0; index_dim < in_info.dimension(axis); ++index_dim)
            {
                uint8_t *in_ptr;
                switch(axis)
                {
                    case 1:
                        in_ptr = input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, index_dim));
                        break;
                    case 2:
                        in_ptr = input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, 0, index_dim));
                        break;
                    case 3:
                        in_ptr = input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, 0, 0, index_dim));
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Not supported");
                }
                const auto vec_elements = wrapper::vloadq(in_ptr);

                switch(op)
                {
                    case ReductionOperation::SUM:
                    case ReductionOperation::MEAN_SUM:
                    {
                        const auto temp16x8t_1 = wrapper::vmovl(wrapper::vgetlow(vec_elements));
                        const auto temp16x8t_2 = wrapper::vmovl(wrapper::vgethigh(vec_elements));

                        const auto temp32x4t_1 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_1));
                        const auto temp32x4t_2 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_1));
                        const auto temp32x4t_3 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_2));
                        const auto temp32x4t_4 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_2));

                        vec_res_value1 = wrapper::vadd(temp32x4t_1, vec_res_value1);
                        vec_res_value2 = wrapper::vadd(temp32x4t_2, vec_res_value2);
                        vec_res_value3 = wrapper::vadd(temp32x4t_3, vec_res_value3);
                        vec_res_value4 = wrapper::vadd(temp32x4t_4, vec_res_value4);
                        break;
                    }
                    case ReductionOperation::PROD:
                    {
                        const auto offset32x4f_4 = vdupq_n_f32(in_info.quantization_info().offset);
                        const auto scale32x4f_4  = vdupq_n_f32(in_info.quantization_info().scale);

                        const auto temp16x8t_1 = vmovl_u8(vget_low_u8(vec_elements));
                        const auto temp16x8t_2 = vmovl_u8(vget_high_u8(vec_elements));

                        const auto temp32x4t_1 = vmovl_u16(vget_low_u16(temp16x8t_1));
                        const auto temp32x4t_2 = vmovl_u16(vget_high_u16(temp16x8t_1));
                        const auto temp32x4t_3 = vmovl_u16(vget_low_u16(temp16x8t_2));
                        const auto temp32x4t_4 = vmovl_u16(vget_high_u16(temp16x8t_2));

                        auto temp32x4f_1 = vcvtq_f32_u32(temp32x4t_1);
                        auto temp32x4f_2 = vcvtq_f32_u32(temp32x4t_2);
                        auto temp32x4f_3 = vcvtq_f32_u32(temp32x4t_3);
                        auto temp32x4f_4 = vcvtq_f32_u32(temp32x4t_4);

                        //de-quantize vec_elements
                        temp32x4f_1 = vmulq_f32(vsubq_f32(temp32x4f_1, offset32x4f_4), scale32x4f_4);
                        temp32x4f_2 = vmulq_f32(vsubq_f32(temp32x4f_2, offset32x4f_4), scale32x4f_4);
                        temp32x4f_3 = vmulq_f32(vsubq_f32(temp32x4f_3, offset32x4f_4), scale32x4f_4);
                        temp32x4f_4 = vmulq_f32(vsubq_f32(temp32x4f_4, offset32x4f_4), scale32x4f_4);

                        vec_res_value1_f = vmulq_f32(temp32x4f_1, vec_res_value1_f);
                        vec_res_value2_f = vmulq_f32(temp32x4f_2, vec_res_value2_f);
                        vec_res_value3_f = vmulq_f32(temp32x4f_3, vec_res_value3_f);
                        vec_res_value4_f = vmulq_f32(temp32x4f_4, vec_res_value4_f);
                        break;
                    }
                    case ReductionOperation::ARG_IDX_MIN:
                    {
                        auto temp_vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                        vec_res_idx             = calculate_index(index_dim, temp_vec_res_value, vec_res_value, vec_res_idx, op, axis);
                        vec_res_value           = temp_vec_res_value;
                        break;
                    }
                    case ReductionOperation::ARG_IDX_MAX:
                    {
                        auto temp_vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                        vec_res_idx             = calculate_index(index_dim, temp_vec_res_value, vec_res_value, vec_res_idx, op, axis);
                        vec_res_value           = temp_vec_res_value;
                        break;
                    }
                    default:
                        ARM_COMPUTE_ERROR("Not supported");
                }
            }

            if(op == ReductionOperation::MEAN_SUM)
            {
                const auto vec_width_inv = wrapper::vinv(vdupq_n_f32(in_info.dimension(axis)));
                vec_res_value1_f         = wrapper::vmul(vcvtq_f32_u32(vec_res_value1), vec_width_inv);
                vec_res_value2_f         = wrapper::vmul(vcvtq_f32_u32(vec_res_value2), vec_width_inv);
                vec_res_value3_f         = wrapper::vmul(vcvtq_f32_u32(vec_res_value3), vec_width_inv);
                vec_res_value4_f         = wrapper::vmul(vcvtq_f32_u32(vec_res_value4), vec_width_inv);

                vec_res_value1 = vcvtq_u32_f32(vec_res_value1_f);
                vec_res_value2 = vcvtq_u32_f32(vec_res_value2_f);
                vec_res_value3 = vcvtq_u32_f32(vec_res_value3_f);
                vec_res_value4 = vcvtq_u32_f32(vec_res_value4_f);
            }
            else if(op == ReductionOperation::PROD)
            {
                const auto offset32x4f_4 = vdupq_n_f32(in_info.quantization_info().offset);
                const auto iscale32x4f_4 = vinvq_f32(vdupq_n_f32(in_info.quantization_info().scale));

                //re-quantize
                vec_res_value1_f = vaddq_f32(vmulq_f32(vec_res_value1_f, iscale32x4f_4), offset32x4f_4);
                vec_res_value2_f = vaddq_f32(vmulq_f32(vec_res_value2_f, iscale32x4f_4), offset32x4f_4);
                vec_res_value3_f = vaddq_f32(vmulq_f32(vec_res_value3_f, iscale32x4f_4), offset32x4f_4);
                vec_res_value4_f = vaddq_f32(vmulq_f32(vec_res_value4_f, iscale32x4f_4), offset32x4f_4);

                vec_res_value1 = vcvtq_u32_f32(vec_res_value1_f);
                vec_res_value2 = vcvtq_u32_f32(vec_res_value2_f);
                vec_res_value3 = vcvtq_u32_f32(vec_res_value3_f);
                vec_res_value4 = vcvtq_u32_f32(vec_res_value4_f);
            }

            if(op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::ARG_IDX_MAX)
            {
                wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr()), vec_res_idx.val[0]);
                wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr()) + 4, vec_res_idx.val[1]);
                wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr()) + 8, vec_res_idx.val[2]);
                wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr()) + 12, vec_res_idx.val[3]);
            }
            else
            {
                const auto temp16x8t_1 = vcombine_u16(wrapper::vqmovn(vec_res_value1), wrapper::vqmovn(vec_res_value2));
                const auto temp16x8t_2 = vcombine_u16(wrapper::vqmovn(vec_res_value3), wrapper::vqmovn(vec_res_value4));
                auto       res         = vcombine_u8(wrapper::vqmovn(temp16x8t_1), wrapper::vqmovn(temp16x8t_2));
                wrapper::vstore(output.ptr(), res);
            }

        },
        input, output);
    }
};

void reduce_op(const Window &window, const ITensor *input, ITensor *output, unsigned int axis, const ReductionOperation op)
{
    const bool is_complex = (input->info()->num_channels() == 2);

    if(is_complex)
    {
        switch(axis)
        {
            case 2:
                switch(input->info()->data_type())
                {
                    case DataType::F32:
                        switch(op)
                        {
                            case ReductionOperation::SUM:
                                return Reducer<RedOpYZW_complex<float, 4, 2, ReductionOperation::SUM>>::reduceZ(window, input, output, RedOpYZW_complex<float, 4, 2, ReductionOperation::SUM>(), op);
                            default:
                                ARM_COMPUTE_ERROR("Not supported");
                        }
                    default:
                        ARM_COMPUTE_ERROR("Not supported");
                }
            default:
                ARM_COMPUTE_ERROR("Not supported");
        }
    }

    switch(axis)
    {
        case 0:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpX_qasymm8>::reduceX(window, input, output, RedOpX_qasymm8(), op);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpX<float16_t, 8>>::reduceX(window, input, output, RedOpX<float16_t, 8>(), op);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpX<float, 4>>::reduceX(window, input, output, RedOpX<float, 4>(), op);
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 1:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpYZW_qasymm8>::reduceY(window, input, output, RedOpYZW_qasymm8(), op);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8>>::reduceY(window, input, output, RedOpYZW<float16_t, 8>(), op);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4>>::reduceY(window, input, output, RedOpYZW<float, 4>(), op);
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 2:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpYZW_qasymm8>::reduceZ(window, input, output, RedOpYZW_qasymm8(), op);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8>>::reduceZ(window, input, output, RedOpYZW<float16_t, 8>(), op);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4>>::reduceZ(window, input, output, RedOpYZW<float, 4>(), op);
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        case 3:
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    return Reducer<RedOpYZW_qasymm8>::reduceW(window, input, output, RedOpYZW_qasymm8(), op);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    return Reducer<RedOpYZW<float16_t, 8>>::reduceW(window, input, output, RedOpYZW<float16_t, 8>(), op);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F32:
                    return Reducer<RedOpYZW<float, 4>>::reduceW(window, input, output, RedOpYZW<float, 4>(), op);
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction axis");
    }
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_UNUSED(op);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);

    if(input->num_channels() == 1)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 2, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON(op != ReductionOperation::SUM);
        ARM_COMPUTE_RETURN_ERROR_ON(axis != 2);
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");

    if(output->total_size() != 0)
    {
        bool is_arg_min_max = (op == ReductionOperation::ARG_IDX_MAX || op == ReductionOperation::ARG_IDX_MIN);
        if(!is_arg_min_max)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
            ARM_COMPUTE_RETURN_ERROR_ON(input->num_channels() != output->num_channels());
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U32);
        }

        const TensorShape output_shape         = arm_compute::misc::shape_calculator::compute_reduced_shape(input->tensor_shape(), axis);
        const TensorInfo  tensor_info_reshaped = input->clone()->set_tensor_shape(output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_reshaped);
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    // Calculate output shape and set if empty
    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_reduced_shape(input->tensor_shape(), axis);

    // Output auto initialization if not yet initialized
    const bool is_arg_min_max   = (op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::ARG_IDX_MAX);
    DataType   output_data_type = is_arg_min_max ? DataType::U32 : input->data_type();
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape).set_data_type(output_data_type).reset_padding().set_is_resizable(true));

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
    auto win_config = validate_and_configure_window(_input->info(), _output->info(), axis, op);

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));
}

Status NEReductionOperationKernel::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis, op));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get(), axis, op)));

    return Status{};
}

void NEReductionOperationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    reduce_op(window, _input, _output, _reduction_axis, _op);
}
} // namespace arm_compute
