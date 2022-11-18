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
#ifndef SRC_CORE_NEON_KERNELS_SCALE_LIST_H
#define SRC_CORE_NEON_KERNELS_SCALE_LIST_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/utils/ScaleUtils.h"
#include "support/Rounding.h"

namespace arm_compute
{
namespace cpu
{
#define DECLARE_SCALE_KERNEL(func_name)                                                                                         \
    void func_name(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,              \
                   InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, float sampling_offset, \
                   bool align_corners, const Window &window)

DECLARE_SCALE_KERNEL(s16_neon_scale);
DECLARE_SCALE_KERNEL(u8_neon_scale);
DECLARE_SCALE_KERNEL(s8_neon_scale);
DECLARE_SCALE_KERNEL(qasymm8_neon_scale);
DECLARE_SCALE_KERNEL(qasymm8_signed_neon_scale);

#undef DECLARE_SCALE_KERNEL

template <typename T>
void nearest_neon_scale(const ITensor *src, ITensor *dst, const ITensor *offsets, float sampling_offset,
                        bool align_corners, const Window &window)
{
    ARM_COMPUTE_UNUSED(offsets);

    // Compute the ratio between source and destination dimensions
    const float scale_x = scale_utils::calculate_resize_ratio(src->info()->dimension(1), dst->info()->dimension(1), align_corners);
    const float scale_y = scale_utils::calculate_resize_ratio(src->info()->dimension(2), dst->info()->dimension(2), align_corners);

    const int in_stride_y  = src->info()->strides_in_bytes()[1];
    const int in_stride_z  = src->info()->strides_in_bytes()[2];
    const int in_stride_w  = src->info()->strides_in_bytes()[3];
    const int out_stride_y = dst->info()->strides_in_bytes()[1];
    const int out_stride_z = dst->info()->strides_in_bytes()[2];
    const int out_stride_w = dst->info()->strides_in_bytes()[3];
    const int out_dim_ch   = dst->info()->dimension(0);
    const int step_cout    = 16 / sizeof(T);

    Window window_execution = window;
    window_execution.set(Window::DimX, Window::Dimension(0, 1, 1));
    Window win_in_out(window);
    win_in_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_in_out.set(Window::DimZ, Window::Dimension(0, 0, 0));
    Iterator in(src, win_in_out);
    Iterator out(dst, win_in_out);

    const int xo_start = window_execution.y().start();
    const int xo_end   = window_execution.y().end();
    const int xo_step  = window_execution.y().step();
    const int yo_start = window_execution.z().start();
    const int yo_end   = window_execution.z().end();
    const int yo_step  = window_execution.z().step();
    const int bo_start = window_execution[3].start();
    const int bo_end   = window_execution[3].end();
    const int bo_step  = window_execution[3].step();

    for(int bo = bo_start; bo < bo_end; bo += bo_step)
    {
        const uint8_t *in_ptr_base  = in.ptr() + bo * in_stride_w;
        uint8_t       *out_ptr_base = out.ptr() + bo * out_stride_w;

        for(int yo = yo_start; yo < yo_end; yo += yo_step)
        {
            // Floating-point coordinate
            float yi_f = ((yo + sampling_offset) * scale_y);
            int   yi   = 0;
            if(align_corners)
            {
                yi = utils::rounding::round_half_away_from_zero(yi_f);
            }
            else
            {
                yi = static_cast<int>(std::floor(yi_f));
            }

            for(int xo = xo_start; xo < xo_end; xo += xo_step)
            {
                // Floating-point coordinate
                float xi_f = ((xo + sampling_offset) * scale_x);
                int   xi   = 0;
                if(align_corners)
                {
                    xi = utils::rounding::round_half_away_from_zero(xi_f);
                }
                else
                {
                    xi = static_cast<int>(std::floor(xi_f));
                }

                const uint8_t *in_ptr  = in_ptr_base + xi * in_stride_y + yi * in_stride_z;
                uint8_t       *out_ptr = out_ptr_base + xo * out_stride_y + yo * out_stride_z;

                int cout = 0;
                for(; cout <= (out_dim_ch - step_cout); cout += step_cout)
                {
                    auto out0 = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T)));
                    wrapper::vstore(reinterpret_cast<T *>(out_ptr + cout * sizeof(T)), out0);
                }

                for(; cout < out_dim_ch; ++cout)
                {
                    auto out0                                            = *(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T)));
                    *(reinterpret_cast<T *>(out_ptr + cout * sizeof(T))) = out0;
                }
            }
        }
    }
}

template <typename T>
void bilinear_neon_scale(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                         BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                         bool align_corners, const Window &window)
{
    ARM_COMPUTE_UNUSED(offsets);
    ARM_COMPUTE_UNUSED(dx);
    ARM_COMPUTE_UNUSED(dy);
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    // Compute the ratio between source and destination dimensions
    const float scale_x = scale_utils::calculate_resize_ratio(src->info()->dimension(1), dst->info()->dimension(1), align_corners);
    const float scale_y = scale_utils::calculate_resize_ratio(src->info()->dimension(2), dst->info()->dimension(2), align_corners);

    const int in_stride_y  = src->info()->strides_in_bytes()[1];
    const int in_stride_z  = src->info()->strides_in_bytes()[2];
    const int in_stride_w  = src->info()->strides_in_bytes()[3];
    const int out_stride_y = dst->info()->strides_in_bytes()[1];
    const int out_stride_z = dst->info()->strides_in_bytes()[2];
    const int out_stride_w = dst->info()->strides_in_bytes()[3];
    const int in_dim_w     = src->info()->dimension(1);
    const int in_dim_h     = src->info()->dimension(2);
    const int out_dim_ch   = dst->info()->dimension(0);
    const int step_cout    = 16 / sizeof(T);

    Window window_execution = window;
    window_execution.set(Window::DimX, Window::Dimension(0, 1, 1));
    Window win_in_out(window);
    win_in_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_in_out.set(Window::DimZ, Window::Dimension(0, 0, 0));
    Iterator in(src, win_in_out);
    Iterator out(dst, win_in_out);

    const int xo_start = window_execution.y().start();
    const int xo_end   = window_execution.y().end();
    const int xo_step  = window_execution.y().step();
    const int yo_start = window_execution.z().start();
    const int yo_end   = window_execution.z().end();
    const int yo_step  = window_execution.z().step();
    const int bo_start = window_execution[3].start();
    const int bo_end   = window_execution[3].end();
    const int bo_step  = window_execution[3].step();

    if(border_mode == BorderMode::CONSTANT)
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        using ConstType = typename std::conditional<std::is_same<T, float16_t>::value, half, T>::type;
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        using ConstType = T;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        const T const_border_value = static_cast<T>(constant_border_value.get<ConstType>());

        for(int bo = bo_start; bo < bo_end; bo += bo_step)
        {
            const uint8_t *in_ptr_base  = in.ptr() + bo * in_stride_w;
            uint8_t       *out_ptr_base = out.ptr() + bo * out_stride_w;

            for(int yo = yo_start; yo < yo_end; yo += yo_step)
            {
                // Floating-point coordinate
                const float yi_f = ((yo + sampling_offset) * scale_y - sampling_offset);
                // Integer coordinate
                const auto yi = static_cast<int>(std::floor(yi_f));
                // Weight for the y coordinate
                const auto a1 = (yi_f - static_cast<float>(yi));
                const auto b1 = (1.f - a1);

                for(int xo = xo_start; xo < xo_end; xo += xo_step)
                {
                    // Floating-point coordinate
                    const float xi_f = ((xo + sampling_offset) * scale_x - sampling_offset);
                    // Integer coordinate
                    const auto xi = static_cast<int>(std::floor(xi_f));
                    // Weight for the x coordinate
                    const auto a = (xi_f - static_cast<float>(xi));
                    const auto b = (1.f - a);

                    const auto s00_s = static_cast<T>(b * b1);
                    const auto s01_s = static_cast<T>(a * b1);
                    const auto s10_s = static_cast<T>(b * a1);
                    const auto s11_s = static_cast<T>(a * a1);

                    const uint8_t *in_ptr  = in_ptr_base + xi * in_stride_y + yi * in_stride_z;
                    uint8_t       *out_ptr = out_ptr_base + xo * out_stride_y + yo * out_stride_z;

                    int cout = 0;
                    for(; cout <= (out_dim_ch - step_cout); cout += step_cout)
                    {
                        auto in00 = wrapper::vdup_n(static_cast<T>(const_border_value), ExactTagType{});
                        auto in01 = wrapper::vdup_n(static_cast<T>(const_border_value), ExactTagType{});
                        auto in10 = wrapper::vdup_n(static_cast<T>(const_border_value), ExactTagType{});
                        auto in11 = wrapper::vdup_n(static_cast<T>(const_border_value), ExactTagType{});
                        if((yi >= 0) && (yi < in_dim_h))
                        {
                            if((xi >= 0) && (xi < in_dim_w))
                            {
                                in00 = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T)));
                            }
                            if(((xi + 1) >= 0) && ((xi + 1) < in_dim_w))
                            {
                                in01 = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + in_stride_y));
                            }
                        }
                        if(((yi + 1) >= 0) && ((yi + 1) < in_dim_h))
                        {
                            if((xi >= 0) && (xi < in_dim_w))
                            {
                                in10 = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + in_stride_z));
                            }
                            if(((xi + 1) >= 0) && ((xi + 1) < in_dim_w))
                            {
                                in11 = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + in_stride_y + in_stride_z));
                            }
                        }

                        const auto s00  = wrapper::vdup_n(s00_s, ExactTagType{});
                        const auto s01  = wrapper::vdup_n(s01_s, ExactTagType{});
                        const auto s10  = wrapper::vdup_n(s10_s, ExactTagType{});
                        const auto s11  = wrapper::vdup_n(s11_s, ExactTagType{});
                        auto       out0 = wrapper::vdup_n(static_cast<T>(0), ExactTagType{});
                        out0            = wrapper::vmla(out0, in00, s00);
                        out0            = wrapper::vmla(out0, in01, s01);
                        out0            = wrapper::vmla(out0, in10, s10);
                        out0            = wrapper::vmla(out0, in11, s11);
                        wrapper::vstore(reinterpret_cast<T *>(out_ptr + cout * sizeof(T)), out0);
                    }

                    for(; cout < out_dim_ch; ++cout)
                    {
                        auto in00 = static_cast<T>(const_border_value);
                        auto in01 = static_cast<T>(const_border_value);
                        auto in10 = static_cast<T>(const_border_value);
                        auto in11 = static_cast<T>(const_border_value);
                        if((yi >= 0) && (yi < in_dim_h))
                        {
                            if((xi >= 0) && (xi < in_dim_w))
                            {
                                in00 = *(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T)));
                            }
                            if(((xi + 1) >= 0) && ((xi + 1) < in_dim_w))
                            {
                                in01 = *(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + in_stride_y));
                            }
                        }
                        if(((yi + 1) >= 0) && ((yi + 1) < in_dim_h))
                        {
                            if((xi >= 0) && (xi < in_dim_w))
                            {
                                in10 = *(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + in_stride_z));
                            }
                            if(((xi + 1) >= 0) && ((xi + 1) < in_dim_w))
                            {
                                in11 = *(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + in_stride_y + in_stride_z));
                            }
                        }
                        auto out0 = static_cast<T>(0);
                        out0 += in00 * s00_s;
                        out0 += in01 * s01_s;
                        out0 += in10 * s10_s;
                        out0 += in11 * s11_s;
                        *(reinterpret_cast<T *>(out_ptr + cout * sizeof(T))) = out0;
                    }
                }
            }
        }
    }
    else if(border_mode == BorderMode::REPLICATE)
    {
        for(int bo = bo_start; bo < bo_end; bo += bo_step)
        {
            const uint8_t *in_ptr  = in.ptr() + bo * in_stride_w;
            uint8_t       *out_ptr = out.ptr() + bo * out_stride_w;

            for(int yo = yo_start; yo < yo_end; yo += yo_step)
            {
                // Floating-point coordinate
                const float yi_f = ((yo + sampling_offset) * scale_y - sampling_offset);
                // Integer coordinate
                const auto yi = static_cast<int>(std::floor(yi_f));
                // Weight for the y coordinate
                const auto a1 = (yi_f - static_cast<float>(yi));
                const auto b1 = (1.f - a1);

                const int yi0 = utility::clamp<int>(yi, 0, in_dim_h - 1);
                const int yi1 = utility::clamp<int>(yi + 1, 0, in_dim_h - 1);

                const int yi0_offset = yi0 * in_stride_z;
                const int yi1_offset = yi1 * in_stride_z;

                const int y_offset = yo * out_stride_z;
                for(int xo = xo_start; xo < xo_end; xo += xo_step)
                {
                    // Floating-point coordinate
                    const float xi_f = ((xo + sampling_offset) * scale_x - sampling_offset);
                    // Integer coordinate
                    const auto xi = static_cast<int>(std::floor(xi_f));
                    // Weight for the x coordinate
                    const auto a = (xi_f - static_cast<float>(xi));
                    const auto b = (1.f - a);

                    const auto s00_s = static_cast<T>(b * b1);
                    const auto s01_s = static_cast<T>(a * b1);
                    const auto s10_s = static_cast<T>(b * a1);
                    const auto s11_s = static_cast<T>(a * a1);

                    const auto s00 = wrapper::vdup_n(s00_s, ExactTagType{});
                    const auto s01 = wrapper::vdup_n(s01_s, ExactTagType{});
                    const auto s10 = wrapper::vdup_n(s10_s, ExactTagType{});
                    const auto s11 = wrapper::vdup_n(s11_s, ExactTagType{});

                    const int xi0 = utility::clamp<int>(xi, 0, in_dim_w - 1);
                    const int xi1 = utility::clamp<int>(xi + 1, 0, in_dim_w - 1);

                    const int xi0_offset = xi0 * in_stride_y;
                    const int xi1_offset = xi1 * in_stride_y;

                    const int offset = xo * out_stride_y + y_offset;

                    int cout = 0;
                    for(; cout <= (out_dim_ch - step_cout); cout += step_cout)
                    {
                        const auto in00 = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + xi0_offset + yi0_offset));
                        const auto in01 = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + xi1_offset + yi0_offset));
                        const auto in10 = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + xi0_offset + yi1_offset));
                        const auto in11 = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + xi1_offset + yi1_offset));

                        auto out0 = wrapper::vmul(in00, s00);
                        out0      = wrapper::vmla(out0, in01, s01);
                        out0      = wrapper::vmla(out0, in10, s10);
                        out0      = wrapper::vmla(out0, in11, s11);
                        wrapper::vstore(reinterpret_cast<T *>(out_ptr + offset + cout * sizeof(T)), out0);
                    }

                    for(; cout < out_dim_ch; ++cout)
                    {
                        const T in00 = *(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + xi0_offset + yi0_offset));
                        const T in01 = *(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + xi1_offset + yi0_offset));
                        const T in10 = *(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + xi0_offset + yi1_offset));
                        const T in11 = *(reinterpret_cast<const T *>(in_ptr + cout * sizeof(T) + xi1_offset + yi1_offset));

                        T out0 = in00 * s00_s;
                        out0 += in01 * s01_s;
                        out0 += in10 * s10_s;
                        out0 += in11 * s11_s;
                        *(reinterpret_cast<T *>(out_ptr + offset + cout * sizeof(T))) = out0;
                    }
                }
            }
        }
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
}

template <typename T>
void common_neon_scale(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                       InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                       bool align_corners, const Window &window)
{
    if(policy == InterpolationPolicy::BILINEAR)
    {
        bilinear_neon_scale<T>(src, dst, offsets, dx, dy, border_mode, constant_border_value, sampling_offset, align_corners, window);
    }
    else if(policy == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        nearest_neon_scale<T>(src, dst, offsets, sampling_offset, align_corners, window);
    }
}
} // namespace cpu
} // namespace arm_compute

#endif /* SRC_CORE_NEON_KERNELS_SCALE_LIST_H */
