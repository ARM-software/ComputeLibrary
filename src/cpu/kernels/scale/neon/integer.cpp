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
#include "arm_compute/core/Helpers.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/ScaleHelpers.h"
#include "src/core/utils/ScaleUtils.h"
#include "support/Rounding.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
void u8_neon_scale_nearest(const ITensor *src, ITensor *dst, const ITensor *offsets,
                           float sampling_offset, bool align_corners, const Window &window)
{
    const size_t in_stride_c  = src->info()->dimension(0) + src->info()->padding().left + src->info()->padding().right;
    const size_t in_stride_w  = src->info()->dimension(1) + src->info()->padding().top + src->info()->padding().bottom;
    const size_t in_stride_wc = in_stride_w * in_stride_c;
    const size_t in_dim_h     = src->info()->dimension(2);

    // Compute the ratio between source height and destination height
    const auto hr             = scale_utils::calculate_resize_ratio(in_dim_h, dst->info()->dimension(2), align_corners);
    const auto window_start_x = static_cast<int32_t>(window.x().start());
    const auto window_end_x   = static_cast<int32_t>(window.x().end());
    const int  window_step_x  = 16;

    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator out(dst, win);

    const uint8_t     *in_ptr_start        = src->buffer() + src->info()->offset_first_element_in_bytes();
    const unsigned int in_stride_bytes_hwc = src->info()->strides_in_bytes()[3];

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const int32_t  offset     = *reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z()))) * in_stride_c;
        const auto     in_hi      = static_cast<int>(align_corners ? utils::rounding::round_half_away_from_zero((id.z() + sampling_offset) * hr) : std::floor((id.z() + sampling_offset) * hr));
        const int      offset_row = in_hi * in_stride_wc;
        int32_t        x          = window_start_x;
        const uint8_t *in_ptr     = reinterpret_cast<const uint8_t *>(in_ptr_start + in_stride_bytes_hwc * id[3]);

        for(; x <= window_end_x - window_step_x; x += window_step_x)
        {
            wrapper::vstore(reinterpret_cast<uint8_t *>(out.ptr()) + x,
                            wrapper::vloadq(in_ptr + offset + offset_row + x));
        }
        for(; x < window_end_x; ++x)
        {
            *(reinterpret_cast<uint8_t *>(out.ptr()) + x) = *(in_ptr + offset + offset_row + x);
        }
    },
    out);
}

void u8_neon_scale_bilinear(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                            BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                            bool align_corners, const Window &window)
{
    // Compute the ratio between source and destination dimensions
    const float scale_x = scale_utils::calculate_resize_ratio(src->info()->dimension(1), dst->info()->dimension(1), align_corners);
    const float scale_y = scale_utils::calculate_resize_ratio(src->info()->dimension(2), dst->info()->dimension(2), align_corners);

    const int input_width  = src->info()->dimension(1);
    const int input_height = src->info()->dimension(2);

    if(border_mode == BorderMode::CONSTANT)
    {
        Iterator  out(dst, window);
        const int in_stride_c  = src->info()->dimension(0) + src->info()->padding().left + src->info()->padding().right;
        const int in_stride_wc = in_stride_c * (input_width + src->info()->padding().top + src->info()->padding().bottom);

        // Don't increment in Y and Z direction for the input tensor
        // A pointer to the start of this plane is needed as base for the precomputed offsets
        Window win_in(window);
        win_in.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_in.set(Window::DimZ, Window::Dimension(0, 0, 0));
        Iterator in(src, win_in);

        const uint8_t const_border_value = static_cast<uint8_t>(constant_border_value.get<uint8_t>());
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const auto     offset = *reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto     dx_val = *reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto     dy_val = *reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id.y(), id.z())));
            const int32_t  in_hi  = std::floor((id.z() + sampling_offset) * scale_y - sampling_offset);
            const uint8_t *in_ptr = reinterpret_cast<const uint8_t *>(in.ptr()) + offset * in_stride_c + in_hi * in_stride_wc;

            const auto a00 = (0 <= offset && offset < input_width && 0 <= in_hi && in_hi < input_height) ? *in_ptr : const_border_value;
            const auto a01 = (-1 <= offset && offset < input_width - 1 && 0 <= in_hi && in_hi < input_height) ? *(in_ptr + in_stride_c) : const_border_value;
            const auto a10 = (0 <= offset && offset < input_width && -1 <= in_hi && in_hi < input_height - 1) ? *(in_ptr + in_stride_wc) : const_border_value;
            const auto a11 = (-1 <= offset && offset < input_width - 1 && -1 <= in_hi && in_hi < input_height - 1) ? *(in_ptr + in_stride_c + in_stride_wc) : const_border_value;

            *reinterpret_cast<uint8_t *>(out.ptr()) = static_cast<uint8_t>(scale_helpers::delta_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        in, out);
    }
    else if(border_mode == BorderMode::REPLICATE)
    {
        using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<float, wrapper::traits::BitWidth::W128>;

        const int in_stride_x  = src->info()->strides_in_bytes()[1];
        const int in_stride_y  = src->info()->strides_in_bytes()[2];
        const int in_stride_b  = src->info()->strides_in_bytes()[3];
        const int out_stride_x = dst->info()->strides_in_bytes()[1];
        const int out_stride_y = dst->info()->strides_in_bytes()[2];
        const int out_stride_b = dst->info()->strides_in_bytes()[3];

        const int     out_dim_ch = dst->info()->dimension(0);
        constexpr int step_cout  = 16;

        Window window_execution = window;
        window_execution.set(Window::DimX, Window::Dimension(0, 1, 1));
        Window win_in_out(window);
        win_in_out.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_in_out.set(Window::DimZ, Window::Dimension(0, 0, 0));
        Iterator in(src, win_in_out);
        Iterator out(dst, win_in_out);

        const int xo_start = window_execution[1].start();
        const int xo_end   = window_execution[1].end();
        const int xo_step  = window_execution[1].step();
        const int yo_start = window_execution[2].start();
        const int yo_end   = window_execution[2].end();
        const int yo_step  = window_execution[2].step();
        const int bo_start = window_execution[3].start();
        const int bo_end   = window_execution[3].end();
        const int bo_step  = window_execution[3].step();

        const float fp_coord_offset_y = sampling_offset * (scale_y - 1);
        const float fp_coord_offset_x = sampling_offset * (scale_x - 1);

        for(int bo = bo_start; bo < bo_end; bo += bo_step)
        {
            const uint8_t *in_ptr  = in.ptr() + bo * in_stride_b;
            uint8_t       *out_ptr = out.ptr() + bo * out_stride_b;

            for(int yo = yo_start; yo < yo_end; yo += yo_step)
            {
                // Floating-point coordinate
                const float yi_f = yo * scale_y + fp_coord_offset_y;
                // Integer coordinate
                const int yi = static_cast<int>(std::floor(yi_f));
                // Weight for the y coordinate
                const float a1 = (yi_f - static_cast<float>(yi));
                const float b1 = (1.f - a1);

                const int yi0 = utility::clamp<int>(yi, 0, input_height - 1);
                const int yi1 = utility::clamp<int>(yi + 1, 0, input_height - 1);

                const uint8_t *in_ptr_yi0 = in_ptr + yi0 * in_stride_y;
                const uint8_t *in_ptr_yi1 = in_ptr + yi1 * in_stride_y;

                uint8_t *out_ptr_yo = out_ptr + yo * out_stride_y;
                for(int xo = xo_start; xo < xo_end; xo += xo_step)
                {
                    // Floating-point coordinate
                    const float xi_f = xo * scale_x + fp_coord_offset_x;
                    // Integer coordinate
                    const int xi = static_cast<int>(std::floor(xi_f));
                    // Weight for the x coordinate
                    const float a = (xi_f - static_cast<float>(xi));
                    const float b = (1.f - a);

                    const float s00_s = b * b1;
                    const float s01_s = a * b1;
                    const float s10_s = b * a1;
                    const float s11_s = a * a1;

                    const auto s00 = wrapper::vdup_n(s00_s, ExactTagType{});
                    const auto s01 = wrapper::vdup_n(s01_s, ExactTagType{});
                    const auto s10 = wrapper::vdup_n(s10_s, ExactTagType{});
                    const auto s11 = wrapper::vdup_n(s11_s, ExactTagType{});

                    const int xi0 = utility::clamp<int>(xi, 0, input_width - 1);
                    const int xi1 = utility::clamp<int>(xi + 1, 0, input_width - 1);

                    const auto in_ptr_xi0_yi0 = in_ptr_yi0 + xi0 * in_stride_x;
                    const auto in_ptr_xi1_yi0 = in_ptr_yi0 + xi1 * in_stride_x;
                    const auto in_ptr_xi0_yi1 = in_ptr_yi1 + xi0 * in_stride_x;
                    const auto in_ptr_xi1_yi1 = in_ptr_yi1 + xi1 * in_stride_x;

                    uint8_t *out_ptr_xo_yo = out_ptr_yo + xo * out_stride_x;

                    int cout = 0;
                    for(; cout <= (out_dim_ch - step_cout); cout += step_cout)
                    {
                        const auto in00 = wrapper::vloadq(in_ptr_xi0_yi0 + cout * sizeof(uint8_t));
                        const auto in01 = wrapper::vloadq(in_ptr_xi1_yi0 + cout * sizeof(uint8_t));
                        const auto in10 = wrapper::vloadq(in_ptr_xi0_yi1 + cout * sizeof(uint8_t));
                        const auto in11 = wrapper::vloadq(in_ptr_xi1_yi1 + cout * sizeof(uint8_t));

                        const uint16x8_t in00_low  = wrapper::vmovl(wrapper::vgetlow(in00));
                        const uint16x8_t in00_high = wrapper::vmovl(wrapper::vgethigh(in00));

                        const auto in00_0 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in00_low)));
                        const auto in00_1 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in00_low)));
                        const auto in00_2 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in00_high)));
                        const auto in00_3 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in00_high)));

                        const uint16x8_t in01_low  = wrapper::vmovl(wrapper::vgetlow(in01));
                        const uint16x8_t in01_high = wrapper::vmovl(wrapper::vgethigh(in01));

                        const auto in01_0 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in01_low)));
                        const auto in01_1 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in01_low)));
                        const auto in01_2 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in01_high)));
                        const auto in01_3 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in01_high)));

                        const uint16x8_t in10_low  = wrapper::vmovl(wrapper::vgetlow(in10));
                        const uint16x8_t in10_high = wrapper::vmovl(wrapper::vgethigh(in10));

                        const auto in10_0 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in10_low)));
                        const auto in10_1 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in10_low)));
                        const auto in10_2 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in10_high)));
                        const auto in10_3 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in10_high)));

                        const uint16x8_t in11_low  = wrapper::vmovl(wrapper::vgetlow(in11));
                        const uint16x8_t in11_high = wrapper::vmovl(wrapper::vgethigh(in11));

                        const auto in11_0 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in11_low)));
                        const auto in11_1 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in11_low)));
                        const auto in11_2 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in11_high)));
                        const auto in11_3 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in11_high)));

                        auto out_0 = wrapper::vmul(in00_0, s00);
                        out_0      = wrapper::vmla(out_0, in01_0, s01);
                        out_0      = wrapper::vmla(out_0, in10_0, s10);
                        out_0      = wrapper::vmla(out_0, in11_0, s11);

                        auto out_1 = wrapper::vmul(in00_1, s00);
                        out_1      = wrapper::vmla(out_1, in01_1, s01);
                        out_1      = wrapper::vmla(out_1, in10_1, s10);
                        out_1      = wrapper::vmla(out_1, in11_1, s11);

                        auto out_2 = wrapper::vmul(in00_2, s00);
                        out_2      = wrapper::vmla(out_2, in01_2, s01);
                        out_2      = wrapper::vmla(out_2, in10_2, s10);
                        out_2      = wrapper::vmla(out_2, in11_2, s11);

                        auto out_3 = wrapper::vmul(in00_3, s00);
                        out_3      = wrapper::vmla(out_3, in01_3, s01);
                        out_3      = wrapper::vmla(out_3, in10_3, s10);
                        out_3      = wrapper::vmla(out_3, in11_3, s11);

#if defined(__aarch64__) && !defined(BARE_METAL)
                        const auto out_0_int = wrapper::vcvta<uint32_t>(out_0);
                        const auto out_1_int = wrapper::vcvta<uint32_t>(out_1);
                        const auto out_2_int = wrapper::vcvta<uint32_t>(out_2);
                        const auto out_3_int = wrapper::vcvta<uint32_t>(out_3);
#else  // defined(__aarch64__) && !defined(BARE_METAL)
                        const auto out_0_int = wrapper::vcvt<uint32_t>(out_0);
                        const auto out_1_int = wrapper::vcvt<uint32_t>(out_1);
                        const auto out_2_int = wrapper::vcvt<uint32_t>(out_2);
                        const auto out_3_int = wrapper::vcvt<uint32_t>(out_3);
#endif // defined(__aarch64__) && !defined(BARE_METAL)
                        const auto low_part  = wrapper::vqmovn(wrapper::vcombine(wrapper::vqmovn(out_0_int), wrapper::vqmovn(out_1_int)));
                        const auto high_part = wrapper::vqmovn(wrapper::vcombine(wrapper::vqmovn(out_2_int), wrapper::vqmovn(out_3_int)));
                        const auto out       = wrapper::vcombine(low_part, high_part);

                        wrapper::vstore(out_ptr_xo_yo + cout * sizeof(uint8_t), out);
                    }

                    for(; cout < out_dim_ch; ++cout)
                    {
                        const uint8_t in00 = *(in_ptr_xi0_yi0 + cout * sizeof(uint8_t));
                        const uint8_t in01 = *(in_ptr_xi1_yi0 + cout * sizeof(uint8_t));
                        const uint8_t in10 = *(in_ptr_xi0_yi1 + cout * sizeof(uint8_t));
                        const uint8_t in11 = *(in_ptr_xi1_yi1 + cout * sizeof(uint8_t));

                        float out0 = in00 * s00_s;
                        out0 += in01 * s01_s;
                        out0 += in10 * s10_s;
                        out0 += in11 * s11_s;

                        // Rounding modes of vector and scalar loops should match
#if defined(__aarch64__) && !defined(BARE_METAL)
                        *(out_ptr_xo_yo + cout * sizeof(uint8_t)) = static_cast<uint8_t>(std::round(out0));
#else  // defined(__aarch64__) && !defined(BARE_METAL)
                        *(out_ptr_xo_yo + cout * sizeof(uint8_t)) = static_cast<uint8_t>(out0);
#endif // defined(__aarch64__) && !defined(BARE_METAL)
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

void s8_neon_scale_bilinear(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                            BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                            bool align_corners, const Window &window)
{
    ARM_COMPUTE_UNUSED(dx, dy, offsets, constant_border_value);
    if(border_mode == BorderMode::REPLICATE)
    {
        using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<float, wrapper::traits::BitWidth::W128>;

        // Compute the ratio between source and destination dimensions
        const float scale_x = scale_utils::calculate_resize_ratio(src->info()->dimension(1), dst->info()->dimension(1), align_corners);
        const float scale_y = scale_utils::calculate_resize_ratio(src->info()->dimension(2), dst->info()->dimension(2), align_corners);

        const int     in_stride_x  = src->info()->strides_in_bytes()[1];
        const int     in_stride_y  = src->info()->strides_in_bytes()[2];
        const int     in_stride_b  = src->info()->strides_in_bytes()[3];
        const int     out_stride_x = dst->info()->strides_in_bytes()[1];
        const int     out_stride_y = dst->info()->strides_in_bytes()[2];
        const int     out_stride_b = dst->info()->strides_in_bytes()[3];
        const int     input_width  = src->info()->dimension(1);
        const int     input_height = src->info()->dimension(2);
        const int     out_dim_ch   = dst->info()->dimension(0);
        constexpr int step_cout    = 16;

        Window window_execution = window;
        window_execution.set(Window::DimX, Window::Dimension(0, 1, 1));
        Window win_in_out(window);
        win_in_out.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_in_out.set(Window::DimZ, Window::Dimension(0, 0, 0));
        Iterator in(src, win_in_out);
        Iterator out(dst, win_in_out);

        const int xo_start = window_execution[1].start();
        const int xo_end   = window_execution[1].end();
        const int xo_step  = window_execution[1].step();
        const int yo_start = window_execution[2].start();
        const int yo_end   = window_execution[2].end();
        const int yo_step  = window_execution[2].step();
        const int bo_start = window_execution[3].start();
        const int bo_end   = window_execution[3].end();
        const int bo_step  = window_execution[3].step();

        const float fp_coord_offset_y = sampling_offset * (scale_y - 1);
        const float fp_coord_offset_x = sampling_offset * (scale_x - 1);

        for(int bo = bo_start; bo < bo_end; bo += bo_step)
        {
            const int8_t *in_ptr  = reinterpret_cast<int8_t *>(in.ptr() + bo * in_stride_b);
            int8_t       *out_ptr = reinterpret_cast<int8_t *>(out.ptr() + bo * out_stride_b);

            for(int yo = yo_start; yo < yo_end; yo += yo_step)
            {
                // Floating-point coordinate
                const float yi_f = yo * scale_y + fp_coord_offset_y;
                // Integer coordinate
                const int yi = static_cast<int>(std::floor(yi_f));
                // Weight for the y coordinate
                const float a1 = (yi_f - static_cast<float>(yi));
                const float b1 = (1.f - a1);

                const int yi0 = utility::clamp<int>(yi, 0, input_height - 1);
                const int yi1 = utility::clamp<int>(yi + 1, 0, input_height - 1);

                const int8_t *in_ptr_yi0 = in_ptr + yi0 * in_stride_y;
                const int8_t *in_ptr_yi1 = in_ptr + yi1 * in_stride_y;

                int8_t *out_ptr_yo = out_ptr + yo * out_stride_y;
                for(int xo = xo_start; xo < xo_end; xo += xo_step)
                {
                    // Floating-point coordinate
                    const float xi_f = xo * scale_x + fp_coord_offset_x;
                    // Integer coordinate
                    const int xi = static_cast<int>(std::floor(xi_f));
                    // Weight for the x coordinate
                    const float a = (xi_f - static_cast<float>(xi));
                    const float b = (1.f - a);

                    const float s00_s = b * b1;
                    const float s01_s = a * b1;
                    const float s10_s = b * a1;
                    const float s11_s = a * a1;

                    const auto s00 = wrapper::vdup_n(s00_s, ExactTagType{});
                    const auto s01 = wrapper::vdup_n(s01_s, ExactTagType{});
                    const auto s10 = wrapper::vdup_n(s10_s, ExactTagType{});
                    const auto s11 = wrapper::vdup_n(s11_s, ExactTagType{});

                    const int xi0 = utility::clamp<int>(xi, 0, input_width - 1);
                    const int xi1 = utility::clamp<int>(xi + 1, 0, input_width - 1);

                    const auto in_ptr_xi0_yi0 = in_ptr_yi0 + xi0 * in_stride_x;
                    const auto in_ptr_xi1_yi0 = in_ptr_yi0 + xi1 * in_stride_x;
                    const auto in_ptr_xi0_yi1 = in_ptr_yi1 + xi0 * in_stride_x;
                    const auto in_ptr_xi1_yi1 = in_ptr_yi1 + xi1 * in_stride_x;

                    int8_t *out_ptr_xo_yo = out_ptr_yo + xo * out_stride_x;

                    int cout = 0;
                    for(; cout <= (out_dim_ch - step_cout); cout += step_cout)
                    {
                        const auto in00 = wrapper::vloadq(in_ptr_xi0_yi0 + cout * sizeof(int8_t));
                        const auto in01 = wrapper::vloadq(in_ptr_xi1_yi0 + cout * sizeof(int8_t));
                        const auto in10 = wrapper::vloadq(in_ptr_xi0_yi1 + cout * sizeof(int8_t));
                        const auto in11 = wrapper::vloadq(in_ptr_xi1_yi1 + cout * sizeof(int8_t));

                        const int16x8_t in00_low  = wrapper::vmovl(wrapper::vgetlow(in00));
                        const int16x8_t in00_high = wrapper::vmovl(wrapper::vgethigh(in00));

                        const auto in00_0 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in00_low)));
                        const auto in00_1 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in00_low)));
                        const auto in00_2 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in00_high)));
                        const auto in00_3 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in00_high)));

                        const int16x8_t in01_low  = wrapper::vmovl(wrapper::vgetlow(in01));
                        const int16x8_t in01_high = wrapper::vmovl(wrapper::vgethigh(in01));

                        const auto in01_0 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in01_low)));
                        const auto in01_1 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in01_low)));
                        const auto in01_2 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in01_high)));
                        const auto in01_3 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in01_high)));

                        const int16x8_t in10_low  = wrapper::vmovl(wrapper::vgetlow(in10));
                        const int16x8_t in10_high = wrapper::vmovl(wrapper::vgethigh(in10));

                        const auto in10_0 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in10_low)));
                        const auto in10_1 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in10_low)));
                        const auto in10_2 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in10_high)));
                        const auto in10_3 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in10_high)));

                        const int16x8_t in11_low  = wrapper::vmovl(wrapper::vgetlow(in11));
                        const int16x8_t in11_high = wrapper::vmovl(wrapper::vgethigh(in11));

                        const auto in11_0 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in11_low)));
                        const auto in11_1 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in11_low)));
                        const auto in11_2 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgetlow(in11_high)));
                        const auto in11_3 = wrapper::vcvt<float>(wrapper::vmovl(wrapper::vgethigh(in11_high)));

                        auto out_0 = wrapper::vmul(in00_0, s00);
                        out_0      = wrapper::vmla(out_0, in01_0, s01);
                        out_0      = wrapper::vmla(out_0, in10_0, s10);
                        out_0      = wrapper::vmla(out_0, in11_0, s11);

                        auto out_1 = wrapper::vmul(in00_1, s00);
                        out_1      = wrapper::vmla(out_1, in01_1, s01);
                        out_1      = wrapper::vmla(out_1, in10_1, s10);
                        out_1      = wrapper::vmla(out_1, in11_1, s11);

                        auto out_2 = wrapper::vmul(in00_2, s00);
                        out_2      = wrapper::vmla(out_2, in01_2, s01);
                        out_2      = wrapper::vmla(out_2, in10_2, s10);
                        out_2      = wrapper::vmla(out_2, in11_2, s11);

                        auto out_3 = wrapper::vmul(in00_3, s00);
                        out_3      = wrapper::vmla(out_3, in01_3, s01);
                        out_3      = wrapper::vmla(out_3, in10_3, s10);
                        out_3      = wrapper::vmla(out_3, in11_3, s11);

#if defined(__aarch64__) && !defined(BARE_METAL)
                        const auto out_0_int = wrapper::vcvta<int32_t>(out_0);
                        const auto out_1_int = wrapper::vcvta<int32_t>(out_1);
                        const auto out_2_int = wrapper::vcvta<int32_t>(out_2);
                        const auto out_3_int = wrapper::vcvta<int32_t>(out_3);
#else  // defined(__aarch64__) && !defined(BARE_METAL)
                        const auto out_0_int                      = wrapper::vcvt<int32_t>(out_0);
                        const auto out_1_int                      = wrapper::vcvt<int32_t>(out_1);
                        const auto out_2_int                      = wrapper::vcvt<int32_t>(out_2);
                        const auto out_3_int                      = wrapper::vcvt<int32_t>(out_3);
#endif // defined(__aarch64__) && !defined(BARE_METAL)
                        const auto low_part  = wrapper::vqmovn(wrapper::vcombine(wrapper::vqmovn(out_0_int), wrapper::vqmovn(out_1_int)));
                        const auto high_part = wrapper::vqmovn(wrapper::vcombine(wrapper::vqmovn(out_2_int), wrapper::vqmovn(out_3_int)));
                        const auto out       = wrapper::vcombine(low_part, high_part);

                        wrapper::vstore(out_ptr_xo_yo + cout * sizeof(int8_t), out);
                    }

                    for(; cout < out_dim_ch; ++cout)
                    {
                        const int8_t in00 = *(in_ptr_xi0_yi0 + cout * sizeof(int8_t));
                        const int8_t in01 = *(in_ptr_xi1_yi0 + cout * sizeof(int8_t));
                        const int8_t in10 = *(in_ptr_xi0_yi1 + cout * sizeof(int8_t));
                        const int8_t in11 = *(in_ptr_xi1_yi1 + cout * sizeof(int8_t));

                        float out0 = in00 * s00_s;
                        out0 += in01 * s01_s;
                        out0 += in10 * s10_s;
                        out0 += in11 * s11_s;

                        // Rounding modes of vector and scalar loops should match
#if defined(__aarch64__) && !defined(BARE_METAL)
                        *(out_ptr_xo_yo + cout * sizeof(int8_t)) = static_cast<int8_t>(std::round(out0));
#else  // defined(__aarch64__) && !defined(BARE_METAL)
                        *(out_ptr_xo_yo + cout * sizeof(int8_t))  = static_cast<int8_t>(out0);
#endif // defined(__aarch64__) && !defined(BARE_METAL)
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

void s16_neon_scale_nearest(const ITensor *src, ITensor *dst, const ITensor *offsets,
                            float sampling_offset, bool align_corners, const Window &window)
{
    const size_t in_stride_c  = src->info()->dimension(0) + src->info()->padding().left + src->info()->padding().right;
    const size_t in_stride_w  = src->info()->dimension(1) + src->info()->padding().top + src->info()->padding().bottom;
    const size_t in_stride_wc = in_stride_w * in_stride_c;
    const size_t in_dim_h     = src->info()->dimension(2);

    // Compute the ratio between source height and destination height
    const auto hr             = scale_utils::calculate_resize_ratio(in_dim_h, dst->info()->dimension(2), align_corners);
    const auto window_start_x = static_cast<int32_t>(window.x().start());
    const auto window_end_x   = static_cast<int32_t>(window.x().end());
    const int  window_step_x  = 8;

    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator out(dst, win);

    const uint8_t     *in_ptr_start        = src->buffer() + src->info()->offset_first_element_in_bytes();
    const unsigned int in_stride_bytes_hwc = src->info()->strides_in_bytes()[3];

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const int32_t  offset     = *reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z()))) * in_stride_c;
        const auto     in_hi      = static_cast<int>(align_corners ? utils::rounding::round_half_away_from_zero((id.z() + sampling_offset) * hr) : std::floor((id.z() + sampling_offset) * hr));
        const int      offset_row = in_hi * in_stride_wc;
        int32_t        x          = window_start_x;
        const int16_t *in_ptr     = reinterpret_cast<const int16_t *>(in_ptr_start + in_stride_bytes_hwc * id[3]);

        for(; x <= window_end_x - window_step_x; x += window_step_x)
        {
            wrapper::vstore(reinterpret_cast<int16_t *>(out.ptr()) + x,
                            wrapper::vloadq(in_ptr + offset + offset_row + x));
        }
        for(; x < window_end_x; ++x)
        {
            *(reinterpret_cast<int16_t *>(out.ptr()) + x) = *(in_ptr + offset + offset_row + x);
        }
    },
    out);
}

void s16_neon_scale_bilinear(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                             BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                             bool align_corners, const Window &window)
{
    // Compute the ratio between source height and destination height
    const auto hr = scale_utils::calculate_resize_ratio(src->info()->dimension(2), dst->info()->dimension(2), align_corners);

    Iterator  out(dst, window);
    const int in_stride_c  = src->info()->dimension(0) + src->info()->padding().left + src->info()->padding().right;
    const int in_dim_w     = src->info()->dimension(1);
    const int in_dim_h     = src->info()->dimension(2);
    const int in_stride_wc = in_stride_c * (in_dim_w + src->info()->padding().top + src->info()->padding().bottom);

    // Don't increment in Y and Z direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimZ, Window::Dimension(0, 0, 0));
    Iterator in(src, win_in);

    if(border_mode == BorderMode::CONSTANT)
    {
        const int16_t const_border_value = static_cast<int16_t>(constant_border_value.get<int16_t>());
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const auto     offset = *reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto     dx_val = *reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto     dy_val = *reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id.y(), id.z())));
            const int32_t  in_hi  = std::floor((id.z() + sampling_offset) * hr - sampling_offset);
            const int16_t *in_ptr = reinterpret_cast<const int16_t *>(in.ptr()) + offset * in_stride_c + in_hi * in_stride_wc;

            const auto a00 = (0 <= offset && offset < in_dim_w && 0 <= in_hi && in_hi < in_dim_h) ? *in_ptr : const_border_value;
            const auto a01 = (-1 <= offset && offset < in_dim_w - 1 && 0 <= in_hi && in_hi < in_dim_h) ? *(in_ptr + in_stride_c) : const_border_value;
            const auto a10 = (0 <= offset && offset < in_dim_w && -1 <= in_hi && in_hi < in_dim_h - 1) ? *(in_ptr + in_stride_wc) : const_border_value;
            const auto a11 = (-1 <= offset && offset < in_dim_w - 1 && -1 <= in_hi && in_hi < in_dim_h - 1) ? *(in_ptr + in_stride_c + in_stride_wc) : const_border_value;

            *reinterpret_cast<int16_t *>(out.ptr()) = static_cast<int16_t>(scale_helpers::delta_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        in, out);
    }
    else if(border_mode == BorderMode::REPLICATE)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const auto offset = *reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto dx_val = *reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto dy_val = *reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id.y(), id.z())));
            const int  in_hi  = std::floor((id.z() + sampling_offset) * hr - sampling_offset);

            const auto clamped_w  = utility::clamp<int>(offset, 0, in_dim_w - 1);
            const auto clamped_w1 = utility::clamp<int>(offset + 1, 0, in_dim_w - 1);
            const auto clamped_h  = utility::clamp<int>(in_hi, 0, in_dim_h - 1);
            const auto clamped_h1 = utility::clamp<int>(in_hi + 1, 0, in_dim_h - 1);

            const auto a00 = *(reinterpret_cast<const int16_t *>(in.ptr()) + clamped_w * in_stride_c + clamped_h * in_stride_wc);
            const auto a01 = *(reinterpret_cast<const int16_t *>(in.ptr()) + clamped_w1 * in_stride_c + clamped_h * in_stride_wc);
            const auto a10 = *(reinterpret_cast<const int16_t *>(in.ptr()) + clamped_w * in_stride_c + clamped_h1 * in_stride_wc);
            const auto a11 = *(reinterpret_cast<const int16_t *>(in.ptr()) + clamped_w1 * in_stride_c + clamped_h1 * in_stride_wc);

            *reinterpret_cast<int16_t *>(out.ptr()) = static_cast<int16_t>(scale_helpers::delta_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        in, out);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
}
}
namespace cpu
{
void s8_neon_scale(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                   InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                   bool align_corners, const Window &window)
{
    if(policy == InterpolationPolicy::BILINEAR)
    {
        s8_neon_scale_bilinear(src, dst, offsets, dx, dy, border_mode, constant_border_value, sampling_offset, align_corners, window);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
}

void u8_neon_scale(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                   InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                   bool align_corners, const Window &window)
{
    if(policy == InterpolationPolicy::BILINEAR)
    {
        u8_neon_scale_bilinear(src, dst, offsets, dx, dy, border_mode, constant_border_value, sampling_offset, align_corners, window);
    }
    else if(policy == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        u8_neon_scale_nearest(src, dst, offsets, sampling_offset, align_corners, window);
    }
}

void s16_neon_scale(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                    InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                    bool align_corners, const Window &window)
{
    if(policy == InterpolationPolicy::BILINEAR)
    {
        s16_neon_scale_bilinear(src, dst, offsets, dx, dy, border_mode, constant_border_value, sampling_offset, align_corners, window);
    }
    else if(policy == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        s16_neon_scale_nearest(src, dst, offsets, sampling_offset, align_corners, window);
    }
}
} // namespace cpu
} // namespace arm_compute