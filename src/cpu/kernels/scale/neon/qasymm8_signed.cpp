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
#include "src/core/helpers/ScaleHelpers.h"
#include "src/cpu/kernels/scale/neon/list.h"

namespace arm_compute
{
namespace
{
void qasymm8_signed_neon_scale_bilinear(const ITensor *src,
                                        ITensor       *dst,
                                        const ITensor *offsets,
                                        const ITensor *dx,
                                        const ITensor *dy,
                                        BorderMode     border_mode,
                                        PixelValue     constant_border_value,
                                        float          sampling_offset,
                                        bool           align_corners,
                                        const Window  &window)
{
    // Data layout is NHWC
    const UniformQuantizationInfo iq_info = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info = dst->info()->quantization_info().uniform();

    const int32_t input_width  = src->info()->dimension(1);
    const int32_t input_height = src->info()->dimension(2);

    // Compute the ratio between source and destination dimensions
    const float scale_x =
        scale_utils::calculate_resize_ratio(src->info()->dimension(1), dst->info()->dimension(1), align_corners);
    const float scale_y =
        scale_utils::calculate_resize_ratio(src->info()->dimension(2), dst->info()->dimension(2), align_corners);

    if (border_mode == BorderMode::CONSTANT)
    {
        const int32_t in_stride_y = src->info()->strides_in_bytes()[1];
        const int32_t in_stride_z = src->info()->strides_in_bytes()[2];

        Window win_off;
        win_off.set(Window::DimX, Window::Dimension(0, 0, 0));
        win_off.set(Window::DimY, Window::Dimension(0, 0, 0));

        // Don't increment in X and Y direction for the input tensor
        // A pointer to the start of this plane is needed as base for the precomputed offsets
        Window win_in(window);
        win_in.set(1, Window::Dimension(0, 0, 0));
        win_in.set(2, Window::Dimension(0, 0, 0));

        for (size_t d = Window::DimZ; d < offsets->info()->num_dimensions(); ++d)
        {
            win_off.set(d, Window::Dimension(0, 0, 0));
        }

        Iterator in(src, win_in);
        Iterator out(dst, window);

        const int8_t const_border_value = static_cast<int8_t>(constant_border_value.get<int8_t>());
        execute_window_loop(
            window,
            [&](const Coordinates &id)
            {
                const int32_t index_h = std::floor((id[2] + sampling_offset) * scale_y - sampling_offset);
                const int32_t index_w =
                    *(reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id[1], id[2]))));
                const auto dx_val = *(reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id[1], id[2]))));
                const auto dy_val = *(reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id[1], id[2]))));
                const auto pixel_row_ptr = reinterpret_cast<const int8_t *>(in.ptr());

                const auto a00 = (0 <= index_w && index_w < input_width && 0 <= index_h && index_h < input_height)
                                     ? (*(pixel_row_ptr + index_w * in_stride_y + index_h * in_stride_z))
                                     : const_border_value;
                const auto a01 = (-1 <= index_w && index_w + 1 < input_width && 0 <= index_h && index_h < input_height)
                                     ? (*(pixel_row_ptr + (index_w + 1) * in_stride_y + index_h * in_stride_z))
                                     : const_border_value;
                const auto a10 = (0 <= index_w && index_w < input_width && -1 <= index_h && index_h < input_height - 1)
                                     ? (*(pixel_row_ptr + index_w * in_stride_y + (index_h + 1) * in_stride_z))
                                     : const_border_value;
                const auto a11 =
                    (-1 <= index_w && index_w < input_width - 1 && -1 <= index_h && index_h < input_height - 1)
                        ? (*(pixel_row_ptr + (index_w + 1) * in_stride_y + (index_h + 1) * in_stride_z))
                        : const_border_value;

                const float inp00                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a00, iq_info);
                const float inp01                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a01, iq_info);
                const float inp10                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a10, iq_info);
                const float inp11                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a11, iq_info);
                *reinterpret_cast<int8_t *>(out.ptr()) = Qasymm8QuantizationHelper<int8_t>::quantize(
                    scale_helpers::delta_bilinear(inp00, inp01, inp10, inp11, dx_val, dy_val), oq_info);
            },
            in, out);
    }
    else if (border_mode == BorderMode::REPLICATE)
    {
        using FloatTagType = typename wrapper::traits::neon_bitvector_tag_t<float, wrapper::traits::BitWidth::W128>;
        using Int32TagType = typename wrapper::traits::neon_bitvector_tag_t<int32_t, wrapper::traits::BitWidth::W128>;

        const int     in_stride_x  = src->info()->strides_in_bytes()[1];
        const int     in_stride_y  = src->info()->strides_in_bytes()[2];
        const int     in_stride_b  = src->info()->strides_in_bytes()[3];
        const int     out_stride_x = dst->info()->strides_in_bytes()[1];
        const int     out_stride_y = dst->info()->strides_in_bytes()[2];
        const int     out_stride_b = dst->info()->strides_in_bytes()[3];
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

        const UniformQuantizationInfo iq_info = src->info()->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = dst->info()->quantization_info().uniform();

        const float32x4_t vscale_in  = wrapper::vdup_n(iq_info.scale, FloatTagType{});
        const int32x4_t   voffset_in = wrapper::vdup_n(iq_info.offset, Int32TagType{}); // Offsets will be Int32

        const float32x4_t invvscale_o = wrapper::vdup_n(1.f / oq_info.scale, FloatTagType{});
        const float32x4_t voffset_o   = vdupq_n_f32(oq_info.offset);

        for (int bo = bo_start; bo < bo_end; bo += bo_step)
        {
            const int8_t *in_ptr  = reinterpret_cast<int8_t *>(in.ptr() + bo * in_stride_b);
            int8_t       *out_ptr = reinterpret_cast<int8_t *>(out.ptr() + bo * out_stride_b);

            for (int yo = yo_start; yo < yo_end; yo += yo_step)
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
                for (int xo = xo_start; xo < xo_end; xo += xo_step)
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

                    const auto s00 = wrapper::vdup_n(s00_s, FloatTagType{});
                    const auto s01 = wrapper::vdup_n(s01_s, FloatTagType{});
                    const auto s10 = wrapper::vdup_n(s10_s, FloatTagType{});
                    const auto s11 = wrapper::vdup_n(s11_s, FloatTagType{});

                    const int xi0 = utility::clamp<int>(xi, 0, input_width - 1);
                    const int xi1 = utility::clamp<int>(xi + 1, 0, input_width - 1);

                    const auto in_ptr_xi0_yi0 = in_ptr_yi0 + xi0 * in_stride_x;
                    const auto in_ptr_xi1_yi0 = in_ptr_yi0 + xi1 * in_stride_x;
                    const auto in_ptr_xi0_yi1 = in_ptr_yi1 + xi0 * in_stride_x;
                    const auto in_ptr_xi1_yi1 = in_ptr_yi1 + xi1 * in_stride_x;

                    int8_t *out_ptr_xo_yo = out_ptr_yo + xo * out_stride_x;

                    int cout = 0;
                    for (; cout <= (out_dim_ch - step_cout); cout += step_cout)
                    {
                        const auto in00 = wrapper::vloadq(in_ptr_xi0_yi0 + cout * sizeof(int8_t));
                        const auto in01 = wrapper::vloadq(in_ptr_xi1_yi0 + cout * sizeof(int8_t));
                        const auto in10 = wrapper::vloadq(in_ptr_xi0_yi1 + cout * sizeof(int8_t));
                        const auto in11 = wrapper::vloadq(in_ptr_xi1_yi1 + cout * sizeof(int8_t));

                        const int16x8_t in00_low  = wrapper::vmovl(wrapper::vgetlow(in00));
                        const int16x8_t in00_high = wrapper::vmovl(wrapper::vgethigh(in00));

                        const auto in00_0 = wrapper::vmul(
                            wrapper::vcvt<float>(wrapper::vsub(wrapper::vmovl(wrapper::vgetlow(in00_low)), voffset_in)),
                            vscale_in);
                        const auto in00_1 = wrapper::vmul(wrapper::vcvt<float>(wrapper::vsub(
                                                              wrapper::vmovl(wrapper::vgethigh(in00_low)), voffset_in)),
                                                          vscale_in);
                        const auto in00_2 = wrapper::vmul(wrapper::vcvt<float>(wrapper::vsub(
                                                              wrapper::vmovl(wrapper::vgetlow(in00_high)), voffset_in)),
                                                          vscale_in);
                        const auto in00_3 =
                            wrapper::vmul(wrapper::vcvt<float>(
                                              wrapper::vsub(wrapper::vmovl(wrapper::vgethigh(in00_high)), voffset_in)),
                                          vscale_in);

                        const int16x8_t in01_low  = wrapper::vmovl(wrapper::vgetlow(in01));
                        const int16x8_t in01_high = wrapper::vmovl(wrapper::vgethigh(in01));

                        const auto in01_0 = wrapper::vmul(
                            wrapper::vcvt<float>(wrapper::vsub(wrapper::vmovl(wrapper::vgetlow(in01_low)), voffset_in)),
                            vscale_in);
                        const auto in01_1 = wrapper::vmul(wrapper::vcvt<float>(wrapper::vsub(
                                                              wrapper::vmovl(wrapper::vgethigh(in01_low)), voffset_in)),
                                                          vscale_in);
                        const auto in01_2 = wrapper::vmul(wrapper::vcvt<float>(wrapper::vsub(
                                                              wrapper::vmovl(wrapper::vgetlow(in01_high)), voffset_in)),
                                                          vscale_in);
                        const auto in01_3 =
                            wrapper::vmul(wrapper::vcvt<float>(
                                              wrapper::vsub(wrapper::vmovl(wrapper::vgethigh(in01_high)), voffset_in)),
                                          vscale_in);

                        const int16x8_t in10_low  = wrapper::vmovl(wrapper::vgetlow(in10));
                        const int16x8_t in10_high = wrapper::vmovl(wrapper::vgethigh(in10));

                        const auto in10_0 = wrapper::vmul(
                            wrapper::vcvt<float>(wrapper::vsub(wrapper::vmovl(wrapper::vgetlow(in10_low)), voffset_in)),
                            vscale_in);
                        const auto in10_1 = wrapper::vmul(wrapper::vcvt<float>(wrapper::vsub(
                                                              wrapper::vmovl(wrapper::vgethigh(in10_low)), voffset_in)),
                                                          vscale_in);
                        const auto in10_2 = wrapper::vmul(wrapper::vcvt<float>(wrapper::vsub(
                                                              wrapper::vmovl(wrapper::vgetlow(in10_high)), voffset_in)),
                                                          vscale_in);
                        const auto in10_3 =
                            wrapper::vmul(wrapper::vcvt<float>(
                                              wrapper::vsub(wrapper::vmovl(wrapper::vgethigh(in10_high)), voffset_in)),
                                          vscale_in);

                        const int16x8_t in11_low  = wrapper::vmovl(wrapper::vgetlow(in11));
                        const int16x8_t in11_high = wrapper::vmovl(wrapper::vgethigh(in11));

                        const auto in11_0 = wrapper::vmul(
                            wrapper::vcvt<float>(wrapper::vsub(wrapper::vmovl(wrapper::vgetlow(in11_low)), voffset_in)),
                            vscale_in);
                        const auto in11_1 = wrapper::vmul(wrapper::vcvt<float>(wrapper::vsub(
                                                              wrapper::vmovl(wrapper::vgethigh(in11_low)), voffset_in)),
                                                          vscale_in);
                        const auto in11_2 = wrapper::vmul(wrapper::vcvt<float>(wrapper::vsub(
                                                              wrapper::vmovl(wrapper::vgetlow(in11_high)), voffset_in)),
                                                          vscale_in);
                        const auto in11_3 =
                            wrapper::vmul(wrapper::vcvt<float>(
                                              wrapper::vsub(wrapper::vmovl(wrapper::vgethigh(in11_high)), voffset_in)),
                                          vscale_in);

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
                        const auto out_0_int = wrapper::vcvta<int32_t>(wrapper::vmla(voffset_o, out_0, invvscale_o));
                        const auto out_1_int = wrapper::vcvta<int32_t>(wrapper::vmla(voffset_o, out_1, invvscale_o));
                        const auto out_2_int = wrapper::vcvta<int32_t>(wrapper::vmla(voffset_o, out_2, invvscale_o));
                        const auto out_3_int = wrapper::vcvta<int32_t>(wrapper::vmla(voffset_o, out_3, invvscale_o));
#else  // defined(__aarch64__) && !defined(BARE_METAL)
                        const auto out_0_int = wrapper::vcvt<int32_t>(wrapper::vmla(voffset_o, out_0, invvscale_o));
                        const auto out_1_int = wrapper::vcvt<int32_t>(wrapper::vmla(voffset_o, out_1, invvscale_o));
                        const auto out_2_int = wrapper::vcvt<int32_t>(wrapper::vmla(voffset_o, out_2, invvscale_o));
                        const auto out_3_int = wrapper::vcvt<int32_t>(wrapper::vmla(voffset_o, out_3, invvscale_o));
#endif // defined(__aarch64__) && !defined(BARE_METAL)
                        const auto low_part =
                            wrapper::vqmovn(wrapper::vcombine(wrapper::vqmovn(out_0_int), wrapper::vqmovn(out_1_int)));
                        const auto high_part =
                            wrapper::vqmovn(wrapper::vcombine(wrapper::vqmovn(out_2_int), wrapper::vqmovn(out_3_int)));
                        const auto out = wrapper::vcombine(low_part, high_part);

                        wrapper::vstore(out_ptr_xo_yo + cout * sizeof(int8_t), out);
                    }

                    for (; cout < out_dim_ch; ++cout)
                    {
                        const int8_t in00 = *(in_ptr_xi0_yi0 + cout * sizeof(int8_t));
                        const int8_t in01 = *(in_ptr_xi1_yi0 + cout * sizeof(int8_t));
                        const int8_t in10 = *(in_ptr_xi0_yi1 + cout * sizeof(int8_t));
                        const int8_t in11 = *(in_ptr_xi1_yi1 + cout * sizeof(int8_t));

                        const float in00_f = (static_cast<int32_t>(in00) - iq_info.offset) * iq_info.scale;
                        const float in01_f = (static_cast<int32_t>(in01) - iq_info.offset) * iq_info.scale;
                        const float in10_f = (static_cast<int32_t>(in10) - iq_info.offset) * iq_info.scale;
                        const float in11_f = (static_cast<int32_t>(in11) - iq_info.offset) * iq_info.scale;

                        float out = in00_f * s00_s;
                        out += in01_f * s01_s;
                        out += in10_f * s10_s;
                        out += in11_f * s11_s;

                        // Rounding modes of vector and scalar loops should match
#if defined(__aarch64__) && !defined(BARE_METAL)
                        *(out_ptr_xo_yo + cout * sizeof(int8_t)) = quantize_qasymm8_signed(out, oq_info);
#else  // defined(__aarch64__) && !defined(BARE_METAL)
                        *(out_ptr_xo_yo + cout * sizeof(int8_t)) =
                            quantize_qasymm8_signed(out, oq_info, RoundingPolicy::TO_ZERO);
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
} // namespace
namespace cpu
{
void qasymm8_signed_neon_scale(const ITensor      *src,
                               ITensor            *dst,
                               const ITensor      *offsets,
                               const ITensor      *dx,
                               const ITensor      *dy,
                               InterpolationPolicy policy,
                               BorderMode          border_mode,
                               PixelValue          constant_border_value,
                               float               sampling_offset,
                               bool                align_corners,
                               const Window       &window)
{
    if (policy == InterpolationPolicy::BILINEAR)
    {
        if (src->info()->quantization_info() == dst->info()->quantization_info() &&
            border_mode == BorderMode::REPLICATE)
        {
            s8_neon_scale(src, dst, offsets, dx, dy, policy, border_mode, constant_border_value, sampling_offset,
                          align_corners, window);
        }
        else
        {
            qasymm8_signed_neon_scale_bilinear(src, dst, offsets, dx, dy, border_mode, constant_border_value,
                                               sampling_offset, align_corners, window);
        }
    }
    else if (policy == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        nearest_neon_scale<int8_t>(src, dst, offsets, sampling_offset, align_corners, window);
    }
}
} // namespace cpu
} // namespace arm_compute
