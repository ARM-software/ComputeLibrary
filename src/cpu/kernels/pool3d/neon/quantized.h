/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_POOL3D_QUANTIZED_H
#define SRC_CORE_NEON_KERNELS_POOL3D_QUANTIZED_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/PoolingHelpers.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
template <typename T>
void avg_poolingMxNxD_q8_neon_ndhwc(const ITensor *src, ITensor *dst0, Pooling3dLayerInfo &pool_info, const Window &window_out,
                                    const int window_step_x)

{
    using q8x8_t  = typename wrapper::traits::neon_vector<T, 8>::type;
    using q8x16_t = typename wrapper::traits::neon_vector<T, 16>::type;
    using q16_t   = typename wrapper::traits::promote_t<T>;
    using q16x8_t = typename wrapper::traits::neon_vector<q16_t, 8>::type;
    using q32_t   = typename wrapper::traits::promote_t<q16_t>;
    using q32x4_t = typename wrapper::traits::neon_vector<q32_t, 4>::type;

    int pool_stride_x = static_cast<int>(pool_info.stride.width);
    int pool_stride_y = static_cast<int>(pool_info.stride.height);
    int pool_stride_z = static_cast<int>(pool_info.stride.depth);

    const int pool_size_x = pool_info.is_global_pooling ? src->info()->tensor_shape().y() : pool_info.pool_size.width;
    const int pool_size_y = pool_info.is_global_pooling ? src->info()->tensor_shape().z() : pool_info.pool_size.height;
    const int pool_size_z = pool_info.is_global_pooling ? src->info()->tensor_shape()[3] : pool_info.pool_size.depth;

    const int pool_pad_top    = static_cast<int>(pool_info.padding.top);
    const int pool_pad_bottom = static_cast<int>(pool_info.padding.bottom);
    const int pool_pad_left   = static_cast<int>(pool_info.padding.left);
    const int pool_pad_right  = static_cast<int>(pool_info.padding.right);
    const int pool_pad_front  = static_cast<int>(pool_info.padding.front);
    const int pool_pad_back   = static_cast<int>(pool_info.padding.back);

    const int upper_bound_w = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = src->info()->dimension(2) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);
    const int upper_bound_d = src->info()->dimension(3) + (pool_info.exclude_padding ? 0 : pool_pad_back);

    const int input_dim_c = src->info()->dimension(0);
    const int input_dim_w = src->info()->dimension(1);
    const int input_dim_h = src->info()->dimension(2);
    const int input_dim_d = src->info()->dimension(3);

    const int y_stride = static_cast<int>(src->info()->strides_in_bytes().y());
    const int z_stride = static_cast<int>(src->info()->strides_in_bytes().z());
    const int w_stride = static_cast<int>(src->info()->strides_in_bytes()[3]);
    const int n_stride = static_cast<int>(src->info()->strides_in_bytes()[4]);

    const uint8_t *in_ptr_start = src->buffer() + src->info()->offset_first_element_in_bytes();

    const int window_end_x   = input_dim_c;
    const int window_start_x = 0;

    Iterator out(dst0, window_out);

    const float32x4_t             half_scale_v = vdupq_n_f32(0.5f);
    const UniformQuantizationInfo src_qinfo    = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo dst_qinfo    = dst0->info()->quantization_info().uniform();

    const float quant_rescale = dst_qinfo.scale / src_qinfo.scale;
    // "new_offset" doesn't have to consider the "half_scale_v" in its computation
    // With a requantization performed in a single step there won't be uncertainties introduced
    const int32_t new_offset = dst_qinfo.offset - static_cast<int32_t>(static_cast<float>(src_qinfo.offset) / quant_rescale);

    execute_window_loop(window_out, [&](const Coordinates & id)
    {
        // Computing the theoretical input starting/ending points
        const int in_idx_width  = static_cast<int>(id.y()) * pool_stride_x - pool_pad_left;
        const int in_idx_height = static_cast<int>(id.z()) * pool_stride_y - pool_pad_top;
        const int in_idx_depth  = static_cast<int>(id[3]) * pool_stride_z - pool_pad_front;

        const int pool_start_x = std::max(0, -in_idx_width);
        const int pool_end_x_t = std::min(input_dim_w + pool_pad_left - in_idx_width, pool_size_x);
        const int pool_start_y = std::max(0, -in_idx_height);
        const int pool_end_y_t = std::min(input_dim_h + pool_pad_top - in_idx_height, pool_size_y);

        const int pool_start_z = std::max(0, -in_idx_depth);
        const int pool_end_z_t = std::min(input_dim_d + pool_pad_front - in_idx_depth, pool_size_z);

        // The end of width to consider in calculation should exclude PAD_X, PAD_Y and PAD_Z
        const int pool_end_x = std::min(pool_end_x_t, input_dim_w - in_idx_width);
        const int pool_end_y = std::min(pool_end_y_t, input_dim_h - in_idx_height);
        const int pool_end_z = std::min(pool_end_z_t, input_dim_d - in_idx_depth);

        // Calculate scale
        const float scale = calculate_avg_scale_pool3d(pool_info.exclude_padding, id, pool_size_x, pool_size_y, pool_size_z, upper_bound_w, upper_bound_h, upper_bound_d, pool_pad_left,
                                                       pool_pad_top, pool_pad_front, pool_stride_x, pool_stride_y, pool_stride_z);

        const uint8_t *in_ptr_n = in_ptr_start + id[4] * n_stride;

        int x_off = window_start_x;

        for(; x_off <= (window_end_x - window_step_x); x_off += window_step_x) // C
        {
            q32x4_t vres1 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
            q32x4_t vres2 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
            q32x4_t vres3 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
            q32x4_t vres4 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});

            // Perform pooling
            for(int z = pool_start_z; z < pool_end_z; ++z)
            {
                const uint8_t *in_ptr_z = in_ptr_n + (z + in_idx_depth) * w_stride;
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    const uint8_t *in_ptr_y = in_ptr_z + (y + in_idx_height) * z_stride;
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const uint8_t *in_ptr_x = in_ptr_y + (x + in_idx_width) * y_stride;
                        const q8x16_t  data     = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr_x) + x_off);

                        const q16x8_t data_q16  = wrapper::vmovl(wrapper::vgetlow(data));
                        const q16x8_t data2_q16 = wrapper::vmovl(wrapper::vgethigh(data));
                        vres1                   = wrapper::vadd(vres1, wrapper::vmovl(wrapper::vgetlow(data_q16)));
                        vres2                   = wrapper::vadd(vres2, wrapper::vmovl(wrapper::vgethigh(data_q16)));
                        vres3                   = wrapper::vadd(vres3, wrapper::vmovl(wrapper::vgetlow(data2_q16)));
                        vres4                   = wrapper::vadd(vres4, wrapper::vmovl(wrapper::vgethigh(data2_q16)));
                    }
                }
            }

            if(src_qinfo != dst_qinfo)
            {
                const float32x4x4_t vres =
                {
                    {
                        vcvtq_f32_q32(vres1),
                        vcvtq_f32_q32(vres2),
                        vcvtq_f32_q32(vres3),
                        vcvtq_f32_q32(vres4),
                    }
                };
                const auto requantized_dst = vrequantize_pooling_with_scale<q8x16_t>(vres, quant_rescale, scale, new_offset);
                // Store result
                wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off, wrapper::vgetlow(requantized_dst));
                wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off + 8, wrapper::vgethigh(requantized_dst));
            }
            else
            {
                const float32x4_t scale_v = vdupq_n_f32(scale);
                // Divide by scale and add 0.5f to round to nearest instead of rounding towards zero
                vres1 = vcvtq_q32_f32<q32x4_t>(wrapper::vmla(half_scale_v, vcvtq_f32_q32(vres1), scale_v));
                vres2 = vcvtq_q32_f32<q32x4_t>(wrapper::vmla(half_scale_v, vcvtq_f32_q32(vres2), scale_v));
                vres3 = vcvtq_q32_f32<q32x4_t>(wrapper::vmla(half_scale_v, vcvtq_f32_q32(vres3), scale_v));
                vres4 = vcvtq_q32_f32<q32x4_t>(wrapper::vmla(half_scale_v, vcvtq_f32_q32(vres4), scale_v));

                const q8x8_t res1 = wrapper::vmovn(wrapper::vcombine(wrapper::vmovn(vres1), wrapper::vmovn(vres2)));
                const q8x8_t res2 = wrapper::vmovn(wrapper::vcombine(wrapper::vmovn(vres3), wrapper::vmovn(vres4)));
                // Store result
                wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off, res1);
                wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off + 8, res2);
            }
        }

        // Left-overs loop
        for(; x_off < window_end_x; ++x_off)
        {
            q32_t res = static_cast<q32_t>(0.f);

            // Perform pooling
            for(int z = pool_start_z; z < pool_end_z; ++z)
            {
                const uint8_t *in_ptr_z = in_ptr_n + (z + in_idx_depth) * w_stride;
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    const uint8_t *in_ptr_y = in_ptr_z + (y + in_idx_height) * z_stride;
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const uint8_t *in_ptr_x = in_ptr_y + (x + in_idx_width) * y_stride;
                        const T        data     = *(reinterpret_cast<const T *>(in_ptr_x) + x_off);
                        res += data;
                    }
                }
            }

            if(src_qinfo != dst_qinfo)
            {
                const float res_f           = static_cast<float>(res);
                const float new_scale       = quant_rescale / scale;
                const auto  requantized_dst = quantize<T>(res_f, UniformQuantizationInfo(new_scale, new_offset));

                // Store result
                *(reinterpret_cast<T *>(out.ptr()) + x_off) = requantized_dst;
            }
            else
            {
                // Divide by scale and add 0.5f to round to nearest instead of rounding towards zero
                res = static_cast<T>(0.5f + static_cast<float>(res) * scale);

                // Store result
                *(reinterpret_cast<T *>(out.ptr()) + x_off) = res;
            }
        }
    },
    out);
}

template <typename T>
void max_poolingMxNxD_q8_neon_ndhwc(const ITensor *src, ITensor *dst0, Pooling3dLayerInfo &pool_info, const Window &window_out,
                                    const int window_step_x)

{
    using q8x8_t  = typename wrapper::traits::neon_vector<T, 8>::type;
    using q8x16_t = typename wrapper::traits::neon_vector<T, 16>::type;

    const int window_half_step_x = window_step_x / 2;

    int pool_stride_x = static_cast<int>(pool_info.stride.width);
    int pool_stride_y = static_cast<int>(pool_info.stride.height);
    int pool_stride_z = static_cast<int>(pool_info.stride.depth);

    const int pool_size_x = pool_info.is_global_pooling ? src->info()->tensor_shape().y() : pool_info.pool_size.width;
    const int pool_size_y = pool_info.is_global_pooling ? src->info()->tensor_shape().z() : pool_info.pool_size.height;
    const int pool_size_z = pool_info.is_global_pooling ? src->info()->tensor_shape()[3] : pool_info.pool_size.depth;

    const int pool_pad_top   = static_cast<int>(pool_info.padding.top);
    const int pool_pad_left  = static_cast<int>(pool_info.padding.left);
    const int pool_pad_front = static_cast<int>(pool_info.padding.front);

    const int input_dim_c = src->info()->dimension(0);
    const int input_dim_w = src->info()->dimension(1);
    const int input_dim_h = src->info()->dimension(2);
    const int input_dim_d = src->info()->dimension(3);

    const int y_stride = static_cast<int>(src->info()->strides_in_bytes().y());
    const int z_stride = static_cast<int>(src->info()->strides_in_bytes().z());
    const int w_stride = static_cast<int>(src->info()->strides_in_bytes()[3]);
    const int n_stride = static_cast<int>(src->info()->strides_in_bytes()[4]);

    const uint8_t *in_ptr_start = src->buffer() + src->info()->offset_first_element_in_bytes();

    const int window_end_x   = input_dim_c;
    const int window_start_x = 0;

    Iterator out(dst0, window_out);

    const UniformQuantizationInfo src_qinfo = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo dst_qinfo = dst0->info()->quantization_info().uniform();

    const float                   requant_scale  = dst_qinfo.scale / src_qinfo.scale;
    const int32_t                 requant_offset = dst_qinfo.offset - static_cast<int32_t>(static_cast<float>(src_qinfo.offset) / requant_scale);
    const UniformQuantizationInfo requant_qinfo  = UniformQuantizationInfo(requant_scale, requant_offset);

    execute_window_loop(window_out, [&](const Coordinates & id)
    {
        // Computing the theoretical input starting/ending points
        const int in_idx_width  = static_cast<int>(id.y()) * pool_stride_x - pool_pad_left;
        const int in_idx_height = static_cast<int>(id.z()) * pool_stride_y - pool_pad_top;
        const int in_idx_depth  = static_cast<int>(id[3]) * pool_stride_z - pool_pad_front;

        const int pool_start_x = std::max(0, -in_idx_width);
        const int pool_end_x_t = std::min(input_dim_w + pool_pad_left - in_idx_width, pool_size_x);
        const int pool_start_y = std::max(0, -in_idx_height);
        const int pool_end_y_t = std::min(input_dim_h + pool_pad_top - in_idx_height, pool_size_y);

        const int pool_start_z = std::max(0, -in_idx_depth);
        const int pool_end_z_t = std::min(input_dim_d + pool_pad_front - in_idx_depth, pool_size_z);

        // The end of width to consider in calculation should exclude PAD_X, PAD_Y and PAD_Z
        const int pool_end_x = std::min(pool_end_x_t, input_dim_w - in_idx_width);
        const int pool_end_y = std::min(pool_end_y_t, input_dim_h - in_idx_height);
        const int pool_end_z = std::min(pool_end_z_t, input_dim_d - in_idx_depth);

        const uint8_t *in_ptr_n = in_ptr_start + id[4] * n_stride;

        int x_off = window_start_x;

        for(; x_off <= (window_end_x - window_step_x); x_off += window_step_x) // C
        {
            q8x16_t vres = wrapper::vdup_n(std::numeric_limits<T>::min(), wrapper::traits::vector_128_tag{});

            // Perform pooling
            for(int z = pool_start_z; z < pool_end_z; ++z)
            {
                const uint8_t *in_ptr_z = in_ptr_n + (z + in_idx_depth) * w_stride;
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    const uint8_t *in_ptr_y = in_ptr_z + (y + in_idx_height) * z_stride;
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const uint8_t *in_ptr_x = in_ptr_y + (x + in_idx_width) * y_stride;
                        const q8x16_t  data     = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr_x) + x_off);

                        vres = wrapper::vmax(vres, data);
                    }
                }
            }

            // Store result
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off, (src_qinfo != dst_qinfo) ? vrequantize_pooling<q8x8_t, q8x16_t>(wrapper::vgetlow(vres), wrapper::vgethigh(vres),
                            requant_qinfo) :
                            vres);
        }

        // Leftovers using half the window step
        for(; x_off <= (window_end_x - window_half_step_x); x_off += window_half_step_x)
        {
            q8x8_t vres = wrapper::vdup_n(std::numeric_limits<T>::min(), wrapper::traits::vector_64_tag{});

            // Perform pooling
            for(int z = pool_start_z; z < pool_end_z; ++z)
            {
                const uint8_t *in_ptr_z = in_ptr_n + (z + in_idx_depth) * w_stride;
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    const uint8_t *in_ptr_y = in_ptr_z + (y + in_idx_height) * z_stride;
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const uint8_t *in_ptr_x = in_ptr_y + (x + in_idx_width) * y_stride;
                        const q8x8_t   data     = wrapper::vload(reinterpret_cast<const T *>(in_ptr_x) + x_off);

                        vres = wrapper::vmax(vres, data);
                    }
                }
            }

            // Store result
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off,
                            (src_qinfo != dst_qinfo) ? vrequantize_pooling<q8x8_t>(vres, requant_qinfo) : vres);
        }

        // Left-overs loop
        for(; x_off < window_end_x; ++x_off)
        {
            T res = std::numeric_limits<T>::min();

            for(int z = pool_start_z; z < pool_end_z; ++z)
            {
                const uint8_t *in_ptr_z = in_ptr_n + (z + in_idx_depth) * w_stride;
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    const uint8_t *in_ptr_y = in_ptr_z + (y + in_idx_height) * z_stride;
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const uint8_t *in_ptr_x = in_ptr_y + (x + in_idx_width) * y_stride;
                        const T        data     = *(reinterpret_cast<const T *>(in_ptr_x) + x_off);

                        res = std::max(res, data);
                    }
                }
            }

            // Store result
            if(src_qinfo != dst_qinfo)
            {
                const float res_f                           = static_cast<float>(res);
                *(reinterpret_cast<T *>(out.ptr()) + x_off) = quantize<T>(res_f, requant_qinfo);
            }
            else
            {
                *(reinterpret_cast<T *>(out.ptr()) + x_off) = res;
            }
        }
    },
    out);
}

} // namespace cpu
} // namespace arm_compute

#endif // SRC_CORE_NEON_KERNELS_POOL3D_QUANTIZED_H