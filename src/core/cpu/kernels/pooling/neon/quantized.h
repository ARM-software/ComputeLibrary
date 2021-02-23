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
#ifndef SRC_CORE_NEON_KERNELS_QUANTIZED_H
#define SRC_CORE_NEON_KERNELS_QUANTIZED_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
template <typename T>
inline typename std::enable_if<std::is_same<T, int8_t>::value, int8_t>::type
quantize(float val, const UniformQuantizationInfo &info)
{
    return quantize_qasymm8_signed(val, info);
}

template <typename T>
inline typename std::enable_if<std::is_same<T, uint8_t>::value, uint8_t>::type
quantize(float val, const UniformQuantizationInfo &info)
{
    return quantize_qasymm8(val, info);
}

template <typename T>
inline T vcvtq_q32_f32(float32x4_t values);

template <>
inline uint32x4_t vcvtq_q32_f32(float32x4_t values)
{
    return vcvtq_u32_f32(values);
}

template <>
inline int32x4_t vcvtq_q32_f32(float32x4_t values)
{
    return vcvtq_s32_f32(values);
}

template <typename T>
inline float32x4_t vcvtq_f32_q32(T values);

template <>
inline float32x4_t vcvtq_f32_q32(uint32x4_t values)
{
    return vcvtq_f32_u32(values);
}

template <>
inline float32x4_t vcvtq_f32_q32(int32x4_t values)
{
    return vcvtq_f32_s32(values);
}

template <typename Tout>
inline Tout vrequantize_pooling_with_scale(const float32x4x4_t &acc, const float quant_rescale, const float scale_pooling, const int32_t new_offset);

template <>
inline uint8x16_t vrequantize_pooling_with_scale(const float32x4x4_t &acc, const float quant_rescale, const float scale_pooling, const int32_t new_offset)
{
    const float new_scale = quant_rescale / scale_pooling;
    return vquantize(acc, UniformQuantizationInfo(new_scale, new_offset));
}

template <>
inline int8x16_t vrequantize_pooling_with_scale(const float32x4x4_t &acc, const float quant_rescale, const float scale_pooling, const int32_t new_offset)
{
    const float new_scale = quant_rescale / scale_pooling;
    return vquantize_signed(acc, UniformQuantizationInfo(new_scale, new_offset));
}

template <typename Tin, typename Tout>
inline Tout vrequantize_pooling(Tin vec1, Tin vec2, const UniformQuantizationInfo &requant_qinfo);

template <>
inline uint8x16_t vrequantize_pooling(uint8x8_t vec1, uint8x8_t vec2, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x4_t acc =
    {
        {
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8((vec1))))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8((vec1))))),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8((vec2))))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8((vec2))))),
        }
    };
    return vquantize(acc, requant_qinfo);
}

template <>
inline int8x16_t vrequantize_pooling(int8x8_t vec1, int8x8_t vec2, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x4_t acc =
    {
        {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8((vec1))))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8((vec1))))),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8((vec2))))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8((vec2))))),
        }
    };
    return vquantize_signed(acc, requant_qinfo);
}

template <typename T>
inline T vrequantize_pooling(T &vec, const UniformQuantizationInfo &requant_qinfo);

template <>
inline uint8x8_t vrequantize_pooling(uint8x8_t &vec, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x2_t acc =
    {
        {
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8((vec))))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8((vec))))),
        }
    };
    return vquantize(acc, requant_qinfo);
}

template <>
inline int8x8_t vrequantize_pooling(int8x8_t &vec, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x2_t acc =
    {
        {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8((vec))))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8((vec))))),
        }
    };
    return vquantize_signed(acc, requant_qinfo);
}

inline float calculate_avg_scale(bool exclude_padding, DataLayout data_layout, const Coordinates &id, const int pool_size_x, const int pool_size_y, const int upper_bound_w, const int upper_bound_h,
                                 const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    const unsigned int idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    int start_x = id[idx_width] * stride_x - pad_x;
    int start_y = id[idx_height] * stride_y - pad_y;

    const int end_x = std::min(start_x + pool_size_x, upper_bound_w);
    const int end_y = std::min(start_y + pool_size_y, upper_bound_h);
    if(exclude_padding)
    {
        start_x = std::max(0, start_x);
        start_y = std::max(0, start_y);
    }
    return 1.f / ((end_y - start_y) * (end_x - start_x));
}

template <typename T>
void poolingMxN_q8_neon_nhwc(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    ARM_COMPUTE_UNUSED(dst1);

    const int window_start_x     = window.x().start();
    const int window_end_x       = window.x().end();
    const int window_step_x      = 16;
    const int window_half_step_x = window_step_x / 2;

    Window window_out = window;
    window_out.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, window_src);
    Iterator out(dst0, window_out);

    using q8x8_t  = typename wrapper::traits::neon_vector<T, 8>::type;
    using q8x16_t = typename wrapper::traits::neon_vector<T, 16>::type;
    using q16_t   = typename wrapper::traits::promote_t<T>;
    using q16x8_t = typename wrapper::traits::neon_vector<q16_t, 8>::type;
    using q32_t   = typename wrapper::traits::promote_t<q16_t>;
    using q32x4_t = typename wrapper::traits::neon_vector<q32_t, 4>::type;

    const int pool_size_x     = pool_info.is_global_pooling ? src->info()->tensor_shape().y() : pool_info.pool_size.width;
    const int pool_size_y     = pool_info.is_global_pooling ? src->info()->tensor_shape().z() : pool_info.pool_size.height;
    const int pool_pad_right  = pool_info.pad_stride_info.pad_right();
    const int pool_pad_top    = pool_info.pad_stride_info.pad_top();
    const int pool_pad_left   = pool_info.pad_stride_info.pad_left();
    const int pool_pad_bottom = pool_info.pad_stride_info.pad_bottom();

    int pool_stride_x = 0;
    int pool_stride_y = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int upper_bound_w = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = src->info()->dimension(2) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);

    const float32x4_t             half_scale_v = vdupq_n_f32(0.5f);
    const UniformQuantizationInfo src_qinfo    = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo dst_qinfo    = dst0->info()->quantization_info().uniform();

    const float quant_rescale = dst_qinfo.scale / src_qinfo.scale;
    // "new_offset" doesn't have to consider the "half_scale_v" in its computation
    // With a requantization performed in a single step there won't be uncertainties introduced
    const int32_t new_offset = dst_qinfo.offset - static_cast<int32_t>(static_cast<float>(src_qinfo.offset) / quant_rescale);

    const float                   requant_scale  = dst_qinfo.scale / src_qinfo.scale;
    const int32_t                 requant_offset = dst_qinfo.offset - static_cast<int32_t>(static_cast<float>(src_qinfo.offset) / requant_scale);
    const UniformQuantizationInfo requant_qinfo  = UniformQuantizationInfo(requant_scale, requant_offset);

    execute_window_loop(window_out, [&](const Coordinates & id)
    {
        const int idx_width    = id.y() * pool_stride_x;
        const int idx_height   = id.z() * pool_stride_y;
        const int pool_limit_y = pool_pad_top - idx_height;
        const int pool_limit_x = pool_pad_left - idx_width;

        const int pool_start_y = std::max(0, window_src.z().start() + pool_limit_y);
        const int pool_end_y   = std::min(pool_size_y, window_src.z().end() + pool_limit_y);
        const int pool_start_x = std::max(0, window_src.y().start() + pool_limit_x);
        const int pool_end_x   = std::min(pool_size_x, window_src.y().end() + pool_limit_x);

        int x_off = window_start_x;
        for(; x_off <= (window_end_x - window_step_x); x_off += window_step_x)
        {
            if(pool_info.pool_type != PoolingType::MAX)
            {
                q32x4_t vres1 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
                q32x4_t vres2 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
                q32x4_t vres3 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
                q32x4_t vres4 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});

                // Calculate scale
                const float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NHWC, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                        pool_stride_y);

                // Perform pooling
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const q8x16_t data = wrapper::vloadq(reinterpret_cast<const T *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                         (src->info()->strides_in_bytes().z())) + x_off);

                        const q16x8_t data_q16  = wrapper::vmovl(wrapper::vgetlow(data));
                        const q16x8_t data2_q16 = wrapper::vmovl(wrapper::vgethigh(data));
                        vres1                   = wrapper::vadd(vres1, wrapper::vmovl(wrapper::vgetlow(data_q16)));
                        vres2                   = wrapper::vadd(vres2, wrapper::vmovl(wrapper::vgethigh(data_q16)));
                        vres3                   = wrapper::vadd(vres3, wrapper::vmovl(wrapper::vgetlow(data2_q16)));
                        vres4                   = wrapper::vadd(vres4, wrapper::vmovl(wrapper::vgethigh(data2_q16)));
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
            else
            {
                q8x16_t vres = wrapper::vdup_n(std::numeric_limits<T>::min(), wrapper::traits::vector_128_tag{});

                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const q8x16_t data = wrapper::vloadq(reinterpret_cast<const T *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                         (src->info()->strides_in_bytes().z())) + x_off);
                        vres               = wrapper::vmax(vres, data);
                    }
                }

                // Store result
                wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off, (src_qinfo != dst_qinfo) ? vrequantize_pooling<q8x8_t, q8x16_t>(wrapper::vgetlow(vres), wrapper::vgethigh(vres),
                                requant_qinfo) :
                                vres);
            }
        }

        if(pool_info.pool_type == PoolingType::MAX)
        {
            for(; x_off <= (window_end_x - window_half_step_x); x_off += window_half_step_x)
            {
                q8x8_t vres = wrapper::vdup_n(std::numeric_limits<T>::min(), wrapper::traits::vector_64_tag{});
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const q8x8_t data = wrapper::vload(reinterpret_cast<const T *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                       (src->info()->strides_in_bytes().z())) + x_off);
                        vres              = wrapper::vmax(vres, data);
                    }
                }

                // Store result
                wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off,
                                (src_qinfo != dst_qinfo) ? vrequantize_pooling<q8x8_t>(vres, requant_qinfo) : vres);
            }
        }

        // Left-overs loop
        for(; x_off < window_end_x; ++x_off)
        {
            if(pool_info.pool_type != PoolingType::MAX)
            {
                q32_t res = static_cast<q32_t>(0.f);

                // Calculate scale
                const float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NHWC, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                        pool_stride_y);

                // Perform pooling
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const T data = *(reinterpret_cast<const T *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                     (src->info()->strides_in_bytes().z())) + x_off);
                        res += data;
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
            else
            {
                T res = std::numeric_limits<T>::min();

                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const T data = *(reinterpret_cast<const T *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                     (src->info()->strides_in_bytes().z())) + x_off);
                        res          = std::max(res, data);
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
        }

    },
    in, out);
}

#if defined(ENABLE_NCHW_KERNELS)
template <typename T, typename TVec>
inline void scale_vector_q16x8(bool exclude_padding, TVec &v, const Coordinates &id, int id_offset, int step,
                               const int pool_size, const int upper_bound_w, const int upper_bound_h,
                               const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int       start_x = (id.x() + id_offset) * stride_x - pad_x;
    int       start_y = id.y() * stride_y - pad_y;
    const int end_y   = std::min(start_y + pool_size, upper_bound_h);
    if(exclude_padding)
    {
        start_y = std::max(0, start_y);
    }

    std::array<T, 8> elems =
    {
        {
            wrapper::vgetlane(v, 0),
            wrapper::vgetlane(v, 1),
            wrapper::vgetlane(v, 2),
            wrapper::vgetlane(v, 3),
            wrapper::vgetlane(v, 4),
            wrapper::vgetlane(v, 5),
            wrapper::vgetlane(v, 6),
            wrapper::vgetlane(v, 7),
        }
    };

    for(auto &el : elems)
    {
        int       c_start_x = start_x;
        const int end_x     = std::min(c_start_x + pool_size, upper_bound_w);
        if(exclude_padding)
        {
            c_start_x = std::max(0, c_start_x);
        }
        float scale = 1.f / ((end_y - start_y) * (end_x - c_start_x));
        el *= scale;
        start_x += step * stride_x;
    }

    v = wrapper::vsetlane(elems[0], v, 0);
    v = wrapper::vsetlane(elems[1], v, 1);
    v = wrapper::vsetlane(elems[2], v, 2);
    v = wrapper::vsetlane(elems[3], v, 3);
    v = wrapper::vsetlane(elems[4], v, 4);
    v = wrapper::vsetlane(elems[5], v, 5);
    v = wrapper::vsetlane(elems[6], v, 6);
    v = wrapper::vsetlane(elems[7], v, 7);
}

template <typename T>
void pooling2_quantized_neon_nchw(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    Iterator in(src, window_src);
    Iterator out(dst0, window);

    /** Neon vector types */
    using q8x8_t    = typename wrapper::traits::neon_vector<T, 8>::type;
    using q8x16_t   = typename wrapper::traits::neon_vector<T, 16>::type;
    using q8x8x2_t  = typename std::conditional<std::is_same<T, uint8_t>::value, uint8x8x2_t, int8x8x2_t>::type;
    using q16_t     = typename wrapper::traits::promote_t<T>;
    using q16x4_t   = typename wrapper::traits::neon_vector<q16_t, 4>::type;
    using q16x8_t   = typename wrapper::traits::neon_vector<q16_t, 8>::type;
    using q16x8x2_t = typename wrapper::traits::neon_vector<q16_t, 16>::type;

    constexpr int pool_size       = 2;
    int           pool_stride_x   = 0;
    int           pool_stride_y   = 0;
    const int     pool_pad_right  = pool_info.pad_stride_info.pad_right();
    const int     pool_pad_top    = pool_info.pad_stride_info.pad_top();
    const int     pool_pad_left   = pool_info.pad_stride_info.pad_left();
    const int     pool_pad_bottom = pool_info.pad_stride_info.pad_bottom();
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int upper_bound_w = src->info()->dimension(0) + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);

    const T *const src_top_ptr    = reinterpret_cast<const T *>(src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top))));
    const T *const src_bottom_ptr = reinterpret_cast<const T *>(src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1)));

    const int scale_step_x = (pool_stride_x == 1) ? 2 : 1;

    const UniformQuantizationInfo src_qinfo            = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo dst_qinfo            = dst0->info()->quantization_info().uniform();
    const bool                    have_different_qinfo = src_qinfo != dst_qinfo;

    const float                   requant_scale  = dst_qinfo.scale / src_qinfo.scale;
    const int32_t                 requant_offset = dst_qinfo.offset - static_cast<int32_t>(static_cast<float>(src_qinfo.offset) / requant_scale);
    const UniformQuantizationInfo requant_qinfo  = UniformQuantizationInfo(requant_scale, requant_offset);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto top_data    = wrapper::vloadq(src_top_ptr + in.offset());
        const auto bottom_data = wrapper::vloadq(src_bottom_ptr + in.offset());
        q8x8_t     lower_res   = {};
        q8x8_t     upper_res   = {};

        if(pool_info.pool_type != PoolingType::MAX)
        {
            const q16x8x2_t top_data_q16    = { { wrapper::vmovl(wrapper::vgetlow(top_data)), wrapper::vmovl(wrapper::vgethigh(top_data)) } };
            const q16x8x2_t bottom_data_q16 = { { wrapper::vmovl(wrapper::vgetlow(bottom_data)), wrapper::vmovl(wrapper::vgethigh(bottom_data)) } };

            // Add rows
            const q16x8x2_t vrsum =
            {
                {
                    wrapper::vadd(top_data_q16.val[0], bottom_data_q16.val[0]),
                    wrapper::vadd(top_data_q16.val[1], bottom_data_q16.val[1]),
                }
            };

            // Pair-wise add row data
            const q16x4_t vpsum_1 = wrapper::vpadd(wrapper::vgetlow(vrsum.val[0]), wrapper::vgethigh(vrsum.val[0]));
            const q16x4_t vpsum_2 = wrapper::vpadd(wrapper::vgetlow(vrsum.val[1]), wrapper::vgethigh(vrsum.val[1]));

            q16x8_t res_lower = wrapper::vcombine(vpsum_1, vpsum_2);

            // Scale lower result
            scale_vector_q16x8<q16_t, q16x8_t>(pool_info.exclude_padding, res_lower, id, 0, scale_step_x,
                                               pool_size, upper_bound_w, upper_bound_h,
                                               pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
            lower_res = wrapper::vmovn(res_lower);

            // Compute upper result for stride_x == 1
            if(pool_stride_x == 1)
            {
                // Shifted row sum
                const q16x8x2_t vrsum_shifted =
                {
                    {
                        wrapper::vext_1(vrsum.val[0], vrsum.val[1]),
                        wrapper::vext_1(vrsum.val[1], vrsum.val[1])
                    }
                };

                // Pair-wise add shifted row
                q16x8_t res_upper = wrapper::vcombine(
                                        wrapper::vpadd(wrapper::vgetlow(vrsum_shifted.val[0]), wrapper::vgethigh(vrsum_shifted.val[0])),
                                        wrapper::vpadd(wrapper::vgetlow(vrsum_shifted.val[1]), wrapper::vgethigh(vrsum_shifted.val[1])));

                // Scale upper result
                scale_vector_q16x8<q16_t, q16x8_t>(pool_info.exclude_padding, res_upper, id, 1, 2,
                                                   pool_size, upper_bound_w, upper_bound_h,
                                                   pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
                upper_res = wrapper::vmovn(res_upper);
            }
        }
        else
        {
            const q8x16_t max_data = wrapper::vmax(top_data, bottom_data);
            lower_res              = wrapper::vpmax(wrapper::vgetlow(max_data), wrapper::vgethigh(max_data));
            if(pool_stride_x == 1)
            {
                const q8x16_t max_data_shifted = wrapper::vext_1(max_data, max_data);
                upper_res                      = wrapper::vpmax(wrapper::vgetlow(max_data_shifted), wrapper::vgethigh(max_data_shifted));
            }
        }

        if(have_different_qinfo)
        {
            const auto requantized_dst = vrequantize_pooling<q8x8_t, q8x16_t>(lower_res, upper_res, requant_qinfo);
            lower_res                  = wrapper::vgetlow(requantized_dst);
            upper_res                  = wrapper::vgethigh(requantized_dst);
        }

        // Store result
        if(pool_stride_x == 1)
        {
            const q8x8x2_t res = { { lower_res, upper_res } };
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()), res);
        }
        else
        {
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()), lower_res);
        }
    },
    in, out);
}

template <typename T>
void pooling3_quantized_neon_nchw(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    Iterator in(src, window_src);
    Iterator out(dst0, window);

    /** Neon vector types */
    using q8x8_t    = typename wrapper::traits::neon_vector<T, 8>::type;
    using q8x16_t   = typename wrapper::traits::neon_vector<T, 16>::type;
    using q8x8x2_t  = typename std::conditional<std::is_same<T, uint8_t>::value, uint8x8x2_t, int8x8x2_t>::type;
    using q16_t     = typename wrapper::traits::promote_t<T>;
    using q16x8_t   = typename wrapper::traits::neon_vector<q16_t, 8>::type;
    using q16x8x2_t = typename wrapper::traits::neon_vector<q16_t, 16>::type;

    constexpr int pool_size       = 3;
    const int     pool_pad_right  = pool_info.pad_stride_info.pad_right();
    const int     pool_pad_top    = pool_info.pad_stride_info.pad_top();
    const int     pool_pad_left   = pool_info.pad_stride_info.pad_left();
    const int     pool_pad_bottom = pool_info.pad_stride_info.pad_bottom();
    int           pool_stride_x   = 0;
    int           pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int upper_bound_w = src->info()->dimension(0) + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);

    const UniformQuantizationInfo &src_qinfo = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo &dst_qinfo = dst0->info()->quantization_info().uniform();

    const float                   requant_scale  = dst_qinfo.scale / src_qinfo.scale;
    const int32_t                 requant_offset = dst_qinfo.offset - static_cast<int32_t>(static_cast<float>(src_qinfo.offset) / requant_scale);
    const UniformQuantizationInfo requant_qinfo  = UniformQuantizationInfo(requant_scale, requant_offset);

    const T *const src_top_ptr    = reinterpret_cast<const T *>(src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top))));
    const T *const src_middle_ptr = reinterpret_cast<const T *>(src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1)));
    const T *const src_bottom_ptr = reinterpret_cast<const T *>(src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 2)));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto top_data    = wrapper::vloadq(src_top_ptr + in.offset());
        const auto middle_data = wrapper::vloadq(src_middle_ptr + in.offset());
        const auto bottom_data = wrapper::vloadq(src_bottom_ptr + in.offset());
        q8x8_t     fres        = {};
        q8x16_t    fqres       = {};

        if(pool_info.pool_type == PoolingType::AVG)
        {
            // Convert data to u16
            const q16x8x2_t top_data_q16    = { { wrapper::vmovl(wrapper::vgetlow(top_data)), wrapper::vmovl(wrapper::vgethigh(top_data)) } };
            const q16x8x2_t middle_data_q16 = { { wrapper::vmovl(wrapper::vgetlow(middle_data)), wrapper::vmovl(wrapper::vgethigh(middle_data)) } };
            const q16x8x2_t bottom_data_q16 = { { wrapper::vmovl(wrapper::vgetlow(bottom_data)), wrapper::vmovl(wrapper::vgethigh(bottom_data)) } };

            // Calculate row sums
            const q16x8x2_t vrsum =
            {
                {
                    wrapper::vadd(wrapper::vadd(top_data_q16.val[0], bottom_data_q16.val[0]), middle_data_q16.val[0]),
                    wrapper::vadd(wrapper::vadd(top_data_q16.val[1], bottom_data_q16.val[1]), middle_data_q16.val[1]),
                }
            };
            const q16x8x2_t vrsum_shifted_1 =
            {
                {
                    wrapper::vext_1(vrsum.val[0], vrsum.val[1]),
                    wrapper::vext_1(vrsum.val[1], vrsum.val[1])
                }
            };
            const q16x8x2_t vrsum_shifted_2 =
            {
                {
                    wrapper::vext_2(vrsum.val[0], vrsum.val[1]),
                    wrapper::vext_2(vrsum.val[1], vrsum.val[1])
                }
            };
            // Calculate final sum
            q16x8x2_t final_sum =
            {
                {
                    wrapper::vadd(wrapper::vadd(vrsum.val[0], vrsum_shifted_1.val[0]), vrsum_shifted_2.val[0]),
                    wrapper::vadd(wrapper::vadd(vrsum.val[1], vrsum_shifted_1.val[1]), vrsum_shifted_2.val[1]),
                }
            };
            if(pool_stride_x == 2)
            {
                q16x8_t res =
                {
                    wrapper::vgetlane(final_sum.val[0], 0),
                    wrapper::vgetlane(final_sum.val[0], 2),
                    wrapper::vgetlane(final_sum.val[0], 4),
                    wrapper::vgetlane(final_sum.val[0], 6),
                    wrapper::vgetlane(final_sum.val[1], 0),
                    wrapper::vgetlane(final_sum.val[1], 2),
                    wrapper::vgetlane(final_sum.val[1], 4),
                    wrapper::vgetlane(final_sum.val[1], 6),
                };

                scale_vector_q16x8<q16_t, q16x8_t>(pool_info.exclude_padding, res, id, 0, 1,
                                                   pool_size, upper_bound_w, upper_bound_h,
                                                   pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
                fres = wrapper::vmovn(res);
            }
            else
            {
                // Scale lower result
                scale_vector_q16x8<q16_t, q16x8_t>(pool_info.exclude_padding, final_sum.val[0], id, 0, 1,
                                                   pool_size, upper_bound_w, upper_bound_h,
                                                   pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
                // Scale lower result
                scale_vector_q16x8<q16_t, q16x8_t>(pool_info.exclude_padding, final_sum.val[1], id, 8, 1,
                                                   pool_size, upper_bound_w, upper_bound_h,
                                                   pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
                fqres = wrapper::vcombine(wrapper::vmovn(final_sum.val[0]), wrapper::vmovn(final_sum.val[1]));
            }
        }
        else
        {
            const q8x16_t max_data        = wrapper::vmax(wrapper::vmax(top_data, bottom_data), middle_data);
            const q8x16_t max_data_shift1 = wrapper::vext_1(max_data, max_data);
            const q8x16_t max_data_shift2 = wrapper::vext_2(max_data, max_data);
            const q8x16_t final_max       = wrapper::vmax(wrapper::vmax(max_data, max_data_shift1), max_data_shift2);

            if(pool_stride_x == 2)
            {
                const q8x8x2_t      table      = { { wrapper::vgetlow(final_max), wrapper::vgethigh(final_max) } };
                static const q8x8_t lookup_val = { 0, 2, 4, 6, 8, 10, 12, 14 };
                fres                           = wrapper::vtbl(table, lookup_val);
            }
            else
            {
                fqres = final_max;
            }
        }

        // Store result
        if(pool_stride_x == 1)
        {
            if(src_qinfo != dst_qinfo)
            {
                fqres = vrequantize_pooling<q8x8_t, q8x16_t>(wrapper::vgetlow(fqres), wrapper::vgethigh(fqres), requant_qinfo);
            }
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()), fqres);
        }
        else
        {
            if(src_qinfo != dst_qinfo)
            {
                fres = vrequantize_pooling<q8x8_t>(fres, requant_qinfo);
            }
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()), fres);
        }
    },
    in, out);
}

template <typename T>
void poolingMxN_quantized_neon_nchw(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    Iterator in(src, window_src);
    Iterator out(dst0, window);

    /** Neon vector types */
    using q8x8_t  = typename wrapper::traits::neon_vector<T, 8>::type;
    using q16_t   = typename wrapper::traits::promote_t<T>;
    using q16x8_t = typename wrapper::traits::neon_vector<q16_t, 8>::type;
    using q32_t   = typename wrapper::traits::promote_t<q16_t>;
    using q32x4_t = typename wrapper::traits::neon_vector<q32_t, 4>::type;

    const int pool_size_x     = pool_info.is_global_pooling ? src->info()->tensor_shape().x() : pool_info.pool_size.width;
    const int pool_size_y     = pool_info.is_global_pooling ? src->info()->tensor_shape().y() : pool_info.pool_size.height;
    const int pool_pad_right  = pool_info.pad_stride_info.pad_right();
    const int pool_pad_top    = pool_info.pad_stride_info.pad_top();
    const int pool_pad_left   = pool_info.pad_stride_info.pad_left();
    const int pool_pad_bottom = pool_info.pad_stride_info.pad_bottom();
    int       pool_stride_x   = 0;
    int       pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int upper_bound_w = src->info()->dimension(0) + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);

    const UniformQuantizationInfo &src_qinfo = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo &dst_qinfo = dst0->info()->quantization_info().uniform();

    execute_window_loop(window, [&](const Coordinates & id)
    {
        T res = std::numeric_limits<T>::min();

        if(pool_info.pool_type != PoolingType::MAX)
        {
            q32x4_t vres = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
            q32_t   sres = 0;

            // Calculate scale
            const float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                    pool_stride_y);

            // Perform pooling
            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 8); x += 8)
                {
                    const q8x8_t data = wrapper::vload(reinterpret_cast<const T *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                                   (src->info()->strides_in_bytes().y())));

                    const q16x8_t data_q16 = wrapper::vmovl(data);
                    vres                   = wrapper::vadd(vres, wrapper::vaddl(wrapper::vgethigh(data_q16), wrapper::vgetlow(data_q16)));
                }

                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    T data = *(reinterpret_cast<const T *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                           (src->info()->strides_in_bytes().y())));
                    sres += data;
                }
            }

            // Reduction
            const auto tmp = wrapper::vpadd(wrapper::vgethigh(vres), wrapper::vgetlow(vres));
            sres += wrapper::vgetlane(tmp, 0) + wrapper::vgetlane(tmp, 1);

            // Divide by scale
            res = static_cast<T>(support::cpp11::round(sres * scale));
        }
        else
        {
            q8x8_t vres = wrapper::vdup_n(std::numeric_limits<T>::min(), wrapper::traits::vector_64_tag{});

            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 8); x += 8)
                {
                    const q8x8_t data = wrapper::vload(reinterpret_cast<const T *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                                   (src->info()->strides_in_bytes().y())));
                    vres              = wrapper::vmax(vres, data);
                }
                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    const T data = *(reinterpret_cast<const T *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                 (src->info()->strides_in_bytes().y())));
                    res          = std::max(res, data);
                }
            }

            // Reduce max
            vres = wrapper::vpmax(vres, vres);
            vres = wrapper::vpmax(vres, vres);
            vres = wrapper::vpmax(vres, vres);

            // Get max value
            res = std::max(res, wrapper::vgetlane(vres, 0));
        }
        // Store result
        res                                 = (src_qinfo != dst_qinfo) ? Qasymm8QuantizationHelper<T>::quantize(Qasymm8QuantizationHelper<T>::dequantize(res, src_qinfo), dst_qinfo) : res;
        *(reinterpret_cast<T *>(out.ptr())) = res;
    },
    in, out);
}
#endif /* defined(ENABLE_NCHW_KERNELS) */
} // namespace cpu
} // namespace arm_compute

#endif // SRC_CORE_NEON_KERNELS_QUANTIZED_H
