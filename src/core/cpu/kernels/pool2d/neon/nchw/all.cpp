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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include "src/core/cpu/kernels/pool2d/neon/list.h"
#include "src/core/helpers/WindowHelpers.h"

#ifdef ENABLE_NCHW_KERNELS
namespace arm_compute
{
namespace cpu
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void pooling3_fp16_neon_nchw(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    ARM_COMPUTE_UNUSED(pool_info.pool_type);
    ARM_COMPUTE_UNUSED(pool_info.exclude_padding);

    Iterator in(src, window_src);
    Iterator out(dst0, window);

    constexpr const int pool_size       = 3;
    const int           pool_pad_right  = pool_info.pad_stride_info.pad_right();
    const int           pool_pad_top    = pool_info.pad_stride_info.pad_top();
    const int           pool_pad_left   = pool_info.pad_stride_info.pad_left();
    const int           pool_pad_bottom = pool_info.pad_stride_info.pad_bottom();
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int upper_bound_w = src->info()->dimension(0) + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);

    const unsigned char *const src_top_ptr    = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
    const unsigned char *const src_middle_ptr = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));
    const unsigned char *const src_bottom_ptr = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 2));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float16x4_t top_data    = vld1_f16(reinterpret_cast<const float16_t *>(src_top_ptr + in.offset()));
        float16x4_t middle_data = vld1_f16(reinterpret_cast<const float16_t *>(src_middle_ptr + in.offset()));
        float16x4_t bottom_data = vld1_f16(reinterpret_cast<const float16_t *>(src_bottom_ptr + in.offset()));
        float16x4_t res         = {};

        // Get power of 2 in case of l2 pooling
        if(pool_info.pool_type == PoolingType::L2)
        {
            top_data    = vmul_f16(top_data, top_data);
            middle_data = vmul_f16(middle_data, middle_data);
            bottom_data = vmul_f16(bottom_data, bottom_data);
        }

        if(pool_info.pool_type != PoolingType::MAX)
        {
            // Calculate scale
            const float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size, pool_size, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                    pool_stride_y);
            const float16x4_t scale_v = vdup_n_f16(scale);
            // Perform pooling
            const float16x4_t sum_data = vadd_f16(vadd_f16(top_data, bottom_data), middle_data);
            res                        = vpadd_f16(vset_lane_f16(0.f, sum_data, 3), sum_data);
            res                        = vmul_f16(vpadd_f16(res, res), scale_v);
        }
        else
        {
            const float16x4_t max_data = vmax_f16(vmax_f16(top_data, bottom_data), middle_data);
            res                        = vpmax_f16(vset_lane_f16(-std::numeric_limits<float>::max(), max_data, 3), max_data);
            res                        = vpmax_f16(res, res);
        }

        // Calculate square-root in case of l2 pooling
        if(pool_info.pool_type == PoolingType::L2)
        {
            res = vinv_f16(vinvsqrt_f16(res));
        }

        *(reinterpret_cast<float16_t *>(out.ptr())) = vget_lane_f16(res, 0);
    },
    in, out);
}

template <typename T>
inline typename std::enable_if<std::is_same<T, float16_t>::value, float32x2_t>::type
f16_to_f32(float16x4_t in)
{
    float32x2_t out = { static_cast<float>(vget_lane_f16(in, 0)), static_cast<float>(vget_lane_f16(in, 1)) };
    return out;
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template <typename T>
inline typename std::enable_if<std::is_same<T, float>::value, float32x2_t>::type
f16_to_f32(float32x2_t in)
{
    return in;
}

template <typename T>
void pooling2_nchw_maxpool_indices(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    Iterator  in(src, window_src);
    Iterator  out(dst0, window);
    Iterator  indices(dst1, window);
    const int pool_pad_top  = pool_info.pad_stride_info.pad_top();
    const int pool_pad_left = pool_info.pad_stride_info.pad_left();
    int       pool_stride_x = 0;
    int       pool_stride_y = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const uint8_t *const src_top_ptr    = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
    const uint8_t *const src_bottom_ptr = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));
    const int            pad_left       = src->info()->padding().left;
    const int            pad_right      = src->info()->padding().right;
    const int            in_stride_y    = static_cast<int>(src->info()->strides_in_bytes().y());

    execute_window_loop(window, [&](const Coordinates & id)
    {
        auto        top_data        = wrapper::vload(reinterpret_cast<const T *>(src_top_ptr + in.offset()));
        auto        bottom_data     = wrapper::vload(reinterpret_cast<const T *>(src_bottom_ptr + in.offset()));
        float32x2_t top_data_f32    = f16_to_f32<T>(top_data);
        float32x2_t bottom_data_f32 = f16_to_f32<T>(bottom_data);

        // Calculate max data, compare top first, then bottom, to make sue the first max is recorded.
        const float32x2_t max_data_top      = vpmax_f32(top_data_f32, top_data_f32);
        const float32x2_t max_data_bottom   = vpmax_f32(bottom_data_f32, bottom_data_f32);
        const float32x2_t max_data          = vmax_f32(max_data_top, max_data_bottom);
        *(reinterpret_cast<T *>(out.ptr())) = static_cast<T>(vget_lane_f32(max_data, 0));

        // Calculate max data indice, which will be used in max unpool.
        const uint32_t   offset_base              = offset_no_padding<T>(in.offset(), id, *src->info(), pool_stride_x, pool_stride_y, DataLayout::NCHW);
        const uint32_t   offset_top               = (uint32_t)(offset_base / sizeof(T));
        const uint32_t   offset_bottom            = offset_top + in_stride_y / sizeof(T) - pad_right - pad_left;
        const uint32x2_t voffset_top              = { offset_top, offset_top + 1u };
        const uint32x2_t voffset_bottom           = { offset_bottom, offset_bottom + 1u };
        const uint32x2_t tmp_indices_top          = vbsl_u32(vcge_f32(top_data_f32, vrev64_f32(top_data_f32)), voffset_top, vrev64_u32(voffset_top));
        const uint32x2_t tmp_indices_bottom       = vbsl_u32(vcge_f32(bottom_data_f32, vrev64_f32(bottom_data_f32)), voffset_bottom, vrev64_u32(voffset_bottom));
        *(reinterpret_cast<int *>(indices.ptr())) = vget_lane_u32(vbsl_u32(vcge_f32(max_data_top, max_data_bottom), tmp_indices_top, tmp_indices_bottom), 0);
    },
    in, out, indices);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void pooling2_fp16_neon_nchw(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    if(pool_info.pool_type == PoolingType::MAX && dst1)
    {
        pooling2_nchw_maxpool_indices<float16_t>(src, dst0, dst1, pool_info, window_src, window);
    }
    else
    {
        Iterator      in(src, window_src);
        Iterator      out(dst0, window);
        constexpr int pool_size       = 2;
        const int     pool_pad_right  = pool_info.pad_stride_info.pad_right();
        const int     pool_pad_top    = pool_info.pad_stride_info.pad_top();
        const int     pool_pad_left   = pool_info.pad_stride_info.pad_left();
        const int     pool_pad_bottom = pool_info.pad_stride_info.pad_bottom();
        int           pool_stride_x, pool_stride_y = 0;
        std::tie(pool_stride_x, pool_stride_y)     = pool_info.pad_stride_info.stride();
        const int upper_bound_w = src->info()->dimension(0) + (pool_info.exclude_padding ? 0 : pool_pad_right);
        const int upper_bound_h = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);

        const unsigned char *const src_top_ptr    = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
        const unsigned char *const src_bottom_ptr = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));

        execute_window_loop(window, [&](const Coordinates & id)
        {
            float16x4_t top_data    = vld1_f16(reinterpret_cast<const float16_t *>(src_top_ptr + in.offset()));
            float16x4_t bottom_data = vld1_f16(reinterpret_cast<const float16_t *>(src_bottom_ptr + in.offset()));
            float16x4_t res         = {};

            // Get power of 2 in case of l2 pooling
            if(pool_info.pool_type == PoolingType::L2)
            {
                top_data    = vmul_f16(top_data, top_data);
                bottom_data = vmul_f16(bottom_data, bottom_data);
            }

            if(pool_info.pool_type != PoolingType::MAX)
            {
                const float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size, pool_size, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                        pool_stride_y);
                const float16x4_t scale_v = vdup_n_f16(scale);

                const float16x4_t sum_data = vadd_f16(top_data, bottom_data);
                res                        = vmul_f16(vpadd_f16(sum_data, sum_data), scale_v);
            }
            else
            {
                const float16x4_t max_data = vmax_f16(top_data, bottom_data);
                res                        = vpmax_f16(max_data, max_data);
            }

            // Calculate square-root in case of l2 pooling
            if(pool_info.pool_type == PoolingType::L2)
            {
                res = vinv_f16(vinvsqrt_f16(res));
            }

            // Store result
            *(reinterpret_cast<float16_t *>(out.ptr())) = vget_lane_f16(res, 0);
        },
        in, out);
    }
}

void poolingMxN_fp16_neon_nchw(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    Iterator in(src, window_src);
    Iterator out(dst0, window);

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

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float16_t   res  = 0.0f;
        float16x8_t vres = vdupq_n_f16(0.0f);

        if(pool_info.pool_type != PoolingType::MAX)
        {
            // Calculate scale
            const float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                    pool_stride_y);

            // Perform pooling

            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 8); x += 8)
                {
                    const float16x8_t data = vld1q_f16(reinterpret_cast<const float16_t *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                                           (src->info()->strides_in_bytes().y())));

                    // Get power of 2 in case of l2 pooling and accumulate
                    if(pool_info.pool_type == PoolingType::L2)
                    {
                        vres = vaddq_f16(vres, vmulq_f16(data, data));
                    }
                    else
                    {
                        vres = vaddq_f16(vres, data);
                    }
                }

                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    float16_t data = *(reinterpret_cast<const float16_t *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x())
                                                                           + (y - pool_pad_top) * static_cast<int>(src->info()->strides_in_bytes().y())));

                    // Get power of 2 in case of l2 pooling
                    if(pool_info.pool_type == PoolingType::L2)
                    {
                        data *= data;
                    }

                    res += data;
                }
            }

            // Reduction
            float16x4_t tmp = vpadd_f16(vget_high_f16(vres), vget_low_f16(vres));
            res += vget_lane_f16(tmp, 0);
            res += vget_lane_f16(tmp, 1);
            res += vget_lane_f16(tmp, 2);
            res += vget_lane_f16(tmp, 3);

            // Divide by scale
            res *= scale;
        }
        else
        {
            float16x8_t vres = vdupq_n_f16(std::numeric_limits<float>::lowest());
            res              = std::numeric_limits<float>::lowest();

            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 8); x += 8)
                {
                    const float16x8_t data = vld1q_f16(reinterpret_cast<const float16_t *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                                           (src->info()->strides_in_bytes().y())));
                    vres                   = vmaxq_f16(vres, data);
                }

                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    const float16_t data = *(reinterpret_cast<const float16_t *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x())
                                                                                 + (y - pool_pad_top) * static_cast<int>(src->info()->strides_in_bytes().y())));
                    res = std::max(res, data);
                }
            }

            float16x4_t tmp = vpmax_f16(vget_high_f16(vres), vget_low_f16(vres));
            res             = std::max(res, vget_lane_f16(tmp, 0));
            res             = std::max(res, vget_lane_f16(tmp, 1));
            res             = std::max(res, vget_lane_f16(tmp, 2));
            res             = std::max(res, vget_lane_f16(tmp, 3));
        }

        // Calculate square-root in case of l2 pooling
        if(pool_info.pool_type == PoolingType::L2)
        {
            res = std::sqrt(res);
        }

        // Store result
        *(reinterpret_cast<float16_t *>(out.ptr())) = res;
    },
    in, out);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

void poolingMxN_fp32_neon_nchw(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    Iterator in(src, window_src);
    Iterator out(dst0, window);

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

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float res = 0.0f;

        if(pool_info.pool_type != PoolingType::MAX)
        {
            // Calculate scale
            const float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                    pool_stride_y);

            // Perform pooling
            float32x4_t vres = vdupq_n_f32(0.0f);

            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 4); x += 4)
                {
                    const float32x4_t data = vld1q_f32(reinterpret_cast<const float *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                                       (src->info()->strides_in_bytes().y())));

                    // Get power of 2 in case of l2 pooling and accumulate
                    if(pool_info.pool_type == PoolingType::L2)
                    {
                        vres = vmlaq_f32(vres, data, data);
                    }
                    else
                    {
                        vres = vaddq_f32(vres, data);
                    }
                }

                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    float data = *(reinterpret_cast<const float *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                   (src->info()->strides_in_bytes().y())));

                    // Get power of 2 in case of l2 pooling
                    if(pool_info.pool_type == PoolingType::L2)
                    {
                        data *= data;
                    }

                    res += data;
                }
            }

#if defined(__aarch64__)
            // Reduction operation available on 64 bit architectures only
            res += vaddvq_f32(vres);
#else  // __aarch64__
            // Reduction
            float32x2_t tmp = vpadd_f32(vget_high_f32(vres), vget_low_f32(vres));
            tmp             = vpadd_f32(tmp, tmp);

            res += vget_lane_f32(tmp, 0);
#endif // __aarch64__
            // Divide by scale
            res *= scale;
        }
        else
        {
            float32x4_t vres = vdupq_n_f32(std::numeric_limits<float>::lowest());
            res              = std::numeric_limits<float>::lowest();

            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 4); x += 4)
                {
                    const float32x4_t data = vld1q_f32(reinterpret_cast<const float *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                                       (src->info()->strides_in_bytes().y())));
                    vres                   = vmaxq_f32(vres, data);
                }

                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    const float data = *(reinterpret_cast<const float *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                         (src->info()->strides_in_bytes().y())));
                    res              = std::max(res, data);
                }
            }
#if defined(__aarch64__)
            // Reduction operation available on 64 bit architectures only
            res = std::max(vmaxvq_f32(vres), res);
#else  // __aarch64__
            float32x2_t tmp = vpmax_f32(vget_high_f32(vres), vget_low_f32(vres));
            tmp             = vpmax_f32(tmp, tmp);

            res = std::max(res, vget_lane_f32(tmp, 0));
#endif // __aarch64__
        }

        // Calculate square-root in case of l2 pooling
        if(pool_info.pool_type == PoolingType::L2)
        {
            res = std::sqrt(res);
        }

        // Store result
        *(reinterpret_cast<float *>(out.ptr())) = res;
    },
    in, out);
}

void pooling2_fp32_neon_nchw(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    if(pool_info.pool_type == PoolingType::MAX && dst1)
    {
        pooling2_nchw_maxpool_indices<float>(src, dst0, dst1, pool_info, window_src, window);
    }
    else
    {
        Iterator      in(src, window_src);
        Iterator      out(dst0, window);
        constexpr int pool_size       = 2;
        const int     pool_pad_right  = pool_info.pad_stride_info.pad_right();
        const int     pool_pad_top    = pool_info.pad_stride_info.pad_top();
        const int     pool_pad_left   = pool_info.pad_stride_info.pad_left();
        const int     pool_pad_bottom = pool_info.pad_stride_info.pad_bottom();
        int           pool_stride_x   = 0;
        int           pool_stride_y   = 0;
        std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
        const int upper_bound_w = src->info()->dimension(0) + (pool_info.exclude_padding ? 0 : pool_pad_right);
        const int upper_bound_h = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);

        const uint8_t *const src_top_ptr    = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
        const uint8_t *const src_bottom_ptr = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));

        execute_window_loop(window, [&](const Coordinates & id)
        {
            const auto  in_top_ptr    = reinterpret_cast<const float *>(src_top_ptr + in.offset());
            const auto  in_bottom_ptr = reinterpret_cast<const float *>(src_bottom_ptr + in.offset());
            float32x2_t top_data      = vld1_f32(in_top_ptr);
            float32x2_t bottom_data   = vld1_f32(in_bottom_ptr);
            float32x2_t res           = {};
            float       final_res     = 0;
            // Get power of 2 in case of l2 pooling
            if(pool_info.pool_type == PoolingType::L2)
            {
                top_data    = vmul_f32(top_data, top_data);
                bottom_data = vmul_f32(bottom_data, bottom_data);
            }

            if(pool_info.pool_type != PoolingType::MAX)
            {
                // Calculate scale
                float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size, pool_size, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                  pool_stride_y);
                const float32x2_t scale_v = vdup_n_f32(scale);

                // Perform pooling
                const float32x2_t sum_data = vadd_f32(top_data, bottom_data);
                res                        = vmul_f32(vpadd_f32(sum_data, sum_data), scale_v);
            }
            else
            {
                const float32x2_t max_data = vmax_f32(top_data, bottom_data);
                res                        = vpmax_f32(max_data, max_data);
            }
            final_res = vget_lane_f32(res, 0);

            // Calculate square-root in case of l2 pooling
            if(pool_info.pool_type == PoolingType::L2)
            {
                final_res = sqrt(final_res);
            }

            // Store result
            *(reinterpret_cast<float *>(out.ptr())) = final_res;
        },
        in, out);
    }
}

void pooling3_fp32_neon_nchw(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    Iterator in(src, window_src);
    Iterator out(dst0, window);

    constexpr const int pool_size       = 3;
    const int           pool_pad_right  = pool_info.pad_stride_info.pad_right();
    const int           pool_pad_top    = pool_info.pad_stride_info.pad_top();
    const int           pool_pad_left   = pool_info.pad_stride_info.pad_left();
    const int           pool_pad_bottom = pool_info.pad_stride_info.pad_bottom();
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int upper_bound_w = src->info()->dimension(0) + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);

    const uint8_t *const src_top_ptr    = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
    const uint8_t *const src_middle_ptr = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));
    const uint8_t *const src_bottom_ptr = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 2));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float32x4_t top_data    = vld1q_f32(reinterpret_cast<const float *>(src_top_ptr + in.offset()));
        float32x4_t middle_data = vld1q_f32(reinterpret_cast<const float *>(src_middle_ptr + in.offset()));
        float32x4_t bottom_data = vld1q_f32(reinterpret_cast<const float *>(src_bottom_ptr + in.offset()));
        float32x2_t res         = {};
        float       final_res   = 0;

        // Get power of 2 in case of l2 pooling
        if(pool_info.pool_type == PoolingType::L2)
        {
            top_data    = vmulq_f32(top_data, top_data);
            middle_data = vmulq_f32(middle_data, middle_data);
            bottom_data = vmulq_f32(bottom_data, bottom_data);
        }

        if(pool_info.pool_type != PoolingType::MAX)
        {
            // Calculate scale
            float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size, pool_size, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                              pool_stride_y);
            const float32x2_t scale_v = vdup_n_f32(scale);

            // Perform pooling
            const float32x4_t sum_data = vaddq_f32(vaddq_f32(top_data, bottom_data), middle_data);
            res                        = vpadd_f32(vget_high_f32(vsetq_lane_f32(0.f, sum_data, 3)), vget_low_f32(sum_data));
            res                        = vmul_f32(vpadd_f32(res, res), scale_v);
        }
        else
        {
            const float32x4_t max_data = vmaxq_f32(vmaxq_f32(top_data, bottom_data), middle_data);
            res                        = vpmax_f32(vget_high_f32(vsetq_lane_f32(-std::numeric_limits<float>::max(), max_data, 3)), vget_low_f32(max_data));
            res                        = vpmax_f32(res, res);
        }
        final_res = vget_lane_f32(res, 0);

        // Calculate square-root in case of l2 pooling
        if(pool_info.pool_type == PoolingType::L2)
        {
            final_res = sqrt(final_res);
        }

        // Store result
        *(reinterpret_cast<float *>(out.ptr())) = final_res;
    },
    in, out);
}

void pooling7_fp32_neon_nchw(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    Iterator in(src, window_src);
    Iterator out(dst0, window);

    constexpr const int pool_size       = 7;
    const int           pool_pad_right  = pool_info.pad_stride_info.pad_right();
    const int           pool_pad_top    = pool_info.pad_stride_info.pad_top();
    const int           pool_pad_left   = pool_info.pad_stride_info.pad_left();
    const int           pool_pad_bottom = pool_info.pad_stride_info.pad_bottom();
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int upper_bound_w = src->info()->dimension(0) + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);

    std::array<const uint8_t *, pool_size> src_ptrs{ {} };
    for(int i = 0; i < pool_size; ++i)
    {
        src_ptrs[i] = src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + i));
    }

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float32x2_t res       = {};
        float       final_res = 0.f;
        if(pool_info.pool_type != PoolingType::MAX)
        {
            // Calculate scale
            float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size, pool_size, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                              pool_stride_y);
            const float32x2_t scale_v = vdup_n_f32(scale);

            // Perform pooling
            float32x4x2_t data = vld2q_f32(reinterpret_cast<const float *>(src_ptrs[0] + in.offset()));
            // Get power of 2 in case of l2 pooling
            if(pool_info.pool_type == PoolingType::L2)
            {
                data.val[0] = vmulq_f32(data.val[0], data.val[0]);
                data.val[1] = vmulq_f32(data.val[1], data.val[1]);
            }
            float32x4_t sum_data = vaddq_f32(data.val[0], vsetq_lane_f32(0.f, data.val[1], 3));
            for(int i = 1; i < pool_size; ++i)
            {
                data = vld2q_f32(reinterpret_cast<const float *>(src_ptrs[i] + in.offset()));
                // Get power of 2 in case of l2 pooling
                if(pool_info.pool_type == PoolingType::L2)
                {
                    data.val[0] = vmulq_f32(data.val[0], data.val[0]);
                    data.val[1] = vmulq_f32(data.val[1], data.val[1]);
                }
                sum_data = vaddq_f32(sum_data, data.val[0]);
                sum_data = vaddq_f32(sum_data, vsetq_lane_f32(0.f, data.val[1], 3));
            }
            res = vpadd_f32(vget_high_f32(sum_data), vget_low_f32(sum_data));
            res = vmul_f32(vpadd_f32(res, res), scale_v);
        }
        else
        {
            float32x4x2_t max_data = vld2q_f32(reinterpret_cast<const float *>(src_ptrs[0] + in.offset()));
            for(int i = 1; i < pool_size; ++i)
            {
                const float32x4x2_t data = vld2q_f32(reinterpret_cast<const float *>(src_ptrs[i] + in.offset()));
                max_data                 = vmax2q_f32(max_data, data);
            }
            res = vpmax_f32(vget_high_f32(vsetq_lane_f32(-std::numeric_limits<float>::max(), max_data.val[1], 3)), vget_low_f32(max_data.val[1]));
            res = vpmax_f32(res, vpmax_f32(vget_high_f32(max_data.val[0]), vget_low_f32(max_data.val[0])));
            res = vpmax_f32(res, res);
        }
        final_res = vget_lane_f32(res, 0);

        // Calculate square-root in case of l2 pooling
        if(pool_info.pool_type == PoolingType::L2)
        {
            final_res = sqrt(final_res);
        }

        // Store result
        *(reinterpret_cast<float *>(out.ptr())) = final_res;
    },
    in, out);
}
} // namespace cpu
} // namespace arm_compute

#endif // ENABLE_NCHW_KERNELS