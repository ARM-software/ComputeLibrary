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
#include "src/core/cpu/kernels/pooling/neon/list.h"
#include "src/core/helpers/WindowHelpers.h"

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

namespace arm_compute
{
namespace cpu
{
namespace
{
void pooling2_f16_maxpool_indices(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();
    const int window_step_x  = 8;

    Window window_out = window;
    window_out.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, window_src);
    Iterator out(dst0, window_out);
    Iterator indices(dst1, window_out);

    const int pool_pad_top  = pool_info.pad_stride_info.pad_top();
    const int pool_pad_left = pool_info.pad_stride_info.pad_left();

    int pool_stride_x = 0;
    int pool_stride_y = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();

    const int pad_right   = src->info()->padding().right;
    const int in_stride_y = static_cast<int>(src->info()->strides_in_bytes().y());
    const int in_stride_z = static_cast<int>(src->info()->strides_in_bytes().z());

    execute_window_loop(window_out, [&](const Coordinates & id)
    {
        const int idx_width    = id.y() * pool_stride_x;
        const int idx_height   = id.z() * pool_stride_y;
        const int pool_limit_y = pool_pad_top - idx_height;
        const int pool_limit_x = pool_pad_left - idx_width;

        const int pool_start_y = std::max(0, window_src.z().start() + pool_limit_y);
        const int pool_start_x = std::max(0, window_src.y().start() + pool_limit_x);
        const int in_x0_offset = (pool_start_x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (pool_start_y - pool_pad_top) * static_cast<int>(src->info()->strides_in_bytes().z());
        const int in_x1_offset = (pool_start_x + 1 - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (pool_start_y - pool_pad_top) * static_cast<int>
                                 (src->info()->strides_in_bytes().z());
        const int in_x2_offset = (pool_start_x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (pool_start_y + 1 - pool_pad_top) * static_cast<int>
                                 (src->info()->strides_in_bytes().z());
        const int in_x3_offset = (pool_start_x + 1 - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (pool_start_y + 1 - pool_pad_top) * static_cast<int>
                                 (src->info()->strides_in_bytes().z());

        int x_off = window_start_x;
        for(; x_off <= (window_end_x - window_step_x); x_off += window_step_x)
        {
            const auto  in_x0_ptr = reinterpret_cast<const float16_t *>(in.ptr() + in_x0_offset) + x_off;
            const auto  in_x1_ptr = reinterpret_cast<const float16_t *>(in.ptr() + in_x1_offset) + x_off;
            const auto  in_x2_ptr = reinterpret_cast<const float16_t *>(in.ptr() + in_x2_offset) + x_off;
            const auto  in_x3_ptr = reinterpret_cast<const float16_t *>(in.ptr() + in_x3_offset) + x_off;
            const auto  v_x0      = vld1q_f16(in_x0_ptr);
            const auto  v_x1      = vld1q_f16(in_x1_ptr);
            const auto  v_x2      = vld1q_f16(in_x2_ptr);
            const auto  v_x3      = vld1q_f16(in_x3_ptr);
            float16x8_t vres      = vmaxq_f16(vmaxq_f16(v_x2, v_x3), vmaxq_f16(v_x0, v_x1));
            // Store result
            vst1q_f16(reinterpret_cast<float16_t *>(out.ptr()) + x_off, vres);

            const uint32_t   offset_base    = offset_no_padding<float16_t>(in.offset(), id, *src->info(), pool_stride_x, pool_stride_y);
            const uint32_t   offset_x0      = (uint32_t)offset_base / sizeof(float16_t) + x_off;
            const uint32_t   offset_x1      = (uint32_t)offset_x0 + in_stride_y / sizeof(float16_t) - pad_right;
            const uint32_t   offset_x2      = (uint32_t)offset_x0 + in_stride_z / sizeof(float16_t) - pad_right * src->info()->tensor_shape()[1];
            const uint32_t   offset_x3      = (uint32_t)offset_x2 + in_stride_y / sizeof(float16_t) - pad_right;
            const uint32x4_t voffset_x0_0   = { offset_x0, offset_x0 + 1, offset_x0 + 2, offset_x0 + 3 };
            const uint32x4_t voffset_x0_1   = { offset_x0 + 4, offset_x0 + 5, offset_x0 + 6, offset_x0 + 7 };
            const uint16x8_t voffset_x0     = vcombine_u16(vmovn_u32(voffset_x0_0), vmovn_u32(voffset_x0_1));
            const uint32x4_t voffset_x1_0   = { offset_x1, offset_x1 + 1, offset_x1 + 2, offset_x1 + 3 };
            const uint32x4_t voffset_x1_1   = { offset_x1 + 4, offset_x1 + 5, offset_x1 + 6, offset_x1 + 7 };
            const uint16x8_t voffset_x1     = vcombine_u16(vmovn_u32(voffset_x1_0), vmovn_u32(voffset_x1_1));
            const uint32x4_t voffset_x2_0   = { offset_x2, offset_x2 + 1, offset_x2 + 2, offset_x2 + 3 };
            const uint32x4_t voffset_x2_1   = { offset_x2 + 4, offset_x2 + 5, offset_x2 + 6, offset_x2 + 7 };
            const uint16x8_t voffset_x2     = vcombine_u16(vmovn_u32(voffset_x2_0), vmovn_u32(voffset_x2_1));
            const uint32x4_t voffset_x3_0   = { offset_x3, offset_x3 + 1, offset_x3 + 2, offset_x3 + 3 };
            const uint32x4_t voffset_x3_1   = { offset_x3 + 4, offset_x3 + 5, offset_x3 + 6, offset_x3 + 7 };
            const uint16x8_t voffset_x3     = vcombine_u16(vmovn_u32(voffset_x3_0), vmovn_u32(voffset_x3_1));
            const uint16x8_t tmp_indices0   = vbslq_u16(vcgeq_f16(v_x0, v_x1), voffset_x0, voffset_x1);
            const uint16x8_t tmp_indices1   = vbslq_u16(vcgeq_f16(v_x2, v_x3), voffset_x2, voffset_x3);
            const uint16x8_t tmp_indices2   = vbslq_u16(vcgeq_f16(vmaxq_f16(v_x0, v_x1), vmaxq_f16(v_x2, v_x3)), tmp_indices0, tmp_indices1);
            const uint32x4_t tmp_indeces3_0 = vmovl_u16(vget_low_u16(tmp_indices2));
            const uint32x4_t tmp_indeces3_1 = vmovl_u16(vget_high_u16(tmp_indices2));
            // Store indicies
            vst1q_u32(reinterpret_cast<uint32_t *>(indices.ptr()) + x_off, tmp_indeces3_0);
            vst1q_u32(reinterpret_cast<uint32_t *>(indices.ptr() + 16) + x_off, tmp_indeces3_1);
        }

        // Left-overs loop
        for(; x_off < window_end_x; ++x_off)
        {
            const auto x0  = *(reinterpret_cast<const float16_t *>(in.ptr() + in_x0_offset) + x_off);
            const auto x1  = *(reinterpret_cast<const float16_t *>(in.ptr() + in_x1_offset) + x_off);
            const auto x2  = *(reinterpret_cast<const float16_t *>(in.ptr() + in_x2_offset) + x_off);
            const auto x3  = *(reinterpret_cast<const float16_t *>(in.ptr() + in_x3_offset) + x_off);
            float16_t  res = std::max(std::max(x2, x3), std::max(x0, x1));

            // Store result
            *(reinterpret_cast<float16_t *>(out.ptr()) + x_off) = res;

            const uint32_t offset_base = offset_no_padding<float16_t>(in.offset(), id, *src->info(), pool_stride_x, pool_stride_y);
            const uint32_t offset_x0   = (uint32_t)offset_base / sizeof(float16_t) + x_off;
            const uint32_t offset_x1   = (uint32_t)offset_x0 + in_stride_y / sizeof(float16_t) - pad_right;
            const uint32_t offset_x2   = (uint32_t)offset_x0 + in_stride_z / sizeof(float16_t) - pad_right * src->info()->tensor_shape()[1];
            const uint32_t offset_x3   = (uint32_t)offset_x2 + in_stride_y / sizeof(float16_t) - pad_right;
            const uint32_t tmp_idx0    = (x0 >= x1) ? offset_x0 : offset_x1;
            const uint32_t tmp_idx1    = (x2 >= x3) ? offset_x2 : offset_x3;
            const uint32_t tmp_idx2    = (std::max(x0, x1) >= std::max(x2, x3)) ? tmp_idx0 : tmp_idx1;

            // Store indices
            *(reinterpret_cast<uint32_t *>(indices.ptr()) + x_off) = tmp_idx2;
        }
    },
    in, out, indices);
}
}

void poolingMxN_fp16_neon_nhwc(const ITensor *src, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &pool_info, const Window &window_src, const Window &window)
{
    if(pool_info.pool_size == Size2D(2, 2) && pool_info.pool_type == PoolingType::MAX && dst1)
    {
        pooling2_f16_maxpool_indices(src, dst0, dst1, pool_info, window_src, window);
    }
    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();
    const int window_step_x  = 8;

    Window window_out = window;
    window_out.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, window_src);
    Iterator out(dst0, window_out);

    const int pool_size_x     = pool_info.is_global_pooling ? src->info()->tensor_shape().y() : pool_info.pool_size.width;
    const int pool_size_y     = pool_info.is_global_pooling ? src->info()->tensor_shape().z() : pool_info.pool_size.height;
    const int pool_pad_right  = pool_info.pad_stride_info.pad_right();
    const int pool_pad_top    = pool_info.pad_stride_info.pad_top();
    const int pool_pad_left   = pool_info.pad_stride_info.pad_left();
    const int pool_pad_bottom = pool_info.pad_stride_info.pad_bottom();
    int       pool_stride_x   = 0;
    int       pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int upper_bound_w = src->info()->dimension(1) + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = src->info()->dimension(2) + (pool_info.exclude_padding ? 0 : pool_pad_bottom);

    float16x8_t vres;

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
                // Calculate scale
                const float scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NHWC, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                        pool_stride_y);
                const float16x8_t scale_v = vdupq_n_f16(scale);

                // Perform pooling
                vres = vdupq_n_f16(0.0f);
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const float16x8_t data = vld1q_f16(reinterpret_cast<const float16_t *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                               (src->info()->strides_in_bytes().z())) + x_off);

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
                }
                // Divide by scale
                vres = vmulq_f16(vres, scale_v);
            }
            else
            {
                vres = vdupq_n_f16(std::numeric_limits<float>::lowest());

                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const float16x8_t data = vld1q_f16(reinterpret_cast<const float16_t *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                               (src->info()->strides_in_bytes().z())) + x_off);
                        vres                   = vmaxq_f16(vres, data);
                    }
                }
            }

            // Calculate square-root in case of l2 pooling
            if(pool_info.pool_type == PoolingType::L2)
            {
                float16x8_t sqrt_reciprocal = vrsqrteq_f16(vres);
                vres                        = vmulq_f16(vres, vmulq_f16(vrsqrtsq_f16(vmulq_f16(vres, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal));
            }

            // Store result
            vst1q_f16(reinterpret_cast<float16_t *>(out.ptr()) + x_off, vres);
        }

        // Left-overs loop
        for(; x_off < window_end_x; ++x_off)
        {
            float16_t res = 0.0f;

            if(pool_info.pool_type != PoolingType::MAX)
            {
                // Calculate scale
                const float16_t scale = calculate_avg_scale(pool_info.exclude_padding, DataLayout::NHWC, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                            pool_stride_y);

                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const float data = *(reinterpret_cast<const float16_t *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                 (src->info()->strides_in_bytes().z())) + x_off);

                        // Get power of 2 in case of l2 pooling and accumulate
                        if(pool_info.pool_type == PoolingType::L2)
                        {
                            res += data * data;
                        }
                        else
                        {
                            res += data;
                        }
                    }
                }

                // Divide by scale
                res *= scale;
            }
            else
            {
                res = std::numeric_limits<float>::lowest();
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const float16_t data = *(reinterpret_cast<const float16_t *>(in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                     (src->info()->strides_in_bytes().z())) + x_off);
                        res                  = std::max(res, data);
                    }
                }
            }

            // Calculate square-root in case of l2 pooling
            if(pool_info.pool_type == PoolingType::L2)
            {
                res = std::sqrt(res);
            }

            // Store result
            *(reinterpret_cast<float16_t *>(out.ptr()) + x_off) = res;
        }
    },
    in, out);
}
} // namespace cpu
} // namespace arm_compute

#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */