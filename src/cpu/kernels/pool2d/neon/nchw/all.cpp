/*
 * Copyright (c) 2021-2023 Arm Limited.
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

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include "src/cpu/kernels/pool2d/neon/impl.h"
#include "src/cpu/kernels/pool2d/neon/list.h"

#include <limits>

#ifdef ENABLE_NCHW_KERNELS
namespace arm_compute
{
namespace cpu
{
#define READ_2_RIGHT_BOUNDARY_AWARE(height, width, pad_left, pad_top, x, y, ptr, fval) \
    (x == width + pad_left - 1) ? vset_lane_f32(*(ptr), vdup_n_f32(fval), 0) : vld1_f32(ptr)
#define READ_2_LEFT_BOUNDARY_AWARE(height, width, pad_left, pad_top, x, y, ptr, fval) \
    (x == pad_left - 1) ? vset_lane_f32(*(1 + ptr), vdup_n_f32(fval), 1)              \
                        : READ_2_RIGHT_BOUNDARY_AWARE(height, width, pad_left, pad_top, x, y, ptr, fval)
#define READ_2_BOUNDARY_AWARE(height, width, pad_left, pad_top, x, y, ptr, fval)                   \
    ((y < pad_top) || (x < pad_left - 1) || (y >= height + pad_top) || (x > width + pad_left - 1)) \
        ? vdup_n_f32(fval)                                                                         \
        : READ_2_LEFT_BOUNDARY_AWARE(height, width, pad_left, pad_top, x, y, ptr, fval)

#define READ_4_BOUNDARY_AWARE(height, width, pad_left, pad_top, x, y, ptr, fval)           \
    vcombine_f32(READ_2_BOUNDARY_AWARE(height, width, pad_left, pad_top, x, y, ptr, fval), \
                 READ_2_BOUNDARY_AWARE(height, width, pad_left, pad_top, (x + 2), y, (ptr + 2), fval))

float32x4x2_t
read_8_boundary_aware(int height, int width, int pad_left, int pad_top, int x, int y, const float *ptr, float fval)
{
    float32x4x2_t vec;
    vec.val[0] = READ_4_BOUNDARY_AWARE(height, width, pad_left, pad_top, x, y, ptr, fval);
    vec.val[1] = READ_4_BOUNDARY_AWARE(height, width, pad_left, pad_top, (x + 4), y, (ptr + 4), fval);
    return vec;
}

void poolingMxN_fp32_neon_nchw(const ITensor    *src,
                               ITensor          *dst0,
                               ITensor          *dst1,
                               PoolingLayerInfo &pool_info,
                               const Window     &window_src,
                               const Window     &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    Iterator in(src, window_src);
    Iterator out(dst0, window);

    const int pool_size_x = pool_info.is_global_pooling ? src->info()->tensor_shape().x() : pool_info.pool_size.width;
    const int pool_size_y = pool_info.is_global_pooling ? src->info()->tensor_shape().y() : pool_info.pool_size.height;
    const int pool_pad_right               = pool_info.pad_stride_info.pad_right();
    const int pool_pad_top                 = pool_info.pad_stride_info.pad_top();
    const int pool_pad_left                = pool_info.pad_stride_info.pad_left();
    const int pool_pad_bottom              = pool_info.pad_stride_info.pad_bottom();
    int       pool_stride_x                = 0;
    int       pool_stride_y                = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int   src_w                      = src->info()->dimension(0);
    const int   src_h                      = src->info()->dimension(1);
    const int   upper_bound_w              = src_w + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int   upper_bound_h              = src_h + (pool_info.exclude_padding ? 0 : pool_pad_bottom);
    const float min_value                  = get_initial_min<float>(pool_info.use_inf_as_limit);
    const float fill_value                 = (pool_info.pool_type == PoolingType::MAX) ? min_value : 0.0f;

    execute_window_loop(
        window,
        [&](const Coordinates &id)
        {
            float res = 0.0f;

            if (pool_info.pool_type != PoolingType::MAX)
            {
                // Calculate scale
                const float scale = calculate_avg_scale_pool2d(
                    pool_info.exclude_padding, DataLayout::NCHW, id, pool_size_x, pool_size_y, upper_bound_w,
                    upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);

                // Perform pooling
                for (int y = 0; y < pool_size_y; ++y)
                {
                    for (int x = 0; x < pool_size_x; ++x)
                    {
                        const auto ptr = reinterpret_cast<const float *>(
                            in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) +
                            (y - pool_pad_top) * static_cast<int>(src->info()->strides_in_bytes().y()));

                        const int idx  = x + id.x() * pool_stride_x - pool_pad_left;
                        const int idy  = y + id.y() * pool_stride_y - pool_pad_top;
                        float     data = (idx < 0 || idy < 0 || idx >= src_w || idy >= src_h) ? fill_value : *ptr;

                        if (pool_info.pool_type == PoolingType::L2)
                        {
                            data *= data;
                        }

                        res += data;
                    }
                }

                // Divide by scale
                res *= scale;
            }
            else // if max pooling
            {
                res = min_value;

                for (int y = 0; y < pool_size_y; ++y)
                {
                    for (int x = 0; x < pool_size_x; ++x)
                    {
                        const auto ptr = reinterpret_cast<const float *>(
                            in.ptr() + (x - pool_pad_left) * static_cast<int>(src->info()->strides_in_bytes().x()) +
                            (y - pool_pad_top) * static_cast<int>(src->info()->strides_in_bytes().y()));

                        const int idx  = x + id.x() * pool_stride_x - pool_pad_left;
                        const int idy  = y + id.y() * pool_stride_y - pool_pad_top;
                        float     data = (idx < 0 || idy < 0 || idx >= src_w || idy >= src_h) ? fill_value : *ptr;
                        res            = std::max(res, data);
                    }
                }
            }

            // Calculate square-root in case of l2 pooling
            if (pool_info.pool_type == PoolingType::L2)
            {
                res = std::sqrt(res);
            }

            // Store result
            *(reinterpret_cast<float *>(out.ptr())) = res;
        },
        in, out);
}

void pooling2_fp32_neon_nchw(const ITensor    *src,
                             ITensor          *dst0,
                             ITensor          *dst1,
                             PoolingLayerInfo &pool_info,
                             const Window     &window_src,
                             const Window     &window)
{
    if (pool_info.pool_type == PoolingType::MAX && dst1)
    {
        pooling2_nchw_maxpool_indices<float>(src, dst0, dst1, pool_info, window_src, window);
    }
    else
    {
        Iterator      in(src, window_src);
        Iterator      out(dst0, window);
        constexpr int pool_size                = 2;
        const int     pool_pad_right           = pool_info.pad_stride_info.pad_right();
        const int     pool_pad_top             = pool_info.pad_stride_info.pad_top();
        const int     pool_pad_left            = pool_info.pad_stride_info.pad_left();
        const int     pool_pad_bottom          = pool_info.pad_stride_info.pad_bottom();
        int           pool_stride_x            = 0;
        int           pool_stride_y            = 0;
        std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
        const int   src_w                      = src->info()->dimension(0);
        const int   src_h                      = src->info()->dimension(1);
        const int   upper_bound_w              = src_w + (pool_info.exclude_padding ? 0 : pool_pad_right);
        const int   upper_bound_h              = src_h + (pool_info.exclude_padding ? 0 : pool_pad_bottom);
        const float min_value                  = get_initial_min<float>(pool_info.use_inf_as_limit);
        const float fill_value                 = (pool_info.pool_type == PoolingType::MAX) ? min_value : 0.0f;

        const uint8_t *const src_top_ptr =
            src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
        const uint8_t *const src_bottom_ptr =
            src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));

        execute_window_loop(
            window,
            [&](const Coordinates &id)
            {
                const auto in_top_ptr    = reinterpret_cast<const float *>(src_top_ptr + in.offset());
                const auto in_bottom_ptr = reinterpret_cast<const float *>(src_bottom_ptr + in.offset());

                const auto x_val      = id.x() * pool_stride_x;
                const auto y_val_0    = id.y() * pool_stride_y;
                const auto y_val_1    = (id.y() * pool_stride_y) + 1;
                auto       top_data   = READ_2_BOUNDARY_AWARE(src_h, src_w, pool_pad_left, pool_pad_top, x_val, y_val_0,
                                                              in_top_ptr, fill_value);
                auto bottom_data      = READ_2_BOUNDARY_AWARE(src_h, src_w, pool_pad_left, pool_pad_top, x_val, y_val_1,
                                                              in_bottom_ptr, fill_value);
                float32x2_t res       = {};
                float       final_res = 0;

                // Get power of 2 in case of l2 pooling
                if (pool_info.pool_type == PoolingType::L2)
                {
                    top_data    = vmul_f32(top_data, top_data);
                    bottom_data = vmul_f32(bottom_data, bottom_data);
                }

                if (pool_info.pool_type != PoolingType::MAX)
                {
                    // Calculate scale
                    float scale = calculate_avg_scale_pool2d(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size,
                                                             pool_size, upper_bound_w, upper_bound_h, pool_pad_left,
                                                             pool_pad_top, pool_stride_x, pool_stride_y);
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
                if (pool_info.pool_type == PoolingType::L2)
                {
                    final_res = sqrt(final_res);
                }

                // Store result
                *(reinterpret_cast<float *>(out.ptr())) = final_res;
            },
            in, out);
    }
}

void pooling3_fp32_neon_nchw(const ITensor    *src,
                             ITensor          *dst0,
                             ITensor          *dst1,
                             PoolingLayerInfo &pool_info,
                             const Window     &window_src,
                             const Window     &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    Iterator in(src, window_src);
    Iterator out(dst0, window);

    constexpr const int pool_size          = 3;
    const int           pool_pad_right     = pool_info.pad_stride_info.pad_right();
    const int           pool_pad_top       = pool_info.pad_stride_info.pad_top();
    const int           pool_pad_left      = pool_info.pad_stride_info.pad_left();
    const int           pool_pad_bottom    = pool_info.pad_stride_info.pad_bottom();
    int                 pool_stride_x      = 0;
    int                 pool_stride_y      = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int   src_w                      = src->info()->dimension(0);
    const int   src_h                      = src->info()->dimension(1);
    const int   upper_bound_w              = src_w + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int   upper_bound_h              = src_h + (pool_info.exclude_padding ? 0 : pool_pad_bottom);
    const float min_value                  = get_initial_min<float>(pool_info.use_inf_as_limit);
    const float fill_value                 = (pool_info.pool_type == PoolingType::MAX) ? min_value : 0.0f;

    const uint8_t *const src_top_ptr =
        src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
    const uint8_t *const src_middle_ptr =
        src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));
    const uint8_t *const src_bottom_ptr =
        src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 2));

    execute_window_loop(
        window,
        [&](const Coordinates &id)
        {
            const auto in_top_ptr    = reinterpret_cast<const float *>(src_top_ptr + in.offset());
            const auto in_middle_ptr = reinterpret_cast<const float *>(src_middle_ptr + in.offset());
            const auto in_bottom_ptr = reinterpret_cast<const float *>(src_bottom_ptr + in.offset());

            const auto x_val   = id.x() * pool_stride_x;
            const auto y_val_0 = id.y() * pool_stride_y;
            const auto y_val_1 = (id.y() * pool_stride_y) + 1;
            const auto y_val_2 = (id.y() * pool_stride_y) + 2;
            auto top_data = READ_4_BOUNDARY_AWARE(src_h, src_w, pool_pad_left, pool_pad_top, x_val, y_val_0, in_top_ptr,
                                                  fill_value);
            auto middle_data = READ_4_BOUNDARY_AWARE(src_h, src_w, pool_pad_left, pool_pad_top, x_val, y_val_1,
                                                     in_middle_ptr, fill_value);
            auto bottom_data = READ_4_BOUNDARY_AWARE(src_h, src_w, pool_pad_left, pool_pad_top, x_val, y_val_2,
                                                     in_bottom_ptr, fill_value);

            float32x2_t res       = {};
            float       final_res = 0;

            // Get power of 2 in case of l2 pooling
            if (pool_info.pool_type == PoolingType::L2)
            {
                top_data    = vmulq_f32(top_data, top_data);
                middle_data = vmulq_f32(middle_data, middle_data);
                bottom_data = vmulq_f32(bottom_data, bottom_data);
            }

            if (pool_info.pool_type != PoolingType::MAX)
            {
                // Calculate scale
                float scale = calculate_avg_scale_pool2d(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size,
                                                         pool_size, upper_bound_w, upper_bound_h, pool_pad_left,
                                                         pool_pad_top, pool_stride_x, pool_stride_y);
                const float32x2_t scale_v = vdup_n_f32(scale);

                // Perform pooling
                const float32x4_t sum_data = vaddq_f32(vaddq_f32(top_data, bottom_data), middle_data);
                res = vpadd_f32(vget_high_f32(vsetq_lane_f32(0.f, sum_data, 3)), vget_low_f32(sum_data));
                res = vmul_f32(vpadd_f32(res, res), scale_v);
            }
            else
            {
                const float32x4_t max_data = vmaxq_f32(vmaxq_f32(top_data, bottom_data), middle_data);
                res = vpmax_f32(vget_high_f32(vsetq_lane_f32(min_value, max_data, 3)), vget_low_f32(max_data));
                res = vpmax_f32(res, res);
            }
            final_res = vget_lane_f32(res, 0);

            // Calculate square-root in case of l2 pooling
            if (pool_info.pool_type == PoolingType::L2)
            {
                final_res = sqrt(final_res);
            }

            // Store result
            *(reinterpret_cast<float *>(out.ptr())) = final_res;
        },
        in, out);
}

void pooling7_fp32_neon_nchw(const ITensor    *src,
                             ITensor          *dst0,
                             ITensor          *dst1,
                             PoolingLayerInfo &pool_info,
                             const Window     &window_src,
                             const Window     &window)
{
    ARM_COMPUTE_UNUSED(dst1);
    Iterator in(src, window_src);
    Iterator out(dst0, window);

    constexpr const int pool_size          = 7;
    const int           pool_pad_right     = pool_info.pad_stride_info.pad_right();
    const int           pool_pad_top       = pool_info.pad_stride_info.pad_top();
    const int           pool_pad_left      = pool_info.pad_stride_info.pad_left();
    const int           pool_pad_bottom    = pool_info.pad_stride_info.pad_bottom();
    int                 pool_stride_x      = 0;
    int                 pool_stride_y      = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int   src_w                      = src->info()->dimension(0);
    const int   src_h                      = src->info()->dimension(1);
    const int   upper_bound_w              = src_w + (pool_info.exclude_padding ? 0 : pool_pad_right);
    const int   upper_bound_h              = src_h + (pool_info.exclude_padding ? 0 : pool_pad_bottom);
    const float min_value                  = get_initial_min<float>(pool_info.use_inf_as_limit);
    const float fill_value                 = (pool_info.pool_type == PoolingType::MAX) ? min_value : 0.0f;

    std::array<const uint8_t *, pool_size> src_ptrs{{}};
    for (int i = 0; i < pool_size; ++i)
    {
        src_ptrs[i] =
            src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + i));
    }

    execute_window_loop(
        window,
        [&](const Coordinates &id)
        {
            auto in_ptr = reinterpret_cast<const float *>(src_ptrs[0] + in.offset());

            auto          x_val = id.x() * pool_stride_x;
            auto          y_val = id.y() * pool_stride_y;
            float32x4x2_t data =
                read_8_boundary_aware(src_h, src_w, pool_pad_left, pool_pad_top, x_val, y_val, in_ptr, fill_value);

            float32x2_t res       = {};
            float       final_res = 0.f;

            if (pool_info.pool_type != PoolingType::MAX)
            {
                // Calculate scale
                float scale = calculate_avg_scale_pool2d(pool_info.exclude_padding, DataLayout::NCHW, id, pool_size,
                                                         pool_size, upper_bound_w, upper_bound_h, pool_pad_left,
                                                         pool_pad_top, pool_stride_x, pool_stride_y);
                const float32x2_t scale_v = vdup_n_f32(scale);

                // Get power of 2 in case of l2 pooling
                if (pool_info.pool_type == PoolingType::L2)
                {
                    data.val[0] = vmulq_f32(data.val[0], data.val[0]);
                    data.val[1] = vmulq_f32(data.val[1], data.val[1]);
                }
                float32x4_t sum_data = vaddq_f32(data.val[0], vsetq_lane_f32(0.f, data.val[1], 3));
                for (int i = 1; i < pool_size; ++i)
                {
                    in_ptr = reinterpret_cast<const float *>(src_ptrs[i] + in.offset());

                    x_val = id.x() * pool_stride_x;
                    y_val = (id.y() * pool_stride_y) + i;
                    data  = read_8_boundary_aware(src_h, src_w, pool_pad_left, pool_pad_top, x_val, y_val, in_ptr,
                                                  fill_value);
                    // Get power of 2 in case of l2 pooling
                    if (pool_info.pool_type == PoolingType::L2)
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
                for (int i = 1; i < pool_size; ++i)
                {
                    in_ptr = reinterpret_cast<const float *>(src_ptrs[i] + in.offset());

                    x_val              = id.x() * pool_stride_x;
                    y_val              = (id.y() * pool_stride_y) + i;
                    float32x4x2_t temp = read_8_boundary_aware(src_h, src_w, pool_pad_left, pool_pad_top, x_val, y_val,
                                                               in_ptr, fill_value);
                    data               = vmax2q_f32(data, temp);
                }
                res = vpmax_f32(vget_high_f32(vsetq_lane_f32(min_value, data.val[1], 3)), vget_low_f32(data.val[1]));
                res = vpmax_f32(res, vpmax_f32(vget_high_f32(data.val[0]), vget_low_f32(data.val[0])));
                res = vpmax_f32(res, res);
            }
            final_res = vget_lane_f32(res, 0);

            // Calculate square-root in case of l2 pooling
            if (pool_info.pool_type == PoolingType::L2)
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
