/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef SRC_CORE_POOLING_3D_LAYER_IMPL_H
#define SRC_CORE_POOLING_3D_LAYER_IMPL_H

#include "arm_compute/core/Helpers.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include "src/core/helpers/PoolingHelpers.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/pool3d/neon/quantized.h"

namespace arm_compute
{
namespace cpu
{
namespace
{
template <typename T>
void max_poolingMxNxD_fp_neon_ndhwc(const ITensor *src, ITensor *dst0, Pooling3dLayerInfo &pool_info, const Window &window_out,
                                    const int window_start_x, const int window_end_x, const int window_step_x)

{
    using vtype       = wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>;
    using vector_type = typename vtype::type;
    using tag_type    = typename vtype::tag_type;

    int pool_stride_x = static_cast<int>(pool_info.stride.width);
    int pool_stride_y = static_cast<int>(pool_info.stride.height);
    int pool_stride_z = static_cast<int>(pool_info.stride.depth);

    const int pool_size_x = pool_info.is_global_pooling ? src->info()->tensor_shape().y() : pool_info.pool_size.width;
    const int pool_size_y = pool_info.is_global_pooling ? src->info()->tensor_shape().z() : pool_info.pool_size.height;
    const int pool_size_z = pool_info.is_global_pooling ? src->info()->tensor_shape()[3] : pool_info.pool_size.depth;

    const int pool_pad_top   = static_cast<int>(pool_info.padding.top);
    const int pool_pad_left  = static_cast<int>(pool_info.padding.left);
    const int pool_pad_front = static_cast<int>(pool_info.padding.front);

    const int input_dim_w = src->info()->dimension(1);
    const int input_dim_h = src->info()->dimension(2);
    const int input_dim_d = src->info()->dimension(3);

    const int y_stride = static_cast<int>(src->info()->strides_in_bytes().y());
    const int z_stride = static_cast<int>(src->info()->strides_in_bytes().z());
    const int w_stride = static_cast<int>(src->info()->strides_in_bytes()[3]);
    const int n_stride = static_cast<int>(src->info()->strides_in_bytes()[4]);

    const uint8_t *in_ptr_start = src->buffer() + src->info()->offset_first_element_in_bytes();

    Iterator out(dst0, window_out);

    vector_type vres;
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
            vres = wrapper::vdup_n(static_cast<T>(-std::numeric_limits<float>::infinity()), tag_type());
            for(int z = pool_start_z; z < pool_end_z; ++z)
            {
                const uint8_t *in_ptr_z = in_ptr_n + (z + in_idx_depth) * w_stride;
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    const uint8_t *in_ptr_y = in_ptr_z + (y + in_idx_height) * z_stride;
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const uint8_t    *in_ptr_x = in_ptr_y + (x + in_idx_width) * y_stride;
                        const vector_type data     = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr_x) + x_off);
                        vres                       = wrapper::vmax(vres, data);
                    }
                }
            }
            // Store result
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off, vres);
        }

        // Left-overs loop
        for(; x_off < window_end_x; ++x_off)
        {
            T res(0);
            res = -std::numeric_limits<float>::infinity();
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
                        res                     = std::max(res, data);
                    }
                }
            }
            // Store result
            *(reinterpret_cast<T *>(out.ptr()) + x_off) = res;
        }
    },
    out);
}

template <typename T>
void avg_poolingMxNxD_fp_neon_ndhwc(const ITensor *src, ITensor *dst0, Pooling3dLayerInfo &pool_info,
                                    const Window &window_out, const int window_start_x, const int window_end_x, const int window_step_x)
{
    using vtype       = wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>;
    using vector_type = typename vtype::type;
    using tag_type    = typename vtype::tag_type;

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

    const int input_dim_w = src->info()->dimension(1);
    const int input_dim_h = src->info()->dimension(2);
    const int input_dim_d = src->info()->dimension(3);

    const int y_stride = static_cast<int>(src->info()->strides_in_bytes().y());
    const int z_stride = static_cast<int>(src->info()->strides_in_bytes().z());
    const int w_stride = static_cast<int>(src->info()->strides_in_bytes()[3]);
    const int n_stride = static_cast<int>(src->info()->strides_in_bytes()[4]);

    const uint8_t *in_ptr_start = src->buffer() + src->info()->offset_first_element_in_bytes();

    Iterator out(dst0, window_out);

    vector_type vres;
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

        // Calculate scale
        const float scale = calculate_avg_scale_pool3d(pool_info.exclude_padding, id, pool_size_x, pool_size_y, pool_size_z, upper_bound_w, upper_bound_h, upper_bound_d, pool_pad_left,
                                                       pool_pad_top, pool_pad_front, pool_stride_x,
                                                       pool_stride_y, pool_stride_z);
        const vector_type scale_v = wrapper::vdup_n(static_cast<T>(scale), tag_type());

        int x_off = window_start_x;

        for(; x_off <= (window_end_x - window_step_x); x_off += window_step_x) // C
        {
            // Perform pooling
            vres = wrapper::vdup_n(static_cast<T>(0.0f), tag_type());
            for(int z = pool_start_z; z < pool_end_z; ++z)
            {
                const uint8_t *in_ptr_z = in_ptr_n + (z + in_idx_depth) * w_stride;
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    const uint8_t *in_ptr_y = in_ptr_z + (y + in_idx_height) * z_stride;
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const uint8_t    *in_ptr_x = in_ptr_y + (x + in_idx_width) * y_stride;
                        const vector_type data     = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr_x) + x_off);
                        vres                       = wrapper::vadd(vres, data);
                    }
                }
            }

            // Divide by scale
            vres = wrapper::vmul(vres, scale_v);

            // Store result
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off, vres);
        }

        // Left-overs loop
        for(; x_off < window_end_x; ++x_off)
        {
            T res(0);

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

            // Divide by scale
            res *= scale;

            // Store result
            *(reinterpret_cast<T *>(out.ptr()) + x_off) = res;
        }
    },
    out);
}

template <typename T>
void l2_poolingMxNxD_fp_neon_ndhwc(const ITensor *src, ITensor *dst0, Pooling3dLayerInfo &pool_info,
                                   const Window &window_out, const int window_start_x, const int window_end_x, const int window_step_x)
{
    using vtype       = wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>;
    using vector_type = typename vtype::type;
    using tag_type    = typename vtype::tag_type;

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

    const int input_dim_w = src->info()->dimension(1);
    const int input_dim_h = src->info()->dimension(2);
    const int input_dim_d = src->info()->dimension(3);

    const int y_stride = static_cast<int>(src->info()->strides_in_bytes().y());
    const int z_stride = static_cast<int>(src->info()->strides_in_bytes().z());
    const int w_stride = static_cast<int>(src->info()->strides_in_bytes()[3]);
    const int n_stride = static_cast<int>(src->info()->strides_in_bytes()[4]);

    const uint8_t *in_ptr_start = src->buffer() + src->info()->offset_first_element_in_bytes();

    Iterator out(dst0, window_out);

    vector_type vres;
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

        // Calculate scale
        const float scale = calculate_avg_scale_pool3d(pool_info.exclude_padding, id, pool_size_x, pool_size_y, pool_size_z, upper_bound_w, upper_bound_h, upper_bound_d, pool_pad_left,
                                                       pool_pad_top, pool_pad_front, pool_stride_x,
                                                       pool_stride_y, pool_stride_z);

        int x_off = window_start_x;

        for(; x_off <= (window_end_x - window_step_x); x_off += window_step_x) // C
        {
            // Perform pooling
            vres = wrapper::vdup_n(static_cast<T>(0.0f), tag_type());
            for(int z = pool_start_z; z < pool_end_z; ++z)
            {
                const uint8_t *in_ptr_z = in_ptr_n + (z + in_idx_depth) * w_stride;
                for(int y = pool_start_y; y < pool_end_y; ++y)
                {
                    const uint8_t *in_ptr_y = in_ptr_z + (y + in_idx_height) * z_stride;
                    for(int x = pool_start_x; x < pool_end_x; ++x)
                    {
                        const uint8_t    *in_ptr_x = in_ptr_y + (x + in_idx_width) * y_stride;
                        const vector_type data     = wrapper::vloadq(reinterpret_cast<const T *>(in_ptr_x) + x_off);
                        vres                       = wrapper::vmla(vres, data, data);
                    }
                }
            }

            const vector_type scale_v = wrapper::vdup_n(static_cast<T>(scale), tag_type());

            // Divide by scale
            vres = wrapper::vmul(vres, scale_v);

            // Calculate square-root
            vres = wrapper::vinv(wrapper::vinvsqrt(vres));

            // Store result
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x_off, vres);
        }

        // Left-overs loop
        for(; x_off < window_end_x; ++x_off)
        {
            T res(0);

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
                        res += data * data;
                    }
                }
            }

            // Divide by scale
            res *= scale;

            // Square root
            res = std::sqrt(res);

            // Store result
            *(reinterpret_cast<T *>(out.ptr()) + x_off) = res;
        }
    },
    out);
}
} // namespace

template <typename T>
void poolingMxNxD_fp_neon_ndhwc(const ITensor *src, ITensor *dst0, Pooling3dLayerInfo &pool_info, const Window &window)
{
    const int     window_start_x = window.x().start();
    const int     window_end_x   = window.x().end();
    constexpr int window_step_x  = 16 / sizeof(T);
    Window        window_out     = window;

    // Needed to handle loop left-over
    window_out.set(Window::DimX, Window::Dimension(0, 1, 1));

    switch(pool_info.pool_type)
    {
        case PoolingType::MAX:
            max_poolingMxNxD_fp_neon_ndhwc<T>(src, dst0, pool_info, window_out, window_start_x, window_end_x, window_step_x);
            break;
        case PoolingType::AVG:
            avg_poolingMxNxD_fp_neon_ndhwc<T>(src, dst0, pool_info, window_out, window_start_x, window_end_x, window_step_x);
            break;
        case PoolingType::L2:
            l2_poolingMxNxD_fp_neon_ndhwc<T>(src, dst0, pool_info, window_out, window_start_x, window_end_x, window_step_x);
            break;
        default:
            ARM_COMPUTE_ERROR("Pool operation not supported");
    }
}

template <typename T>
void poolingMxNxD_q8_neon_ndhwc(const ITensor *src, ITensor *dst0, Pooling3dLayerInfo &pool_info, const Window &window)
{
    constexpr int window_step_x = 16;
    Window        window_out    = window;

    // Needed to handle loop left-over
    window_out.set(Window::DimX, Window::Dimension(0, 1, 1));

    switch(pool_info.pool_type)
    {
        case PoolingType::MAX:
            max_poolingMxNxD_q8_neon_ndhwc<T>(src, dst0, pool_info, window_out, window_step_x);
            break;
        case PoolingType::AVG:
            avg_poolingMxNxD_q8_neon_ndhwc<T>(src, dst0, pool_info, window_out, window_step_x);
            break;
        default:
            ARM_COMPUTE_ERROR("Pool operation not supported");
    }
}
} // namespace cpu
} // namespace arm_compute
#endif //define SRC_CORE_POOLING_3D_LAYER_IMPL_H
