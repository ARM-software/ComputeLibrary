/*
 * Copyright (c) 2023 Arm Limited.
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

#ifndef ACL_SRC_CPU_KERNELS_POOL2D_NEON_IMPL_H
#define ACL_SRC_CPU_KERNELS_POOL2D_NEON_IMPL_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include "src/cpu/kernels/pool2d/neon/list.h"

#include <limits>

#ifdef ENABLE_NCHW_KERNELS
namespace arm_compute
{
namespace cpu
{

namespace
{
template <typename T>
auto read_2_boundary_aware_as_f32(int srcw, int srch, int pad_l, int pad_t, int x, int y, const T *ptr, T fval)
{
    T          vec[2];
    const bool row_in_bounds((y >= pad_t) && (y < (srch + pad_t)));
    for (int i = 0; i < 2; i++)
    {
        if (row_in_bounds && (x + i >= pad_l) && (x + i < (srcw + pad_l)))
        {
            vec[i] = *(ptr + i);
        }
        else
        {
            vec[i] = fval;
        }
    }
    float32_t vec_f32[2] = {vec[0], vec[1]};
    return wrapper::vload(vec_f32);
}
} // namespace

template <typename T>
void pooling2_nchw_maxpool_indices(const ITensor    *src,
                                   ITensor          *dst0,
                                   ITensor          *dst1,
                                   PoolingLayerInfo &pool_info,
                                   const Window     &window_src,
                                   const Window     &window)
{
    Iterator  in(src, window_src);
    Iterator  out(dst0, window);
    Iterator  indices(dst1, window);
    const int pool_pad_top                 = pool_info.pad_stride_info.pad_top();
    const int pool_pad_left                = pool_info.pad_stride_info.pad_left();
    int       pool_stride_x                = 0;
    int       pool_stride_y                = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info.stride();
    const int            src_w             = src->info()->dimension(0);
    const int            src_h             = src->info()->dimension(1);
    const uint8_t *const src_top_ptr =
        src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
    const uint8_t *const src_bottom_ptr =
        src->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));
    const int pad_left    = src->info()->padding().left;
    const int pad_right   = src->info()->padding().right;
    const int in_stride_y = static_cast<int>(src->info()->strides_in_bytes().y());
    const T   float_min   = get_initial_min<T>(pool_info.use_inf_as_limit);
    const T   fill_value  = (pool_info.pool_type == PoolingType::MAX) ? float_min : 0.f;

    execute_window_loop(
        window,
        [&](const Coordinates &id)
        {
            const auto x_val   = id.x() * pool_stride_x;
            const auto y_val_0 = id.y() * pool_stride_y;
            const auto y_val_1 = (id.y() * pool_stride_y) + 1;
            auto       top_data =
                read_2_boundary_aware_as_f32(src_w, src_h, pool_pad_left, pool_pad_top, x_val, y_val_0,
                                             reinterpret_cast<const T *>(src_top_ptr + in.offset()), fill_value);
            auto bottom_data =
                read_2_boundary_aware_as_f32(src_w, src_h, pool_pad_left, pool_pad_top, x_val, y_val_1,
                                             reinterpret_cast<const T *>(src_bottom_ptr + in.offset()), fill_value);

            // Calculate max data, compare top first, then bottom, to make sue the first max is recorded.
            const float32x2_t max_data_top      = vpmax_f32(top_data, top_data);
            const float32x2_t max_data_bottom   = vpmax_f32(bottom_data, bottom_data);
            const float32x2_t max_data          = vmax_f32(max_data_top, max_data_bottom);
            *(reinterpret_cast<T *>(out.ptr())) = static_cast<T>(vget_lane_f32(max_data, 0));

            // Calculate max data indice, which will be used in max unpool.
            const uint32_t offset_base =
                offset_no_padding<T>(in.offset(), id, *src->info(), pool_stride_x, pool_stride_y, DataLayout::NCHW);
            const uint32_t   offset_top     = (uint32_t)(offset_base / sizeof(T));
            const uint32_t   offset_bottom  = offset_top + in_stride_y / sizeof(T) - pad_right - pad_left;
            const uint32x2_t voffset_top    = {offset_top, offset_top + 1u};
            const uint32x2_t voffset_bottom = {offset_bottom, offset_bottom + 1u};
            const uint32x2_t tmp_indices_top =
                vbsl_u32(vcge_f32(top_data, vrev64_f32(top_data)), voffset_top, vrev64_u32(voffset_top));
            const uint32x2_t tmp_indices_bottom =
                vbsl_u32(vcge_f32(bottom_data, vrev64_f32(bottom_data)), voffset_bottom, vrev64_u32(voffset_bottom));
            *(reinterpret_cast<int *>(indices.ptr())) = vget_lane_u32(
                vbsl_u32(vcge_f32(max_data_top, max_data_bottom), tmp_indices_top, tmp_indices_bottom), 0);
        },
        in, out, indices);
}

} // namespace cpu
} // namespace arm_compute

#endif // ENABLE_NCHW_KERNELS

#endif // ACL_SRC_CPU_KERNELS_POOL2D_NEON_IMPL_H
