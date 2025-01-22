/*
 * Copyright (c) 2024 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_SCATTER_GENERIC_NEON_IMPL_H
#define ACL_SRC_CPU_KERNELS_SCATTER_GENERIC_NEON_IMPL_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/function_info/ScatterInfo.h"

#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/CpuTypes.h"

#include <cstdint>

namespace arm_compute
{
namespace cpu
{
template <arm_compute::ScatterFunction sf, typename ScalarType>
void scatter_neon(
    const ITensor *updates, const ITensor *indices, ITensor *dst, const Window &window, const int data_block_length)
{
    const auto updates_info = updates->info();
    const auto idx_info     = indices->info();
    const auto dst_info     = dst->info();

    const auto indices_strides_y = idx_info->strides_in_bytes()[1];

    const int     dst_dims         = dst_info->num_dimensions();
    constexpr int max_index_length = 5;
    int32_t       dst_shape[max_index_length];
    for (int i = 1; i <= max_index_length; ++i)
    {
        dst_shape[i - 1] = dst_info->tensor_shape()[std::max(dst_dims - i, 0)];
    }

    const int  index_len = idx_info->dimension(0);
    const auto num_dims  = dst_info->num_dimensions();
    const int  ind_dims  = idx_info->num_dimensions();

    const int upt_block_stride = updates_info->strides_in_bytes()[updates_info->num_dimensions() - (ind_dims - 1)];

    const int out_block_stride = dst_info->strides_in_bytes()[num_dims - index_len];

    TensorShape  ind_collapsed = idx_info->tensor_shape().collapsed_from(1);
    const size_t num_indices   = ind_collapsed[1];

    Iterator updates_it(updates, window);
    Iterator dst_it(dst, window);

    constexpr int vec_size         = 16 / sizeof(ScalarType);
    uint8_t      *idx_ptr_raw_base = indices->ptr_to_element(arm_compute::Coordinates(0));

    execute_window_loop(
        window,
        [&](const Coordinates &)
        {
            uint8_t *idx_ptr_raw = idx_ptr_raw_base;
            for (size_t index_element = 0; index_element < num_indices; ++index_element)
            {
                int32_t *idx_ptr = reinterpret_cast<int32_t *>(idx_ptr_raw);

                // Out of bounds check
                bool out_of_bounds = false;
                for (int i = 0; i < index_len; ++i)
                {
                    if (idx_ptr[i] >= dst_shape[i] || idx_ptr[i] < 0)
                    {
                        out_of_bounds = true;
                    }
                }

                idx_ptr_raw += indices_strides_y;

                if (out_of_bounds)
                {
                    continue;
                }

                int32_t index = 0;
                for (int i = 0; i < index_len; ++i)
                {
                    index = index * dst_shape[i] + idx_ptr[i];
                }

                const uint8_t *upt_from_index_ptr = updates_it.ptr() + index_element * upt_block_stride;

                uint8_t *dst_from_index_ptr = dst_it.ptr() + index * out_block_stride;

                int x = 0;
                for (; x <= (data_block_length - vec_size); x += vec_size)
                {
                    const auto update_val_vec =
                        wrapper::vloadq(reinterpret_cast<const ScalarType *>(upt_from_index_ptr) + x);
                    const auto dst_val_vec = wrapper::vloadq(reinterpret_cast<ScalarType *>(dst_from_index_ptr) + x);

                    switch (sf)
                    {
                        case ScatterFunction::Update:
                            wrapper::vstore(reinterpret_cast<ScalarType *>(dst_from_index_ptr) + x, update_val_vec);
                            break;
                        case ScatterFunction::Add:
                            wrapper::vstore(reinterpret_cast<ScalarType *>(dst_from_index_ptr) + x,
                                            wrapper::vadd(dst_val_vec, update_val_vec));
                            break;
                        case ScatterFunction::Sub:
                            wrapper::vstore(reinterpret_cast<ScalarType *>(dst_from_index_ptr) + x,
                                            wrapper::vsub(dst_val_vec, update_val_vec));
                            break;
                        case ScatterFunction::Max:
                            wrapper::vstore(reinterpret_cast<ScalarType *>(dst_from_index_ptr) + x,
                                            wrapper::vmax(dst_val_vec, update_val_vec));
                            break;
                        case ScatterFunction::Min:
                            wrapper::vstore(reinterpret_cast<ScalarType *>(dst_from_index_ptr) + x,
                                            wrapper::vmin(dst_val_vec, update_val_vec));
                            break;
                        default:
                            ARM_COMPUTE_ERROR("Invalid reduction function for scatter.");
                    }
                }

                for (; x < data_block_length; ++x)
                {
                    const ScalarType update_val = *(reinterpret_cast<const ScalarType *>(upt_from_index_ptr) + x);
                    const ScalarType dst_val    = *(reinterpret_cast<ScalarType *>(dst_from_index_ptr) + x);
                    ScalarType       output_val;
                    switch (sf)
                    {
                        case ScatterFunction::Update:
                            output_val = update_val;
                            break;
                        case ScatterFunction::Add:
                            output_val = dst_val + update_val;
                            break;
                        case ScatterFunction::Sub:
                            output_val = dst_val - update_val;
                            break;
                        case ScatterFunction::Max:
                            output_val = std::max(dst_val, update_val);
                            break;
                        case ScatterFunction::Min:
                            output_val = std::min(dst_val, update_val);
                            break;
                        default:
                            ARM_COMPUTE_ERROR("Invalid reduction function for scatter.");
                    }
                    *(reinterpret_cast<ScalarType *>(dst_from_index_ptr) + x) = output_val;
                }
            }
        },
        updates_it, dst_it);
}
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_SCATTER_GENERIC_NEON_IMPL_H
