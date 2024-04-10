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

#include "arm_compute/core/Error.h"

#include <cstdint>
#include <cstring>

namespace arm_compute
{
namespace cpu
{

void depth_to_space_nchw_any( //
    const uint8_t  *src,
    uint8_t        *dst,
    const uintptr_t src_shape[4],
    const uintptr_t src_strides[4],
    const uintptr_t dst_strides[4],
    uintptr_t       element_size,
    uintptr_t       block_size)
{
    ARM_COMPUTE_ERROR_ON(src_strides[0] != element_size);
    ARM_COMPUTE_ERROR_ON(dst_strides[0] != element_size);

    const auto dst_channels         = src_shape[2] / (block_size * block_size);
    const auto src_block_col_stride = dst_channels * src_strides[2];
    const auto src_block_row_stride = block_size * dst_channels * src_strides[2];

    auto *src_batch_ptr = src;
    auto *dst_batch_ptr = dst;

    for (uintptr_t batch = 0; batch < src_shape[3]; ++batch)
    {
        auto *src_channel_ptr = src_batch_ptr;
        auto *dst_channel_ptr = dst_batch_ptr;

        for (uintptr_t channel = 0; channel < dst_channels; ++channel)
        {
            auto *src_height_block_ptr = src_channel_ptr;
            auto *dst_row_ptr          = dst_channel_ptr;

            for (uintptr_t height_block = 0; height_block < src_shape[1]; ++height_block)
            {
                auto *src_block_row_ptr = src_height_block_ptr;

                for (uintptr_t block_row = 0; block_row < block_size; ++block_row)
                {
                    auto *src_width_block_ptr = src_block_row_ptr;
                    auto *dst_col_ptr         = dst_row_ptr;

                    for (uintptr_t width_block = 0; width_block < src_shape[0]; ++width_block)
                    {
                        auto *src_block_col_ptr = src_width_block_ptr;

                        for (uintptr_t block_col = 0; block_col < block_size; ++block_col)
                        {
                            // The source pointer is accumulated as:
                            //
                            // src_block_col_ptr =
                            //   src +
                            //   batch * dst_strides[3] +
                            //   (channel + (block_row * block_size + block_col) * dst_channels) * src_strides[2] +
                            //   height_block * src_strides[1] +
                            //   width_block * element_size;
                            //
                            // The destination pointer is accumuated as:
                            //
                            // dst_col_ptr =
                            //   dst +
                            //   batch * dst_strides[3] +
                            //   channel * dst_strides[2] +
                            //   (height_block * block_size + block_row) * dst_strides[1] +
                            //   (width_block * block_size + block_col) * element_size

                            std::memcpy(dst_col_ptr, src_block_col_ptr, element_size);

                            src_block_col_ptr += src_block_col_stride;
                            dst_col_ptr += element_size;
                        }

                        src_width_block_ptr += element_size;
                    }

                    src_block_row_ptr += src_block_row_stride;
                    dst_row_ptr += dst_strides[1];
                }

                src_height_block_ptr += src_strides[1];
            }

            src_channel_ptr += src_strides[2];
            dst_channel_ptr += dst_strides[2];
        }

        src_batch_ptr += src_strides[3];
        dst_batch_ptr += dst_strides[3];
    }
}

} // namespace cpu
} // namespace arm_compute
