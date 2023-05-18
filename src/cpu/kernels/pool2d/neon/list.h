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
#ifndef SRC_CORE_NEON_KERNELS_POOLING_LIST_H
#define SRC_CORE_NEON_KERNELS_POOLING_LIST_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/pool2d/neon/quantized.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
#define DECLARE_POOLING_KERNEL(func_name) \
    void func_name(const ITensor *src0, ITensor *dst0, ITensor *dst1, PoolingLayerInfo &, const Window &window_src, const Window &window)

DECLARE_POOLING_KERNEL(poolingMxN_qasymm8_neon_nhwc);
DECLARE_POOLING_KERNEL(poolingMxN_qasymm8_signed_neon_nhwc);
DECLARE_POOLING_KERNEL(poolingMxN_fp16_neon_nhwc);
DECLARE_POOLING_KERNEL(poolingMxN_fp32_neon_nhwc);

#if defined(ENABLE_NCHW_KERNELS)

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
DECLARE_POOLING_KERNEL(pooling2_fp16_neon_nchw);
DECLARE_POOLING_KERNEL(pooling3_fp16_neon_nchw);
DECLARE_POOLING_KERNEL(poolingMxN_fp16_neon_nchw);
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */

DECLARE_POOLING_KERNEL(pooling2_fp32_neon_nchw);
DECLARE_POOLING_KERNEL(pooling3_fp32_neon_nchw);
DECLARE_POOLING_KERNEL(pooling7_fp32_neon_nchw);
DECLARE_POOLING_KERNEL(poolingMxN_fp32_neon_nchw);
#endif /* defined(ENABLE_NCHW_KERNELS) */

#undef DECLARE_POOLING_KERNEL

template <typename T>
T get_initial_min(bool use_inf_as_limit)
{
    return use_inf_as_limit ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
}

template <typename T>
inline uint32_t offset_no_padding(uint32_t padded_offset, const Coordinates &id, const ITensorInfo &info, int pool_stride_x, int pool_stride_y, DataLayout data_layout)
{
    const int pad_left    = info.padding().left;
    const int pad_right   = info.padding().right;
    const int pad_top     = info.padding().top;
    const int pad_bottom  = info.padding().bottom;
    const int in_stride_y = static_cast<int>(info.strides_in_bytes().y());
    const int in_stride_w = static_cast<int>(info.strides_in_bytes()[3]);
    const int pad_horiz   = pad_left + pad_right;
    const int pad_vert    = pad_top + pad_bottom;

    if(data_layout == DataLayout::NCHW)
    {
        const uint32_t offset_base = padded_offset
                                     - sizeof(T) * pad_horiz * id.y() * pool_stride_y                                            /* subtract padding elems per row */
                                     - pad_top * sizeof(T)                                                                       /* top padding */
                                     - sizeof(T) * pad_horiz * info.tensor_shape()[1] * id.z() - pad_vert * in_stride_y * id.z() /* for each Z plane there are height*pad_right padding elems */
                                     - in_stride_w * id[3];

        return offset_base;
    }
    else
    {
        const uint32_t offset_base = padded_offset
                                     - sizeof(T) * pad_horiz * id.y() * pool_stride_x                          // subtract padding elems per row
                                     - pad_top * sizeof(T)                                                     // top padding
                                     - sizeof(T) * pad_horiz * info.tensor_shape()[1] * id.z() * pool_stride_y // for each Z plane there are width*pad_right padding elems
                                     - in_stride_w * id[3];

        return offset_base;
    }
}
} // namespace cpu
} // namespace arm_compute

#endif // SRC_CORE_NEON_KERNELS_POOLING_LIST_H