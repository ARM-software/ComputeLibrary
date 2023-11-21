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
#ifndef ACL_SRC_CPU_KERNELS_DIRECTCONV2D_LIST_H
#define ACL_SRC_CPU_KERNELS_DIRECTCONV2D_LIST_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"

#include "src/core/common/Registrars.h"

#include <algorithm>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
#define DECLARE_DIRECT_CONV2D_KERNEL(func_name)                                                    \
    void func_name(const Window &window, const ITensor *src, const ITensor *weights, ITensor *dst, \
                   const PadStrideInfo &conv_info)

DECLARE_DIRECT_CONV2D_KERNEL(neon_fp32_nhwc_directconv2d);
DECLARE_DIRECT_CONV2D_KERNEL(neon_fp16_nchw_directconv2d);
DECLARE_DIRECT_CONV2D_KERNEL(neon_fp32_nchw_directconv2d);

#define DECLARE_IM2COL_KERNEL(func_name)                                                                 \
    void func_name(const ITensor *src, ITensor *dst, const Window &window, DataLayout data_layout,       \
                   const PadStrideInfo &conv_info, std::pair<unsigned int, unsigned int> convolved_dims, \
                   const Size2D &kernel_dims, const Size2D &dilation, uint32_t input_pad_right, bool has_bias)

DECLARE_IM2COL_KERNEL(run_im2col_fp32_nchw_pad);
DECLARE_IM2COL_KERNEL(run_im2col_fp32_nchw_nopad);
DECLARE_IM2COL_KERNEL(run_im2col_fp16_nchw_pad);
DECLARE_IM2COL_KERNEL(run_im2col_fp16_nchw_nopad);
DECLARE_IM2COL_KERNEL(run_im2col_bf16_nchw_pad);
DECLARE_IM2COL_KERNEL(run_im2col_bf16_nchw_nopad);
DECLARE_IM2COL_KERNEL(run_im2col_qasymm8_nchw_pad);
DECLARE_IM2COL_KERNEL(run_im2col_qasymm8_nchw_nopad);

DECLARE_IM2COL_KERNEL(run_im2col_fp32_pad);
DECLARE_IM2COL_KERNEL(run_im2col_fp32_nopad);
DECLARE_IM2COL_KERNEL(run_im2col_fp16_pad);
DECLARE_IM2COL_KERNEL(run_im2col_fp16_nopad);
DECLARE_IM2COL_KERNEL(run_im2col_bf16_pad);
DECLARE_IM2COL_KERNEL(run_im2col_bf16_nopad);

#undef DECLARE_DIRECT_CONV2D_KERNEL
#undef DECLARE_IM2COL_KERNEL

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_DIRECTCONV2D_LIST_H
