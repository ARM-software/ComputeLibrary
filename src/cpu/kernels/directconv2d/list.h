/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_CONV2D_LIST_H
#define SRC_CORE_NEON_KERNELS_CONV2D_LIST_H

#include "src/core/common/Registrars.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
#define DECLARE_DIRECT_CONV2D_KERNEL(func_name) \
    void func_name(const Window &window, const ITensor *src, const ITensor *weights, ITensor *dst, const PadStrideInfo &conv_info)

DECLARE_DIRECT_CONV2D_KERNEL(neon_fp32_nhwc_directconv2d);
DECLARE_DIRECT_CONV2D_KERNEL(neon_fp16_nchw_directconv2d);
DECLARE_DIRECT_CONV2D_KERNEL(neon_fp32_nchw_directconv2d);

#undef DECLARE_DIRECT_CONV2D_KERNEL

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif //SRC_CORE_NEON_KERNELS_CONV2D_LIST_H
