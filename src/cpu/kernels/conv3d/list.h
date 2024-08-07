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
#ifndef ACL_SRC_CPU_KERNELS_CONV3D_LIST_H
#define ACL_SRC_CPU_KERNELS_CONV3D_LIST_H

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

#define DECLARE_CONV3D_KERNEL(func_name)                                                        \
    void func_name(const ITensor *src0, const ITensor *src1, const ITensor *src2, ITensor *dst, \
                   const Conv3dInfo &conv_info, const Window &window)

DECLARE_CONV3D_KERNEL(directconv3d_fp16_neon_ndhwc);
DECLARE_CONV3D_KERNEL(directconv3d_fp32_neon_ndhwc);
DECLARE_CONV3D_KERNEL(directconv3d_qu8_neon_ndhwc);
DECLARE_CONV3D_KERNEL(directconv3d_qs8_neon_ndhwc);
#undef DECLARE_CONV3D_KERNEL

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_CONV3D_LIST_H
