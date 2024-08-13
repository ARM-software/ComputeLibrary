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

#ifndef ACL_SRC_CPU_KERNELS_DIRECTCONV2D_OUTPUT_STAGE_LIST_H
#define ACL_SRC_CPU_KERNELS_DIRECTCONV2D_OUTPUT_STAGE_LIST_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

#define DECLARE_DIRECTCONV2D_OUTPUT_STAGE_KERNEL(func_name)                               \
    void func_name(ITensor *src, const ITensor *bias, const Window &window, ITensor *dst, \
                   int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift)

DECLARE_DIRECTCONV2D_OUTPUT_STAGE_KERNEL(output_stage_nhwc_fp32);
DECLARE_DIRECTCONV2D_OUTPUT_STAGE_KERNEL(output_stage_nhwc_fp16);
DECLARE_DIRECTCONV2D_OUTPUT_STAGE_KERNEL(output_stage_nchw_fp32);
DECLARE_DIRECTCONV2D_OUTPUT_STAGE_KERNEL(output_stage_nchw_fp16);
DECLARE_DIRECTCONV2D_OUTPUT_STAGE_KERNEL(output_stage_nhwc_qs8);
DECLARE_DIRECTCONV2D_OUTPUT_STAGE_KERNEL(output_stage_nhwc_qu8);
DECLARE_DIRECTCONV2D_OUTPUT_STAGE_KERNEL(output_stage_nchw_qs8);
DECLARE_DIRECTCONV2D_OUTPUT_STAGE_KERNEL(output_stage_nchw_qu8);

#undef DECLARE_DIRECTCONV2D_OUTPUT_STAGE_KERNEL

} // namespace kernels
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_DIRECTCONV2D_OUTPUT_STAGE_LIST_H
