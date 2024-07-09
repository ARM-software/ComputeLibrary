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

#include "src/cpu/kernels/directconv2d_output_stage/generic/neon/float_impl.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
void output_stage_nhwc_fp32(ITensor       *src,
                            const ITensor *bias,
                            const Window  &window,
                            ITensor       *dst,
                            int            result_fixedpoint_multiplier,
                            int            result_shift,
                            int            result_offset_after_shift)
{
    output_stage_nhwc_fp<float>(src, bias, window, dst, result_fixedpoint_multiplier, result_shift,
                                result_offset_after_shift);
}

void output_stage_nchw_fp32(ITensor       *src,
                            const ITensor *bias,
                            const Window  &window,
                            ITensor       *dst,
                            int            result_fixedpoint_multiplier,
                            int            result_shift,
                            int            result_offset_after_shift)
{
    output_stage_nchw_fp<float>(src, bias, window, dst, result_fixedpoint_multiplier, result_shift,
                                result_offset_after_shift);
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
