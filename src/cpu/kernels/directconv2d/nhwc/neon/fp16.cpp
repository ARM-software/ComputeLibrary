/*
 * Copyright (c) 2022-2025 Arm Limited.
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

#include "src/cpu/kernels/directconv2d/impl.h"

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "src/cpu/kernels/directconv2d/nhwc/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

void neon_fp16_nhwc_directconv2d(
    const Window &window, const ITensor *src, const ITensor *weights, ITensor *dst, const PadStrideInfo &conv_info)
{
    convolve_nhwc<float16_t>(window, src, weights, dst, conv_info);
}

template void convolve_nhwc<float16_t>(
    const Window &window, const ITensor *src, const ITensor *weights, ITensor *dst, const PadStrideInfo &conv_info);

void run_im2col_fp16_pad(const ITensor                        *src,
                         ITensor                              *dst,
                         const Window                         &window,
                         DataLayout                            data_layout,
                         const PadStrideInfo                  &conv_info,
                         std::pair<unsigned int, unsigned int> convolved_dims,
                         const Size2D                         &kernel_dims,
                         const Size2D                         &dilation,
                         uint32_t                              input_pad_right,
                         bool                                  has_bias)
{
    arm_compute::cpu::kernels::run_im2col<float16_t, true, false>(
        src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right, has_bias);
}

void run_im2col_fp16_nopad(const ITensor                        *src,
                           ITensor                              *dst,
                           const Window                         &window,
                           DataLayout                            data_layout,
                           const PadStrideInfo                  &conv_info,
                           std::pair<unsigned int, unsigned int> convolved_dims,
                           const Size2D                         &kernel_dims,
                           const Size2D                         &dilation,
                           uint32_t                              input_pad_right,
                           bool                                  has_bias)
{
    arm_compute::cpu::kernels::run_im2col<float16_t, false, false>(
        src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right, has_bias);
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
