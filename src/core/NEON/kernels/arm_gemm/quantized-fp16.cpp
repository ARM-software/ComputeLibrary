/*
 * Copyright (c) 2025 Arm Limited.
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
#if defined(__aarch64__) && (defined(FP16_KERNELS) || defined(ARM_COMPUTE_ENABLE_FP16))


#include "arm_gemm.hpp"
#include "utils.hpp"

#include <arm_neon.h>

namespace arm_gemm {

// FP16 dequantize
void dequantize_block_32(const DequantizeFloat &qp, unsigned int width, unsigned int height,
                         const int32_t * in_ptr, unsigned int in_stride, __fp16 *out_ptr, unsigned int out_stride,
                         const __fp16 * bias_ptr, bool not_first_pass, const Activation &act)
{
    const float32x4_t vscale = vdupq_n_f32(qp.scale);
    float maxval = std::numeric_limits<float>::infinity();
    float minval = -std::numeric_limits<float>::infinity();

    switch(act.type) {
        default:
        case Activation::Type::None:
            break;
        case Activation::Type::BoundedReLU:
            maxval = static_cast<float>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            minval = 0;
            break;
    }

    const float16x8_t vmin = vdupq_n_f16(static_cast<__fp16>(minval));
    const float16x8_t vmax = vdupq_n_f16(static_cast<__fp16>(maxval));

    for(unsigned int row=0; row<height; row++) {
        unsigned int col=0;
        if (width >= 8) {
            for(; col <= (width-8); col+=8) {
                const int32x4_t vin0 = vld1q_s32(in_ptr + col + (row * in_stride));
                const int32x4_t vin1 = vld1q_s32(in_ptr + col + 4 + (row * in_stride));
                float32x4_t vdeq0 = vmulq_f32(vcvtq_f32_s32(vin0), vscale);
                float32x4_t vdeq1 = vmulq_f32(vcvtq_f32_s32(vin1), vscale);

                float16x8_t vdeq16 = vcombine_f16(vcvt_f16_f32(vdeq0), vcvt_f16_f32(vdeq1));

                if(bias_ptr) {
                    const float16x8_t bin = vld1q_f16(bias_ptr + col);
                    vdeq16 = vaddq_f16(vdeq16, bin);
                }
                if(not_first_pass) {
                    vdeq16 = vaddq_f16(vdeq16, vld1q_f16(out_ptr + col + (row * out_stride)));
                }
                vdeq16 = vminq_f16(vmaxq_f16(vdeq16, vmin), vmax);
                vst1q_f16(out_ptr + col + (row * out_stride), vdeq16);
            }
        }
        // left-over elements
        for(; col < width; ++col) {
            const int32_t val = *(in_ptr + (row * in_stride) + col);
            float res = static_cast<float>(val * qp.scale);
            if(bias_ptr) {
                res += static_cast<float>(*(bias_ptr + col));
            }
            if(not_first_pass) {
                res += *(out_ptr + (row * out_stride) + col);
            }
            res = std::min(std::max(res, minval), maxval);
            *(out_ptr + (row * out_stride) + col) = static_cast<__fp16>(res);
        }
    }
}

} // namespace arm_gemm

#endif // defined(__aarch64__) && (defined(FP16_KERNELS) || defined(ARM_COMPUTE_ENABLE_FP16))

