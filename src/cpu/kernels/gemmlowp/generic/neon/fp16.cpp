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
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "src/cpu/kernels/gemmlowp/generic/neon/impl.h"

namespace arm_compute
{
template <>
inline void add_contribution_boffset_store<float16_t>(const float16_t b_offset_term_scalar, float16_t *mm_result_ptr)
{
    const float16x8_t b_offset_term_f16_vec = vdupq_n_f16(b_offset_term_scalar);
    float16x8x2_t     in_f16                = {{vld1q_f16(mm_result_ptr + 0), vld1q_f16(mm_result_ptr + 8)}};
    // Add the offset terms to GEMM's result
    in_f16.val[0] = vaddq_f16(in_f16.val[0], b_offset_term_f16_vec);
    in_f16.val[1] = vaddq_f16(in_f16.val[1], b_offset_term_f16_vec);
    // Store the result with the offset contribution
    vst1q_f16(mm_result_ptr + 0, in_f16.val[0]);
    vst1q_f16(mm_result_ptr + 8, in_f16.val[1]);
}

template <>
inline void convert_scale_store<float16_t>(int32x4x4_t &offset_term_s32, float16_t scale, float16_t *mm_result_ptr)
{
    float16x8x2_t in_f16 = {{vld1q_f16(mm_result_ptr + 0), vld1q_f16(mm_result_ptr + 8)}};

    const float16x8x2_t offset_term_f16 = {{vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(offset_term_s32.val[0])),
                                                         vcvt_f16_f32(vcvtq_f32_s32(offset_term_s32.val[1]))),
                                            vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(offset_term_s32.val[2])),
                                                         vcvt_f16_f32(vcvtq_f32_s32(offset_term_s32.val[3])))}};

    // Convert and scale the S32 offsets to match the already scaled GEMM results
    float16x8x2_t offset_terms_scaled = {{
        vmulq_n_f16(offset_term_f16.val[0], scale),
        vmulq_n_f16(offset_term_f16.val[1], scale),
    }};

    // Add the offset terms to the GEMM result
    in_f16.val[0] = vaddq_f16(in_f16.val[0], offset_terms_scaled.val[0]);
    in_f16.val[1] = vaddq_f16(in_f16.val[1], offset_terms_scaled.val[1]);
    // Store the result with the offset contribution
    vst1q_f16(mm_result_ptr + 0, in_f16.val[0]);
    vst1q_f16(mm_result_ptr + 8, in_f16.val[1]);
}

namespace cpu
{

void neon_run_offset_contribution_fp16(const Window  &window,
                                       ITensor       *mm_result,
                                       const ITensor *vector_sum_col,
                                       const ITensor *vector_sum_row,
                                       int32_t        a_offset,
                                       int32_t        b_offset,
                                       int32_t        k_offset,
                                       float          scale,
                                       bool           slide_vector_sum_col,
                                       bool           is_gemm3d)
{
    neon_run_offset_contribution_float<float16_t>(window, mm_result, vector_sum_col, vector_sum_row, a_offset, b_offset,
                                                  k_offset, scale, slide_vector_sum_col, is_gemm3d);
}

} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
