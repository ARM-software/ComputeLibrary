/*
 * Copyright (c) 2017-2022,2024-2025 Arm Limited.
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

#include "src/cpu/kernels/gemmlowp/generic/neon/impl.h"

namespace arm_compute
{
template <>
inline void convert_scale_store<float>(int32x4x4_t &offset_term_s32, float scale, float *mm_result_ptr)
{
    float32x4x4_t in_f32 = {{vld1q_f32(mm_result_ptr + 0), vld1q_f32(mm_result_ptr + 4), vld1q_f32(mm_result_ptr + 8),
                             vld1q_f32(mm_result_ptr + 12)}};

    // Convert and scale the S32 offsets to match the already scaled GEMM results
    float32x4x4_t offset_terms_scaled = {{
        vmulq_n_f32(vcvtq_f32_s32(offset_term_s32.val[0]), scale),
        vmulq_n_f32(vcvtq_f32_s32(offset_term_s32.val[1]), scale),
        vmulq_n_f32(vcvtq_f32_s32(offset_term_s32.val[2]), scale),
        vmulq_n_f32(vcvtq_f32_s32(offset_term_s32.val[3]), scale),
    }};

    // Add the offset terms to the GEMM result
    in_f32.val[0] = vaddq_f32(in_f32.val[0], offset_terms_scaled.val[0]);
    in_f32.val[1] = vaddq_f32(in_f32.val[1], offset_terms_scaled.val[1]);
    in_f32.val[2] = vaddq_f32(in_f32.val[2], offset_terms_scaled.val[2]);
    in_f32.val[3] = vaddq_f32(in_f32.val[3], offset_terms_scaled.val[3]);

    // Store the result with the offset contribution
    vst1q_f32(mm_result_ptr + 0, in_f32.val[0]);
    vst1q_f32(mm_result_ptr + 4, in_f32.val[1]);
    vst1q_f32(mm_result_ptr + 8, in_f32.val[2]);
    vst1q_f32(mm_result_ptr + 12, in_f32.val[3]);
}

template <>
inline void add_contribution_boffset_store<float>(const float b_offset_term_scalar, float *mm_result_ptr)
{
    const float32x4_t b_offset_term_f32_vec = vdupq_n_f32(b_offset_term_scalar);

    float32x4x4_t in_f32 = {{vld1q_f32(mm_result_ptr + 0), vld1q_f32(mm_result_ptr + 4), vld1q_f32(mm_result_ptr + 8),
                             vld1q_f32(mm_result_ptr + 12)}};

    // Add the offset terms to GEMM's result
    in_f32.val[0] = vaddq_f32(in_f32.val[0], b_offset_term_f32_vec);
    in_f32.val[1] = vaddq_f32(in_f32.val[1], b_offset_term_f32_vec);
    in_f32.val[2] = vaddq_f32(in_f32.val[2], b_offset_term_f32_vec);
    in_f32.val[3] = vaddq_f32(in_f32.val[3], b_offset_term_f32_vec);

    // Store the result with the offset contribution
    vst1q_f32(mm_result_ptr + 0, in_f32.val[0]);
    vst1q_f32(mm_result_ptr + 4, in_f32.val[1]);
    vst1q_f32(mm_result_ptr + 8, in_f32.val[2]);
    vst1q_f32(mm_result_ptr + 12, in_f32.val[3]);
}

namespace cpu
{

void neon_run_offset_contribution_fp32(const Window  &window,
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
    neon_run_offset_contribution_float<float>(window, mm_result, vector_sum_col, vector_sum_row, a_offset, b_offset,
                                              k_offset, scale, slide_vector_sum_col, is_gemm3d);
}

} // namespace cpu
} // namespace arm_compute
