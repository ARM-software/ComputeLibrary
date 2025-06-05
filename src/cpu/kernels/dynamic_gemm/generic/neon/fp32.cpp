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
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Steps.h"
#include "arm_compute/core/Window.h"

#include "src/common/utils/Validate.h"
#include "src/core/helpers/WindowHelpers.h"

#if defined(__aarch64__) && defined(ENABLE_FP32_KERNELS)
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#endif // __aarch64__ && ENABLE_FP32_KERNELS
#include "src/common/utils/profile/acl_profile.h"

namespace arm_compute
{
namespace cpu
{

#if defined(__aarch64__) && defined(ENABLE_FP32_KERNELS)
void neon_fp32_dynamic_gemm_pack_rhs(const ITensor *rhs, const ITensor *bias, ITensor *pack_b)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "neon_fp32_dynamic_gemm_pack_rhs");
    const size_t      num_groups  = 1;
    const size_t      n           = rhs->info()->tensor_shape().x();
    const size_t      k           = rhs->info()->tensor_shape().y();
    const size_t      nr          = kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
    const size_t      kr          = kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
    const size_t      sr          = kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
    const size_t      rhs_stride  = rhs->info()->strides_in_bytes().y();
    const void *const rhs_ptr     = rhs->buffer() + rhs->info()->offset_first_element_in_bytes();
    const void *const bias_ptr    = bias->buffer() + bias->info()->offset_first_element_in_bytes();
    const void *const scale       = nullptr;
    void *const       rhs_packed  = pack_b->buffer();
    const size_t      extra_bytes = 0;
    const void *const params      = nullptr;
    kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(num_groups, n, k, nr, kr, sr, rhs_stride, rhs_ptr, bias_ptr, scale,
                                                     rhs_packed, extra_bytes, params);
}

void neon_fp32_dynamic_gemm_run(
    const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, ITensor *pack_b, const Window &window)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "neon_fp32_dynamic_gemm_run");
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);

    //  Full dimensions.
    const size_t M = d->info()->tensor_shape().y();
    const size_t N = d->info()->tensor_shape().x();
    const size_t K = a->info()->tensor_shape().x();

    // Buffers start.
    const uint8_t *const lhs_buf = a->buffer() + a->info()->offset_first_element_in_bytes();
    uint8_t *const       dst_buf = d->buffer() + d->info()->offset_first_element_in_bytes();

    const size_t m_start = window.y().start();
    const size_t m_end   = window.y().end();
    const size_t n_start = window.x().start();

    // As the workload is split in Y dimensions only, each window should start
    // from the beginning of a row.
    ARM_COMPUTE_ASSERT(n_start == 0);

    // The window can be bigger than the size of the matrix.
    const size_t m_len_window = m_end - m_start;
    const size_t m_remainder  = M - m_start;
    const size_t m_len        = std::min(m_len_window, m_remainder);

    // As the workload is split in Y dimensions, LHS is processed in full rows.
    const size_t n_len = N;
    const size_t k_len = K;

    const size_t         lhs_stride = a->info()->strides_in_bytes().y();
    const uint8_t *const lhs =
        lhs_buf + kai_get_lhs_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(m_start, lhs_stride);

    const size_t   dst_stride_row = d->info()->strides_in_bytes().y();
    const size_t   dst_stride_col = d->info()->strides_in_bytes().x();
    uint8_t *const dst            = dst_buf + kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(
                                                  m_start, n_start, dst_stride_row);

    const uint8_t *const rhs_packed = pack_b->buffer();

    const float clamp_min = -std::numeric_limits<float>::max();
    const float clamp_max = std::numeric_limits<float>::max();

    kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(m_len, n_len, k_len, lhs, lhs_stride, rhs_packed, dst,
                                                               dst_stride_row, dst_stride_col, clamp_min, clamp_max);
}

size_t neon_fp32_dynamic_gemm_size_of_packed_rhs(size_t rows, size_t columns)
{
    // The 0.5.0 documentation is wrong. In a kxn matrix, k=rows and n=columns.
    return kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(columns, rows);
}

Window neon_fp32_dynamic_gemm_window(const ITensorInfo *dst)
{
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();

    const Steps steps(n_step, m_step);

    const Window window = calculate_max_window(*dst, steps);

    return window;
}
#endif // __aarch64__ && ENABLE_FP32_KERNELS

} // namespace cpu
} // namespace arm_compute
