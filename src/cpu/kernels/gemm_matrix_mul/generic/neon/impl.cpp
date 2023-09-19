/*
 * Copyright (c) 2017-2023 Arm Limited.
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

#include "src/cpu/kernels/gemm_matrix_mul/generic/neon/impl.h"
#include "src/core/utils/helpers/float_ops.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
void vector_matrix_multiply_f32(const ITensor *lhs, const ITensor *rhs, ITensor *dst, const Window &window, const ThreadInfo &info, float alpha)
{
    const auto width_matrix_b  = static_cast<int>(dst->info()->dimension(0));
    const auto in_b_stride     = static_cast<int>(rhs->info()->strides_in_bytes()[1] / data_size_from_type(rhs->info()->data_type()));
    const auto num_elems_vec_a = static_cast<int>(lhs->info()->dimension(0));

    // The implementation computes 16 elements per iteration
    const int window_start_x = 16 * info.thread_id;
    const int window_step_x  = 16 * info.num_threads;
    // Make sure (window_end_x - window_start_x) is a multiple of window_step_x
    const int window_end_x = ceil_to_multiple(width_matrix_b - window_start_x, window_step_x) + window_start_x;

    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(0, 1, 1));
    win_out.set(Window::DimY, Window::Dimension(0, 1, 1));

    Window win_a(window);
    win_a.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_a.set(Window::DimY, Window::Dimension(0, 0, 0));

    Window win_b;
    // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
    // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
    if(rhs->info()->num_dimensions() >= 3)
    {
        win_b = window;
    }
    win_b.set(Window::DimX, Window::Dimension(0, 1, 1));
    win_b.set(Window::DimY, Window::Dimension(0, 1, 1));

    Iterator ina(lhs, win_a);
    Iterator inb(rhs, win_b);
    Iterator out(dst, win_out);

    const bool multiply_alpha = !(helpers::float_ops::is_one(alpha));

    const float32x4_t alpha_f32 = vdupq_n_f32(alpha);

    execute_window_loop(win_out, [&](const Coordinates &)
    {
        int x = window_start_x;
        // Here we don't check for x lower equal than (window_end_x - window_step_x) because of
        // window_end_x is computed above which may cause out-of-bound writes to the dst.
        for(; x < (window_end_x - window_step_x); x += window_step_x)
        {
            if(x > width_matrix_b)
            {
                return;
            }

            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);

            auto vec_a    = reinterpret_cast<const float *>(ina.ptr());
            auto matrix_b = reinterpret_cast<const float *>(inb.ptr()) + x;

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(vec_a)));
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b)));
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + in_b_stride)));
#endif /* __arm__ */

            auto vec_a_end_addr = vec_a + num_elems_vec_a;
            for(; vec_a <= (vec_a_end_addr - 4);)
            {
                float32x2_t a0l = vld1_f32(vec_a);

                float32x4_t b00 = vld1q_f32(matrix_b + 0 + 0 * in_b_stride);
                float32x4_t b01 = vld1q_f32(matrix_b + 4 + 0 * in_b_stride);
                float32x4_t b02 = vld1q_f32(matrix_b + 8 + 0 * in_b_stride);
                float32x4_t b03 = vld1q_f32(matrix_b + 12 + 0 * in_b_stride);

                float32x4_t b10 = vld1q_f32(matrix_b + 0 + 1 * in_b_stride);
                float32x4_t b11 = vld1q_f32(matrix_b + 4 + 1 * in_b_stride);
                float32x4_t b12 = vld1q_f32(matrix_b + 8 + 1 * in_b_stride);
                float32x4_t b13 = vld1q_f32(matrix_b + 12 + 1 * in_b_stride);

#if __arm__
                asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(vec_a)));
                asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 1 * in_b_stride)));
                asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 2 * in_b_stride)));
                asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 3 * in_b_stride)));
                asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 4 * in_b_stride)));
#endif /* __arm__ */

                acc0 = vmlaq_lane_f32(acc0, b00, a0l, 0);
                acc1 = vmlaq_lane_f32(acc1, b01, a0l, 0);
                acc2 = vmlaq_lane_f32(acc2, b02, a0l, 0);
                acc3 = vmlaq_lane_f32(acc3, b03, a0l, 0);

                acc0 = vmlaq_lane_f32(acc0, b10, a0l, 1);
                acc1 = vmlaq_lane_f32(acc1, b11, a0l, 1);
                acc2 = vmlaq_lane_f32(acc2, b12, a0l, 1);
                acc3 = vmlaq_lane_f32(acc3, b13, a0l, 1);

                vec_a += 2;
                matrix_b += 2 * in_b_stride;

                a0l = vld1_f32(vec_a);

                b00 = vld1q_f32(matrix_b + 0 + 0 * in_b_stride);
                b01 = vld1q_f32(matrix_b + 4 + 0 * in_b_stride);
                b02 = vld1q_f32(matrix_b + 8 + 0 * in_b_stride);
                b03 = vld1q_f32(matrix_b + 12 + 0 * in_b_stride);

                b10 = vld1q_f32(matrix_b + 0 + 1 * in_b_stride);
                b11 = vld1q_f32(matrix_b + 4 + 1 * in_b_stride);
                b12 = vld1q_f32(matrix_b + 8 + 1 * in_b_stride);
                b13 = vld1q_f32(matrix_b + 12 + 1 * in_b_stride);

                acc0 = vmlaq_lane_f32(acc0, b00, a0l, 0);
                acc1 = vmlaq_lane_f32(acc1, b01, a0l, 0);
                acc2 = vmlaq_lane_f32(acc2, b02, a0l, 0);
                acc3 = vmlaq_lane_f32(acc3, b03, a0l, 0);

                acc0 = vmlaq_lane_f32(acc0, b10, a0l, 1);
                acc1 = vmlaq_lane_f32(acc1, b11, a0l, 1);
                acc2 = vmlaq_lane_f32(acc2, b12, a0l, 1);
                acc3 = vmlaq_lane_f32(acc3, b13, a0l, 1);

                vec_a += 2;
                matrix_b += 2 * in_b_stride;
            }

            for(; vec_a < vec_a_end_addr; ++vec_a)
            {
                const float a0 = *vec_a;

                const float32x4_t b00 = vld1q_f32(matrix_b + 0 + 0 * in_b_stride);
                const float32x4_t b01 = vld1q_f32(matrix_b + 4 + 0 * in_b_stride);
                const float32x4_t b02 = vld1q_f32(matrix_b + 8 + 0 * in_b_stride);
                const float32x4_t b03 = vld1q_f32(matrix_b + 12 + 0 * in_b_stride);

                acc0 = vmlaq_n_f32(acc0, b00, a0);
                acc1 = vmlaq_n_f32(acc1, b01, a0);
                acc2 = vmlaq_n_f32(acc2, b02, a0);
                acc3 = vmlaq_n_f32(acc3, b03, a0);

                matrix_b += in_b_stride;
            }

            // Multiply by the weight of matrix product (alpha)
            if(multiply_alpha)
            {
                acc0 = vmulq_f32(acc0, alpha_f32);
                acc1 = vmulq_f32(acc1, alpha_f32);
                acc2 = vmulq_f32(acc2, alpha_f32);
                acc3 = vmulq_f32(acc3, alpha_f32);
            }

            const auto vec_out = reinterpret_cast<float *>(out.ptr()) + x;

            vst1q_f32(vec_out + 0, acc0);
            vst1q_f32(vec_out + 4, acc1);
            vst1q_f32(vec_out + 8, acc2);
            vst1q_f32(vec_out + 12, acc3);
        }

        // Left-over loop
        for(; x < window_end_x; ++x)
        {
            if(x > width_matrix_b)
            {
                return;
            }

            float32x4_t vacc = vdupq_n_f32(0.f);

            auto vec_a    = reinterpret_cast<const float *>(ina.ptr());
            auto matrix_b = reinterpret_cast<const float *>(inb.ptr()) + x;

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(vec_a)));
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b)));
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + in_b_stride)));
#endif /* __arm__ */

            auto vec_a_end_addr = vec_a + num_elems_vec_a;
            for(; vec_a <= (vec_a_end_addr - 4); vec_a += 4)
            {
                const float32x4_t a0l = vld1q_f32(vec_a);

                const float32x4_t b_col =
                {
                    *(matrix_b + 0 * in_b_stride),
                    *(matrix_b + 1 * in_b_stride),
                    *(matrix_b + 2 * in_b_stride),
                    *(matrix_b + 3 * in_b_stride),
                };

#if __arm__
                asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(vec_a)));
                asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 1 * in_b_stride)));
                asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 2 * in_b_stride)));
                asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 3 * in_b_stride)));
                asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 4 * in_b_stride)));
#endif /* __arm__ */

                vacc = vmlaq_f32(vacc, b_col, a0l);

                matrix_b += 4 * in_b_stride;
            }

            float acc = vgetq_lane_f32(vacc, 0) + vgetq_lane_f32(vacc, 1) + vgetq_lane_f32(vacc, 2) + vgetq_lane_f32(vacc, 3);

            for(; vec_a < vec_a_end_addr; ++vec_a)
            {
                const float a0 = *vec_a;

                const float b00 = *matrix_b;

                acc += b00 * a0;

                matrix_b += in_b_stride;
            }

            // Multiply by the weight of matrix product (alpha)
            if(multiply_alpha)
            {
                acc *= alpha;
            }

            const auto vec_out = reinterpret_cast<float *>(out.ptr()) + x;

            *vec_out = acc;
        }
    },
    ina, inb, out);
}

void matrix_matrix_multiply_f32(const ITensor *lhs, const ITensor *rhs, ITensor *dst, const Window &window, const ThreadInfo &info, float alpha)
{
    ARM_COMPUTE_UNUSED(info);
    const int    out_width            = static_cast<int>(dst->info()->dimension(0));
    const int    out_height           = static_cast<int>(dst->info()->dimension(1));
    const size_t in_b_stride          = rhs->info()->strides_in_bytes()[1] / data_size_from_type(rhs->info()->data_type());
    const size_t out_stride1          = dst->info()->strides_in_bytes()[1] / data_size_from_type(dst->info()->data_type());
    const size_t out_stride2          = out_stride1 * 2;
    const size_t out_stride3          = out_stride1 * 3;
    const int    num_elems_matrix_b_x = rhs->info()->dimension(0);

    // Set step_x and step_y for matrix A. Scale by a factor of 4 the Y range as the input interleaved matrix A has 4 times less the rows of the dst matrix
    Window win_a(window);
    win_a.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_a.set(Window::DimY, Window::Dimension(window.y().start() / 4, std::max(window.y().end() / 4, 1), 1));

    Window win_b;
    // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
    // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
    if(rhs->info()->num_dimensions() >= 3)
    {
        win_b = window;
    }
    // Set step_x and step_y for matrix B. Scale by a factor of 4 the X range as the input transposed matrix A has 4 times less the cols of the dst matrix
    // The step along the x direction is 2 times the in_b_stride because for each iteration we compute 2 blocks of size 4x4
    win_b.set(Window::DimX, Window::Dimension(window.x().start() / 4, window.x().end() / 4, 2 * in_b_stride));
    win_b.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator ina(lhs, win_a);
    Iterator inb(rhs, win_b);
    Iterator out(dst, window);

    const bool multiply_alpha = !(helpers::float_ops::is_one(alpha));

    const float32x4_t alpha_f32 = vdupq_n_f32(alpha);

    // The implementation assumes that the matrix A and Matrix B have been reshaped respectively with CpuGemmInterleave4x4 and CpuGemmTranspose1xW
    // The reshaping of the matrices helps to have a cache friendly implementation and helps to avoid the data re-arrangements needed for computing 16x4 elements per iteration
    // All the values needed for computing a single 4x4 block will be read from consecutive memory positions
    execute_window_loop(window, [&](const Coordinates & id)
    {
        auto mtx_a0 = reinterpret_cast<const float *>(ina.ptr());
        auto mtx_b0 = reinterpret_cast<const float *>(inb.ptr());
        auto mtx_b1 = mtx_b0 + in_b_stride;

        float32x4_t acc00 = vdupq_n_f32(0.f);
        float32x4_t acc10 = vdupq_n_f32(0.f);
        float32x4_t acc20 = vdupq_n_f32(0.f);
        float32x4_t acc30 = vdupq_n_f32(0.f);

        float32x4_t acc01 = vdupq_n_f32(0.f);
        float32x4_t acc11 = vdupq_n_f32(0.f);
        float32x4_t acc21 = vdupq_n_f32(0.f);
        float32x4_t acc31 = vdupq_n_f32(0.f);

#if __arm__
        asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
        asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
        asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b1)));
#endif /* __arm__ */

        auto mtx_b0_end_addr = mtx_b0 + num_elems_matrix_b_x;
        for(; mtx_b0 <= (mtx_b0_end_addr - 32);)
        {
            float32x4_t a0 = vld1q_dup_f32(mtx_a0 + 0);
            float32x4_t a1 = vld1q_dup_f32(mtx_a0 + 1);
            float32x4_t a2 = vld1q_dup_f32(mtx_a0 + 2);
            float32x4_t a3 = vld1q_dup_f32(mtx_a0 + 3);

            float32x4_t b00 = vld1q_f32(mtx_b0);
            float32x4_t b10 = vld1q_f32(mtx_b1);
            float32x4_t b01 = vld1q_f32(mtx_b0 + 4);
            float32x4_t b11 = vld1q_f32(mtx_b1 + 4);

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b1)));
#endif /* __arm__ */

            // 4x4 block 0
            acc00 = vmlaq_f32(acc00, b00, a0);
            acc10 = vmlaq_f32(acc10, b00, a1);
            acc20 = vmlaq_f32(acc20, b00, a2);
            acc30 = vmlaq_f32(acc30, b00, a3);

            float32x4_t a4 = vld1q_dup_f32(mtx_a0 + 4);
            float32x4_t a5 = vld1q_dup_f32(mtx_a0 + 5);
            float32x4_t a6 = vld1q_dup_f32(mtx_a0 + 6);
            float32x4_t a7 = vld1q_dup_f32(mtx_a0 + 7);

            // 4x4 block 1
            acc01 = vmlaq_f32(acc01, b10, a0);
            acc11 = vmlaq_f32(acc11, b10, a1);
            acc21 = vmlaq_f32(acc21, b10, a2);
            acc31 = vmlaq_f32(acc31, b10, a3);

            // 4x4 block 0
            acc00 = vmlaq_f32(acc00, b01, a4);
            acc10 = vmlaq_f32(acc10, b01, a5);
            acc20 = vmlaq_f32(acc20, b01, a6);
            acc30 = vmlaq_f32(acc30, b01, a7);

            // 4x4 block 1
            acc01 = vmlaq_f32(acc01, b11, a4);
            acc11 = vmlaq_f32(acc11, b11, a5);
            acc21 = vmlaq_f32(acc21, b11, a6);
            acc31 = vmlaq_f32(acc31, b11, a7);

            mtx_a0 += 8;
            mtx_b0 += 8;
            mtx_b1 += 8;

            a0 = vld1q_dup_f32(mtx_a0 + 0);
            a1 = vld1q_dup_f32(mtx_a0 + 1);
            a2 = vld1q_dup_f32(mtx_a0 + 2);
            a3 = vld1q_dup_f32(mtx_a0 + 3);

            b00 = vld1q_f32(mtx_b0);
            b10 = vld1q_f32(mtx_b1);
            b01 = vld1q_f32(mtx_b0 + 4);
            b11 = vld1q_f32(mtx_b1 + 4);

            // 4x4 block 0
            acc00 = vmlaq_f32(acc00, b00, a0);
            acc10 = vmlaq_f32(acc10, b00, a1);
            acc20 = vmlaq_f32(acc20, b00, a2);
            acc30 = vmlaq_f32(acc30, b00, a3);

            a4 = vld1q_dup_f32(mtx_a0 + 4);
            a5 = vld1q_dup_f32(mtx_a0 + 5);
            a6 = vld1q_dup_f32(mtx_a0 + 6);
            a7 = vld1q_dup_f32(mtx_a0 + 7);

            // 4x4 block 1
            acc01 = vmlaq_f32(acc01, b10, a0);
            acc11 = vmlaq_f32(acc11, b10, a1);
            acc21 = vmlaq_f32(acc21, b10, a2);
            acc31 = vmlaq_f32(acc31, b10, a3);

            // 4x4 block 0
            acc00 = vmlaq_f32(acc00, b01, a4);
            acc10 = vmlaq_f32(acc10, b01, a5);
            acc20 = vmlaq_f32(acc20, b01, a6);
            acc30 = vmlaq_f32(acc30, b01, a7);

            // 4x4 block 1
            acc01 = vmlaq_f32(acc01, b11, a4);
            acc11 = vmlaq_f32(acc11, b11, a5);
            acc21 = vmlaq_f32(acc21, b11, a6);
            acc31 = vmlaq_f32(acc31, b11, a7);

            mtx_a0 += 8;
            mtx_b0 += 8;
            mtx_b1 += 8;

            a0  = vld1q_dup_f32(mtx_a0 + 0);
            a1  = vld1q_dup_f32(mtx_a0 + 1);
            a2  = vld1q_dup_f32(mtx_a0 + 2);
            a3  = vld1q_dup_f32(mtx_a0 + 3);
            b00 = vld1q_f32(mtx_b0);
            b10 = vld1q_f32(mtx_b1);
            b01 = vld1q_f32(mtx_b0 + 4);
            b11 = vld1q_f32(mtx_b1 + 4);

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b1)));
#endif /* __arm__ */

            // 4x4 block 0
            acc00 = vmlaq_f32(acc00, b00, a0);
            acc10 = vmlaq_f32(acc10, b00, a1);
            acc20 = vmlaq_f32(acc20, b00, a2);
            acc30 = vmlaq_f32(acc30, b00, a3);

            a4 = vld1q_dup_f32(mtx_a0 + 4);
            a5 = vld1q_dup_f32(mtx_a0 + 5);
            a6 = vld1q_dup_f32(mtx_a0 + 6);
            a7 = vld1q_dup_f32(mtx_a0 + 7);

            // 4x4 block 1
            acc01 = vmlaq_f32(acc01, b10, a0);
            acc11 = vmlaq_f32(acc11, b10, a1);
            acc21 = vmlaq_f32(acc21, b10, a2);
            acc31 = vmlaq_f32(acc31, b10, a3);

            // 4x4 block 0
            acc00 = vmlaq_f32(acc00, b01, a4);
            acc10 = vmlaq_f32(acc10, b01, a5);
            acc20 = vmlaq_f32(acc20, b01, a6);
            acc30 = vmlaq_f32(acc30, b01, a7);

            // 4x4 block 1
            acc01 = vmlaq_f32(acc01, b11, a4);
            acc11 = vmlaq_f32(acc11, b11, a5);
            acc21 = vmlaq_f32(acc21, b11, a6);
            acc31 = vmlaq_f32(acc31, b11, a7);

            mtx_a0 += 8;
            mtx_b0 += 8;
            mtx_b1 += 8;

            a0  = vld1q_dup_f32(mtx_a0 + 0);
            a1  = vld1q_dup_f32(mtx_a0 + 1);
            a2  = vld1q_dup_f32(mtx_a0 + 2);
            a3  = vld1q_dup_f32(mtx_a0 + 3);
            b00 = vld1q_f32(mtx_b0);
            b10 = vld1q_f32(mtx_b1);
            b01 = vld1q_f32(mtx_b0 + 4);
            b11 = vld1q_f32(mtx_b1 + 4);

            // 4x4 block 0
            acc00 = vmlaq_f32(acc00, b00, a0);
            acc10 = vmlaq_f32(acc10, b00, a1);
            acc20 = vmlaq_f32(acc20, b00, a2);
            acc30 = vmlaq_f32(acc30, b00, a3);

            a4 = vld1q_dup_f32(mtx_a0 + 4);
            a5 = vld1q_dup_f32(mtx_a0 + 5);
            a6 = vld1q_dup_f32(mtx_a0 + 6);
            a7 = vld1q_dup_f32(mtx_a0 + 7);

            // 4x4 block 1
            acc01 = vmlaq_f32(acc01, b10, a0);
            acc11 = vmlaq_f32(acc11, b10, a1);
            acc21 = vmlaq_f32(acc21, b10, a2);
            acc31 = vmlaq_f32(acc31, b10, a3);

            // 4x4 block 0
            acc00 = vmlaq_f32(acc00, b01, a4);
            acc10 = vmlaq_f32(acc10, b01, a5);
            acc20 = vmlaq_f32(acc20, b01, a6);
            acc30 = vmlaq_f32(acc30, b01, a7);

            // 4x4 block 1
            acc01 = vmlaq_f32(acc01, b11, a4);
            acc11 = vmlaq_f32(acc11, b11, a5);
            acc21 = vmlaq_f32(acc21, b11, a6);
            acc31 = vmlaq_f32(acc31, b11, a7);

            mtx_a0 += 8;
            mtx_b0 += 8;
            mtx_b1 += 8;
        }

        for(; mtx_b0 < mtx_b0_end_addr;)
        {
            float32x4_t a0  = vld1q_dup_f32(mtx_a0 + 0);
            float32x4_t a1  = vld1q_dup_f32(mtx_a0 + 1);
            float32x4_t a2  = vld1q_dup_f32(mtx_a0 + 2);
            float32x4_t a3  = vld1q_dup_f32(mtx_a0 + 3);
            float32x4_t b00 = vld1q_f32(mtx_b0);
            float32x4_t b10 = vld1q_f32(mtx_b1);

#if __arm__
            asm volatile("PLD [%0, #128*2]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
            asm volatile("PLD [%0, #128*2]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
            asm volatile("PLD [%0, #128*2]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b1)));
#endif /* __arm__ */
            // 4x4 block 0
            acc00 = vmlaq_f32(acc00, b00, a0);
            acc10 = vmlaq_f32(acc10, b00, a1);
            acc20 = vmlaq_f32(acc20, b00, a2);
            acc30 = vmlaq_f32(acc30, b00, a3);

            // 4x4 block 1
            acc01 = vmlaq_f32(acc01, b10, a0);
            acc11 = vmlaq_f32(acc11, b10, a1);
            acc21 = vmlaq_f32(acc21, b10, a2);
            acc31 = vmlaq_f32(acc31, b10, a3);

            mtx_a0 += 4;
            mtx_b0 += 4;
            mtx_b1 += 4;
        }

        // Multiply by the weight of matrix product (alpha)
        if(multiply_alpha)
        {
            acc00 = vmulq_f32(acc00, alpha_f32);
            acc10 = vmulq_f32(acc10, alpha_f32);
            acc20 = vmulq_f32(acc20, alpha_f32);
            acc30 = vmulq_f32(acc30, alpha_f32);
            acc01 = vmulq_f32(acc01, alpha_f32);
            acc11 = vmulq_f32(acc11, alpha_f32);
            acc21 = vmulq_f32(acc21, alpha_f32);
            acc31 = vmulq_f32(acc31, alpha_f32);
        }

        const auto mtx_out0 = reinterpret_cast<float *>(out.ptr());
        const auto mtx_out1 = mtx_out0 + 4;

        if(id.x() < (out_width - 8))
        {
            vst1q_f32(mtx_out0, acc00);
            vst1q_f32(mtx_out1, acc01);
            if(id.y() + 1 < out_height)
            {
                vst1q_f32(mtx_out0 + out_stride1, acc10);
                vst1q_f32(mtx_out1 + out_stride1, acc11);
                if(id.y() + 2 < out_height)
                {
                    vst1q_f32(mtx_out0 + out_stride2, acc20);
                    vst1q_f32(mtx_out1 + out_stride2, acc21);
                    if(id.y() + 3 < out_height)
                    {
                        vst1q_f32(mtx_out0 + out_stride3, acc30);
                        vst1q_f32(mtx_out1 + out_stride3, acc31);
                    }
                }
            }
        }
        else if(id.x() < (out_width - 4))
        {
            vst1q_f32(mtx_out0, acc00);
            if(id.y() + 1 < out_height)
            {
                vst1q_f32(mtx_out0 + out_stride1, acc10);
                if(id.y() + 2 < out_height)
                {
                    vst1q_f32(mtx_out0 + out_stride2, acc20);
                    if(id.y() + 3 < out_height)
                    {
                        vst1q_f32(mtx_out0 + out_stride3, acc30);
                    }
                }
            }
            // Left-over columns
            const int columns_left = out_width - id.x() - 4;
            for(auto x = 0; x < columns_left; ++x)
            {
                *(mtx_out1 + x) = acc01[x];
                if(id.y() + 1 < out_height)
                {
                    *(mtx_out1 + x + out_stride1) = acc11[x];
                    if(id.y() + 2 < out_height)
                    {
                        *(mtx_out1 + x + out_stride2) = acc21[x];
                        if(id.y() + 3 < out_height)
                        {
                            *(mtx_out1 + x + out_stride3) = acc31[x];
                        }
                    }
                }
            }
        }
        else
        {
            // Left-over columns
            const int columns_left = out_width - id.x();
            for(int x = 0; x < columns_left; ++x)
            {
                *(mtx_out0 + x) = acc00[x];
                if(id.y() + 1 < out_height)
                {
                    *(mtx_out0 + x + out_stride1) = acc10[x];
                    if(id.y() + 2 < out_height)
                    {
                        *(mtx_out0 + x + out_stride2) = acc20[x];
                        if(id.y() + 3 < out_height)
                        {
                            *(mtx_out0 + x + out_stride3) = acc30[x];
                        }
                    }
                }
            }
        }
    },
    ina, inb, out);
}
} // namespace cpu

} // namespace arm_compute
