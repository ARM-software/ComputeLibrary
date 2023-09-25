/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#include "src/core/utils/helpers/float_ops.h"
#include "src/cpu/kernels/gemm_matrix_mul/generic/neon/impl.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
void vector_matrix_multiply_f16(const ITensor *lhs, const ITensor *rhs, ITensor *dst, const Window &window, const ThreadInfo &info, float alpha)
{
    const auto width_matrix_b  = static_cast<int>(dst->info()->dimension(0));
    const auto in_b_stride     = static_cast<int>(rhs->info()->strides_in_bytes()[1] / rhs->info()->element_size());
    const auto num_elems_vec_a = static_cast<int>(lhs->info()->dimension(0));

    // The implementation computes 32 elements per iteration
    const int window_start_x = 32 * info.thread_id;
    const int window_step_x  = 32 * info.num_threads;
    const int window_end_x   = ceil_to_multiple(width_matrix_b - window_start_x, window_step_x) + window_start_x;
    ARM_COMPUTE_ERROR_ON_MSG((window_end_x - window_start_x) % window_step_x, " (window_end_x - window_start_x) must be multiple of window_step_x");

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

    const float16x8_t alpha_f16 = vdupq_n_f16(alpha);

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

            auto matrix_b = reinterpret_cast<const float16_t *>(inb.ptr()) + x;

            float16x8_t acc0 = vdupq_n_f16(0.f);
            float16x8_t acc1 = vdupq_n_f16(0.f);
            float16x8_t acc2 = vdupq_n_f16(0.f);
            float16x8_t acc3 = vdupq_n_f16(0.f);

            auto             vec_a          = reinterpret_cast<const float16_t *>(ina.ptr());
            const float16_t *vec_a_end_addr = vec_a + num_elems_vec_a;
            for(; vec_a <= (vec_a_end_addr - 4);)
            {
                const float16x4_t a0l = vld1_f16(vec_a);

                float16x8_t b00 = vld1q_f16(matrix_b + 0 + 0 * in_b_stride);
                float16x8_t b01 = vld1q_f16(matrix_b + 8 + 0 * in_b_stride);
                float16x8_t b02 = vld1q_f16(matrix_b + 16 + 0 * in_b_stride);
                float16x8_t b03 = vld1q_f16(matrix_b + 24 + 0 * in_b_stride);
                float16x8_t b10 = vld1q_f16(matrix_b + 0 + 1 * in_b_stride);
                float16x8_t b11 = vld1q_f16(matrix_b + 8 + 1 * in_b_stride);
                float16x8_t b12 = vld1q_f16(matrix_b + 16 + 1 * in_b_stride);
                float16x8_t b13 = vld1q_f16(matrix_b + 24 + 1 * in_b_stride);

                acc0 = vaddq_f16(acc0, vmulq_lane_f16(b00, a0l, 0));
                acc1 = vaddq_f16(acc1, vmulq_lane_f16(b01, a0l, 0));
                acc2 = vaddq_f16(acc2, vmulq_lane_f16(b02, a0l, 0));
                acc3 = vaddq_f16(acc3, vmulq_lane_f16(b03, a0l, 0));
                acc0 = vaddq_f16(acc0, vmulq_lane_f16(b10, a0l, 1));
                acc1 = vaddq_f16(acc1, vmulq_lane_f16(b11, a0l, 1));
                acc2 = vaddq_f16(acc2, vmulq_lane_f16(b12, a0l, 1));
                acc3 = vaddq_f16(acc3, vmulq_lane_f16(b13, a0l, 1));

                matrix_b += 2 * in_b_stride;

                b00 = vld1q_f16(matrix_b + 0 + 0 * in_b_stride);
                b01 = vld1q_f16(matrix_b + 8 + 0 * in_b_stride);
                b02 = vld1q_f16(matrix_b + 16 + 0 * in_b_stride);
                b03 = vld1q_f16(matrix_b + 24 + 0 * in_b_stride);
                b10 = vld1q_f16(matrix_b + 0 + 1 * in_b_stride);
                b11 = vld1q_f16(matrix_b + 8 + 1 * in_b_stride);
                b12 = vld1q_f16(matrix_b + 16 + 1 * in_b_stride);
                b13 = vld1q_f16(matrix_b + 24 + 1 * in_b_stride);

                acc0 = vaddq_f16(acc0, vmulq_lane_f16(b00, a0l, 2));
                acc1 = vaddq_f16(acc1, vmulq_lane_f16(b01, a0l, 2));
                acc2 = vaddq_f16(acc2, vmulq_lane_f16(b02, a0l, 2));
                acc3 = vaddq_f16(acc3, vmulq_lane_f16(b03, a0l, 2));
                acc0 = vaddq_f16(acc0, vmulq_lane_f16(b10, a0l, 3));
                acc1 = vaddq_f16(acc1, vmulq_lane_f16(b11, a0l, 3));
                acc2 = vaddq_f16(acc2, vmulq_lane_f16(b12, a0l, 3));
                acc3 = vaddq_f16(acc3, vmulq_lane_f16(b13, a0l, 3));

                vec_a += 4;
                matrix_b += 2 * in_b_stride;
            }

            for(; vec_a < vec_a_end_addr; ++vec_a)
            {
                const float16_t   a0  = *vec_a;
                const float16x8_t b00 = vld1q_f16(matrix_b + 0 + 0 * in_b_stride);
                const float16x8_t b01 = vld1q_f16(matrix_b + 8 + 0 * in_b_stride);
                const float16x8_t b02 = vld1q_f16(matrix_b + 16 + 0 * in_b_stride);
                const float16x8_t b03 = vld1q_f16(matrix_b + 24 + 0 * in_b_stride);

                acc0 = vaddq_f16(acc0, vmulq_n_f16(b00, a0));
                acc1 = vaddq_f16(acc1, vmulq_n_f16(b01, a0));
                acc2 = vaddq_f16(acc2, vmulq_n_f16(b02, a0));
                acc3 = vaddq_f16(acc3, vmulq_n_f16(b03, a0));

                matrix_b += in_b_stride;
            }

            // Multiply by the weight of matrix product (alpha)
            if(multiply_alpha)
            {
                acc0 = vmulq_f16(acc0, alpha_f16);
                acc1 = vmulq_f16(acc1, alpha_f16);
                acc2 = vmulq_f16(acc2, alpha_f16);
                acc3 = vmulq_f16(acc3, alpha_f16);
            }

            auto vec_out = reinterpret_cast<float16_t *>(out.ptr()) + x;

            vst1q_f16(vec_out + 0, acc0);
            vst1q_f16(vec_out + 8, acc1);
            vst1q_f16(vec_out + 16, acc2);
            vst1q_f16(vec_out + 24, acc3);
        }

        for(; x < window_end_x; ++x)
        {
            if(x > width_matrix_b)
            {
                return;
            }

            auto matrix_b = reinterpret_cast<const float16_t *>(inb.ptr()) + x;

            float16x4_t vacc = vdup_n_f16(0.f);

            auto             vec_a          = reinterpret_cast<const float16_t *>(ina.ptr());
            const float16_t *vec_a_end_addr = vec_a + num_elems_vec_a;
            for(; vec_a <= (vec_a_end_addr - 4); vec_a += 4)
            {
                const float16x4_t a0l = vld1_f16(vec_a);

                const float16x4_t b_col =
                {
                    *(matrix_b + 0 * in_b_stride),
                    *(matrix_b + 1 * in_b_stride),
                    *(matrix_b + 2 * in_b_stride),
                    *(matrix_b + 3 * in_b_stride),
                };

                vacc = vadd_f16(vacc, vmul_f16(a0l, b_col));

                matrix_b += 4 * in_b_stride;
            }

            float16_t acc = vget_lane_f16(vacc, 0) + vget_lane_f16(vacc, 1) + vget_lane_f16(vacc, 2) + vget_lane_f16(vacc, 3);

            for(; vec_a < vec_a_end_addr; ++vec_a)
            {
                const float16_t a0  = *vec_a;
                const float16_t b00 = *matrix_b;

                acc += b00 * a0;

                matrix_b += in_b_stride;
            }

            // Multiply by the weight of matrix product (alpha)
            if(multiply_alpha)
            {
                acc *= static_cast<float16_t>(alpha);
            }

            auto vec_out = reinterpret_cast<float16_t *>(out.ptr()) + x;

            *(vec_out) = acc;
        }
    },
    ina, inb, out);
}

void matrix_matrix_multiply_f16(const ITensor *lhs, const ITensor *rhs, ITensor *dst, const Window &window, const ThreadInfo &info, float alpha)
{
    ARM_COMPUTE_UNUSED(info);
    const int    out_width            = static_cast<int>(dst->info()->dimension(0));
    const int    out_height           = static_cast<int>(dst->info()->dimension(1));
    const size_t in_b_stride          = rhs->info()->strides_in_bytes()[1] / data_size_from_type(rhs->info()->data_type());
    const size_t out_stride           = dst->info()->strides_in_bytes()[1] / data_size_from_type(dst->info()->data_type());
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
    // Set step_x and step_y for matrix B. Scale by a factor of 8 the X range as the input transposed matrix A has 8 times less the cols of the dst matrix
    win_b.set(Window::DimX, Window::Dimension(window.x().start() / 8, window.x().end() / 8, in_b_stride));
    win_b.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator ina(lhs, win_a);
    Iterator inb(rhs, win_b);
    Iterator out(dst, window);

    const bool multiply_alpha = !(helpers::float_ops::is_one(alpha));

    const float16x8_t alpha_f16 = vdupq_n_f16(alpha);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto   *mtx_a0  = reinterpret_cast<const float16_t *>(ina.ptr());
        const auto   *mtx_b0  = reinterpret_cast<const float16_t *>(inb.ptr());
        auto         *mtx_out = reinterpret_cast<float16_t *>(out.ptr());
        float16x8x4_t c =
        {
            {
                vdupq_n_f16(0.f),
                vdupq_n_f16(0.f),
                vdupq_n_f16(0.f),
                vdupq_n_f16(0.f)
            }
        };

        /*
        This kernel puts the values in a 4x4 block of Matrix A on the same row (Interleaved values)
             |a00 a01 a02 a03 | a04 a05 a06 a07|
             |a10 a11 a12 a13 | a14 a15 a16 a17|
             |a20 a21 a22 a23 | a24 a25 a26 a27| = | a00 a10 a20 a30 || a01 a11 a21 a31 || a02 a12 a22 a32 || a03 a13 a23 a33 | a40 a50 a60 a70 | ...
             |a30 a31 a32 a33 | a34 a35 a36 a37|   | a04 a14 a24 a34 || a05 a15 a25 a35 || a06 a15 a26 a36 || a07 a17 a27 a37 | a44 a54 a64 a74 | ...
             |a40 a41 a42 a43 | a44 a45 a46 a47|
             |a50 a51 a52 a53 | a54 a55 a56 a57|
             |a60 a61 a62 a63 | a64 a65 a66 a67|
             |a70 a71 a72 a73 | a74 a75 a76 a77|

             After this operation, the dst matrix will have the following shape: [ height * 4, width / 4 ]

        B Matrix has been transposed as shown below

           |b00 b01 b02 b03 b04 b05 b06 b07|
           |b10 b11 b12 b13 b14 b15 b16 b17|
           |b20 b21 b22 b23 b24 b25 b26 b27|
           |b30 b31 b32 b33 b34 b35 b36 b37|
          ------------------->

           |b00 b01 b02 b03 b04 b05 b06 b07||b10 b11 b12 b13 b14 b15 b16 b17||b20 b21 b22 b23 b24 b25 b26 b27||b30 b31 b32 b33 b34 b35 b36 b37|

            c.val[0][0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
            c.val[0][1] = a00*b01 + a01*b11 + a02*b21 + a03*b31

        The size of the dst tensor's XY-plane must be the following shape [ width * 8, height / 8 ]. All other dimensions must have the same size.
        */
        const float16_t *mtx_b0_end_addr = mtx_b0 + num_elems_matrix_b_x;

        for(; mtx_b0 <= (mtx_b0_end_addr - 32);)

        {
            const float16x8_t p00 = vld1q_f16(mtx_a0);
            const float16x8_t p02 = vld1q_f16(mtx_a0 + 8);

            const float16x8_t q00 = vld1q_f16(mtx_b0);
            const float16x8_t q02 = vld1q_f16(mtx_b0 + 8);
            const float16x8_t q04 = vld1q_f16(mtx_b0 + 16);
            const float16x8_t q06 = vld1q_f16(mtx_b0 + 24);

            c.val[0] = vaddq_f16(c.val[0], vmulq_n_f16(q00, vgetq_lane_f16(p00, 0)));
            c.val[1] = vaddq_f16(c.val[1], vmulq_n_f16(q00, vgetq_lane_f16(p00, 1)));
            c.val[2] = vaddq_f16(c.val[2], vmulq_n_f16(q00, vgetq_lane_f16(p00, 2)));
            c.val[3] = vaddq_f16(c.val[3], vmulq_n_f16(q00, vgetq_lane_f16(p00, 3)));

            c.val[0] = vaddq_f16(c.val[0], vmulq_n_f16(q02, vgetq_lane_f16(p00, 4)));
            c.val[1] = vaddq_f16(c.val[1], vmulq_n_f16(q02, vgetq_lane_f16(p00, 5)));
            c.val[2] = vaddq_f16(c.val[2], vmulq_n_f16(q02, vgetq_lane_f16(p00, 6)));
            c.val[3] = vaddq_f16(c.val[3], vmulq_n_f16(q02, vgetq_lane_f16(p00, 7)));

            c.val[0] = vaddq_f16(c.val[0], vmulq_n_f16(q04, vgetq_lane_f16(p02, 0)));
            c.val[1] = vaddq_f16(c.val[1], vmulq_n_f16(q04, vgetq_lane_f16(p02, 1)));
            c.val[2] = vaddq_f16(c.val[2], vmulq_n_f16(q04, vgetq_lane_f16(p02, 2)));
            c.val[3] = vaddq_f16(c.val[3], vmulq_n_f16(q04, vgetq_lane_f16(p02, 3)));

            c.val[0] = vaddq_f16(c.val[0], vmulq_n_f16(q06, vgetq_lane_f16(p02, 4)));
            c.val[1] = vaddq_f16(c.val[1], vmulq_n_f16(q06, vgetq_lane_f16(p02, 5)));
            c.val[2] = vaddq_f16(c.val[2], vmulq_n_f16(q06, vgetq_lane_f16(p02, 6)));
            c.val[3] = vaddq_f16(c.val[3], vmulq_n_f16(q06, vgetq_lane_f16(p02, 7)));

            mtx_a0 += 16;
            mtx_b0 += 32;
        }

        for(; mtx_b0 < mtx_b0_end_addr;)

        {
            const float16x4_t p00 = vld1_f16(mtx_a0);
            const float16x8_t q00 = vld1q_f16(mtx_b0);

            c.val[0] = vaddq_f16(c.val[0], vmulq_n_f16(q00, vget_lane_f16(p00, 0)));
            c.val[1] = vaddq_f16(c.val[1], vmulq_n_f16(q00, vget_lane_f16(p00, 1)));
            c.val[2] = vaddq_f16(c.val[2], vmulq_n_f16(q00, vget_lane_f16(p00, 2)));
            c.val[3] = vaddq_f16(c.val[3], vmulq_n_f16(q00, vget_lane_f16(p00, 3)));

            mtx_a0 += 4;
            mtx_b0 += 8;
        }

        if(multiply_alpha)
        {
            c.val[0] = vmulq_f16(c.val[0], alpha_f16);
            c.val[1] = vmulq_f16(c.val[1], alpha_f16);
            c.val[2] = vmulq_f16(c.val[2], alpha_f16);
            c.val[3] = vmulq_f16(c.val[3], alpha_f16);
        }

        if(id.x() < (out_width - 8))
        {
            vst1q_f16(mtx_out, c.val[0]);
            if(id.y() + 1 < out_height)
            {
                vst1q_f16(mtx_out + 1 * out_stride, c.val[1]);
                if(id.y() + 2 < out_height)
                {
                    vst1q_f16(mtx_out + 2 * out_stride, c.val[2]);
                    if(id.y() + 3 < out_height)
                    {
                        vst1q_f16(mtx_out + 3 * out_stride, c.val[3]);
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
                *(mtx_out + x) = c.val[0][x];
                if(id.y() + 1 < out_height)
                {
                    *(mtx_out + x + 1 * out_stride) = c.val[1][x];
                    if(id.y() + 2 < out_height)
                    {
                        *(mtx_out + x + 2 * out_stride) = c.val[2][x];
                        if(id.y() + 3 < out_height)
                        {
                            *(mtx_out + x + 3 * out_stride) = c.val[3][x];
                        }
                    }
                }
            }
        }
    },
    ina, inb, out);
}

void neon_fp16_gemm_matrix_mul(const ITensor *lhs, const ITensor *rhs, ITensor *dst, const Window &window, const ThreadInfo &info, float alpha, const bool is_dst_vector)
{
    return (is_dst_vector) ? vector_matrix_multiply_f16(lhs, rhs, dst, window, info, alpha) : matrix_matrix_multiply_f16(lhs, rhs, dst, window, info, alpha);
}
} // namespce cpu
} // namespace arm_compute
#endif //__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
