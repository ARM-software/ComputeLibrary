/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/helpers/float_ops.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
template <bool multiply_alpha>
void vector_matrix_multiply_f16(const ITensor *input0, const ITensor *input1, ITensor *output, const Window &window, const ThreadInfo &info, float alpha)
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    const auto width_matrix_b  = static_cast<int>(output->info()->dimension(0));
    const auto in_b_stride     = static_cast<int>(input1->info()->strides_in_bytes()[1] / data_size_from_type(input1->info()->data_type()));
    const auto num_elems_vec_a = static_cast<int>(input0->info()->dimension(0));

    // The implementation computes 32 elements per iteration
    const int window_start_x = 32 * info.thread_id;
    const int window_step_x  = 32 * info.num_threads;
    const int window_end_x   = ceil_to_multiple(width_matrix_b - window_start_x, window_step_x) + window_start_x;
    ARM_COMPUTE_ERROR_ON_MSG((window_end_x - window_start_x) % window_step_x, " (window_end_x - window_start_x) must be multiple of window_step_x");

    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(window_start_x, window_end_x, window_step_x));
    win_out.set(Window::DimY, Window::Dimension(0, 1, 1));

    Window win_a(window);
    win_a.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_a.set(Window::DimY, Window::Dimension(0, 0, 0));

    Window win_b;
    // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
    // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
    if(input1->info()->num_dimensions() >= 3)
    {
        win_b = window;
    }
    win_b.set(Window::DimX, Window::Dimension(window_start_x, window_end_x, window_step_x));
    win_b.set(Window::DimY, Window::Dimension(0, 1, 1));

    Iterator ina(input0, win_a);
    Iterator inb(input1, win_b);
    Iterator out(output, win_out);

    const float16x8_t alpha_f16 = vdupq_n_f16(alpha);
    ARM_COMPUTE_UNUSED(alpha_f16);

    execute_window_loop(win_out, [&](const Coordinates & id)
    {
        if(id.x() > width_matrix_b)
        {
            return;
        }

        float16x8_t acc0 = vdupq_n_f16(0.f);
        float16x8_t acc1 = vdupq_n_f16(0.f);
        float16x8_t acc2 = vdupq_n_f16(0.f);
        float16x8_t acc3 = vdupq_n_f16(0.f);

        auto vec_a    = reinterpret_cast<const float16_t *>(ina.ptr());
        auto matrix_b = reinterpret_cast<const float16_t *>(inb.ptr());

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

        for(; vec_a < vec_a_end_addr;)
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

            vec_a += 1;
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

        const auto vec_out = reinterpret_cast<float16_t *>(out.ptr());

        vst1q_f16(vec_out + 0, acc0);
        vst1q_f16(vec_out + 8, acc1);
        vst1q_f16(vec_out + 16, acc2);
        vst1q_f16(vec_out + 24, acc3);

    },
    ina, inb, out);
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    ARM_COMPUTE_UNUSED(input0);
    ARM_COMPUTE_UNUSED(input1);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_ERROR("Not implemented");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

template <bool multiply_alpha>
void vector_matrix_multiply_f32(const ITensor *input0, const ITensor *input1, ITensor *output, const Window &window, const ThreadInfo &info, float alpha)
{
    const auto width_matrix_b  = static_cast<int>(output->info()->dimension(0));
    const auto in_b_stride     = static_cast<int>(input1->info()->strides_in_bytes()[1] / data_size_from_type(input1->info()->data_type()));
    const auto num_elems_vec_a = static_cast<int>(input0->info()->dimension(0));

    // The implementation computes 16 elements per iteration
    const int window_start_x = 16 * info.thread_id;
    const int window_step_x  = 16 * info.num_threads;
    // Make sure (window_end_x - window_start_x) is a multiple of window_step_x
    const int window_end_x = ceil_to_multiple(width_matrix_b - window_start_x, window_step_x) + window_start_x;

    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(window_start_x, window_end_x, window_step_x));
    win_out.set(Window::DimY, Window::Dimension(0, 1, 1));

    Window win_a(window);
    win_a.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_a.set(Window::DimY, Window::Dimension(0, 0, 0));

    Window win_b;
    // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
    // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
    if(input1->info()->num_dimensions() >= 3)
    {
        win_b = window;
    }
    win_b.set(Window::DimX, Window::Dimension(window_start_x, window_end_x, window_step_x));
    win_b.set(Window::DimY, Window::Dimension(0, 1, 1));

    Iterator ina(input0, win_a);
    Iterator inb(input1, win_b);
    Iterator out(output, win_out);

    execute_window_loop(win_out, [&](const Coordinates & id)
    {
        if(id.x() > width_matrix_b)
        {
            return;
        }

        float32x4_t acc0 = vdupq_n_f32(0.f);
        float32x4_t acc1 = vdupq_n_f32(0.f);
        float32x4_t acc2 = vdupq_n_f32(0.f);
        float32x4_t acc3 = vdupq_n_f32(0.f);

        auto vec_a    = reinterpret_cast<const float *>(ina.ptr());
        auto matrix_b = reinterpret_cast<const float *>(inb.ptr());

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

        for(; vec_a < vec_a_end_addr;)
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

            vec_a += 1;
            matrix_b += in_b_stride;
        }

        // Multiply by the weight of matrix product (alpha)
        if(multiply_alpha)
        {
            const float32x4_t alpha_f32 = vdupq_n_f32(alpha);
            acc0                        = vmulq_f32(acc0, alpha_f32);
            acc1                        = vmulq_f32(acc1, alpha_f32);
            acc2                        = vmulq_f32(acc2, alpha_f32);
            acc3                        = vmulq_f32(acc3, alpha_f32);
        }

        const auto vec_out = reinterpret_cast<float *>(out.ptr());

        vst1q_f32(vec_out + 0, acc0);
        vst1q_f32(vec_out + 4, acc1);
        vst1q_f32(vec_out + 8, acc2);
        vst1q_f32(vec_out + 12, acc3);
    },
    ina, inb, out);
}

template <bool multiply_alpha>
void matrix_matrix_multiply_f32(const ITensor *input0, const ITensor *input1, ITensor *output, const Window &window, float alpha)
{
    const size_t in_b_stride          = input1->info()->strides_in_bytes()[1] / data_size_from_type(input1->info()->data_type());
    const size_t out_stride1          = output->info()->strides_in_bytes()[1] / data_size_from_type(output->info()->data_type());
    const size_t out_stride2          = out_stride1 * 2;
    const size_t out_stride3          = out_stride1 * 3;
    const int    num_elems_matrix_b_x = input1->info()->dimension(0);

    // Set step_x and step_y for matrix A. Scale by a factor of 4 the Y range as the input interleaved matrix A has 4 times less the rows of the output matrix
    Window win_a(window);
    win_a.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_a.set(Window::DimY, Window::Dimension(window.y().start() / 4, std::max(window.y().end() / 4, 1), 1));

    Window win_b;
    // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
    // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
    if(input1->info()->num_dimensions() >= 3)
    {
        win_b = window;
    }
    // Set step_x and step_y for matrix B. Scale by a factor of 4 the X range as the input transposed matrix A has 4 times less the cols of the output matrix
    // The step along the x direction is 2 times the in_b_stride because for each iteration we compute 2 blocks of size 4x4
    win_b.set(Window::DimX, Window::Dimension(window.x().start() / 4, window.x().end() / 4, 2 * in_b_stride));
    win_b.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator ina(input0, win_a);
    Iterator inb(input1, win_b);
    Iterator out(output, window);

    // The implementation assumes that the matrix A and Matrix B have been reshaped respectively with NEGEMMInterleave4x4 and NEGEMMTranspose1xW
    // The reshaping of the matrices helps to have a cache friendly implementation and helps to avoid the data re-arrangements needed for computing 16x4 elements per iteration
    // All the values needed for computing a single 4x4 block will be read from consecutive memory positions
    execute_window_loop(window, [&](const Coordinates &)
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
            const float32x4_t alpha_f32 = vdupq_n_f32(alpha);
            acc00                       = vmulq_f32(acc00, alpha_f32);
            acc10                       = vmulq_f32(acc10, alpha_f32);
            acc20                       = vmulq_f32(acc20, alpha_f32);
            acc30                       = vmulq_f32(acc30, alpha_f32);
            acc01                       = vmulq_f32(acc01, alpha_f32);
            acc11                       = vmulq_f32(acc11, alpha_f32);
            acc21                       = vmulq_f32(acc21, alpha_f32);
            acc31                       = vmulq_f32(acc31, alpha_f32);
        }

        const auto mtx_out0 = reinterpret_cast<float *>(out.ptr());
        const auto mtx_out1 = mtx_out0 + 4;

        // Store the 4 blocks
        vst1q_f32(mtx_out0, acc00);
        vst1q_f32(mtx_out1, acc01);
        vst1q_f32(mtx_out0 + out_stride1, acc10);
        vst1q_f32(mtx_out1 + out_stride1, acc11);
        vst1q_f32(mtx_out0 + out_stride2, acc20);
        vst1q_f32(mtx_out1 + out_stride2, acc21);
        vst1q_f32(mtx_out0 + out_stride3, acc30);
        vst1q_f32(mtx_out1 + out_stride3, acc31);
    },
    ina, inb, out);
}

template <bool multiply_alpha>
void matrix_matrix_multiply_f16(const ITensor *input0, const ITensor *input1, ITensor *output, const Window &window, float alpha)
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    const size_t in_b_stride          = input1->info()->strides_in_bytes()[1] / data_size_from_type(input1->info()->data_type());
    const size_t out_stride           = output->info()->strides_in_bytes()[1] / data_size_from_type(output->info()->data_type());
    const int    num_elems_matrix_b_x = input1->info()->dimension(0);

    // Set step_x and step_y for matrix A. Scale by a factor of 4 the Y range as the input interleaved matrix A has 4 times less the rows of the output matrix
    Window win_a(window);
    win_a.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_a.set(Window::DimY, Window::Dimension(window.y().start() / 4, std::max(window.y().end() / 4, 1), 1));

    Window win_b;
    // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
    // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
    if(input1->info()->num_dimensions() >= 3)
    {
        win_b = window;
    }
    // Set step_x and step_y for matrix B. Scale by a factor of 8 the X range as the input transposed matrix A has 8 times less the cols of the output matrix
    win_b.set(Window::DimX, Window::Dimension(window.x().start() / 8, window.x().end() / 8, in_b_stride));
    win_b.set(Window::DimY, Window::Dimension(0, 1, 0));

    Iterator ina(input0, win_a);
    Iterator inb(input1, win_b);
    Iterator out(output, window);

    const float16x8_t alpha_f16 = vdupq_n_f16(alpha);

    execute_window_loop(window, [&](const Coordinates &)
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

             After this operation, the output matrix will have the following shape: [ height * 4, width / 4 ]

        B Matrix has been transposed as shown below

           |b00 b01 b02 b03 b04 b05 b06 b07|
           |b10 b11 b12 b13 b14 b15 b16 b17|
           |b20 b21 b22 b23 b24 b25 b26 b27|
           |b30 b31 b32 b33 b34 b35 b36 b37|
          ------------------->

           |b00 b01 b02 b03 b04 b05 b06 b07||b10 b11 b12 b13 b14 b15 b16 b17||b20 b21 b22 b23 b24 b25 b26 b27||b30 b31 b32 b33 b34 b35 b36 b37|

            c.val[0][0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
            c.val[0][1] = a00*b01 + a01*b11 + a02*b21 + a03*b31

        The size of the output tensor's XY-plane must be the following shape [ width * 8, height / 8 ]. All other dimensions must have the same size.
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

        vst1q_f16(mtx_out + 0 * out_stride, c.val[0]);
        vst1q_f16(mtx_out + 1 * out_stride, c.val[1]);
        vst1q_f16(mtx_out + 2 * out_stride, c.val[2]);
        vst1q_f16(mtx_out + 3 * out_stride, c.val[3]);
    },
    ina, inb, out);
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    ARM_COMPUTE_UNUSED(input0);
    ARM_COMPUTE_UNUSED(input1);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_ERROR("Not implemented");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

inline Status validate_arguments(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, float alpha, bool is_interleaved, const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_UNUSED(alpha);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1, output);

    if(!is_interleaved)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(0) != input1->dimension(1));

        if(output->total_size() != 0)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(input1->dimension(0) != output->dimension(0));
            ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(1) != output->dimension(1));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, output);
        }
    }
    else
    {
        const int m                         = reshape_info.m();
        const int n                         = reshape_info.n();
        const int k                         = reshape_info.k();
        const int mult_transpose1xW_width   = reshape_info.mult_transpose1xW_width();
        const int mult_interleave4x4_height = reshape_info.mult_interleave4x4_height();

        /* Interleave */
        TensorShape tensor_shape0{ input0->tensor_shape() };
        tensor_shape0.set(0, k);
        tensor_shape0.set(1, m);

        const TensorInfo tensor_info0          = input0->clone()->set_tensor_shape(tensor_shape0);
        const TensorInfo tensor_info_reshaped0 = input0->clone()->set_tensor_shape(misc::shape_calculator::compute_interleaved_shape(tensor_info0, mult_interleave4x4_height));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input0, &tensor_info_reshaped0);

        if(n != 0) /* Transpose */
        {
            TensorShape tensor_shape1{ input1->tensor_shape() };
            tensor_shape1.set(0, n);
            tensor_shape1.set(1, k);

            const TensorInfo tensor_info1          = input1->clone()->set_tensor_shape(tensor_shape1);
            const TensorInfo tensor_info_reshaped1 = input1->clone()->set_tensor_shape(misc::shape_calculator::compute_transpose1xW_with_element_size_shape(tensor_info1, mult_transpose1xW_width));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input1, &tensor_info_reshaped1);
        }

        if(output->total_size() != 0)
        {
            if(n != 0)
            {
                ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(0) != static_cast<size_t>(n));
            }
            ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(1) != static_cast<size_t>(m));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, output);
        }
    }

    return Status{};
}

inline std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input0, ITensorInfo *input1, ITensorInfo *output)
{
    bool   window_changed{};
    Window win{};

    unsigned int       num_elems_processed_per_iteration_x = 0;
    const unsigned int num_elems_processed_per_iteration_y = 4;

    // Check if the output tensor is a vector. If so,the kernel runs the vector-matrix multiplication
    if((output->dimension(1) == 1))
    {
        switch(input0->data_type())
        {
            case DataType::F32:
            {
                num_elems_processed_per_iteration_x = 16;
                break;
            }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
            {
                num_elems_processed_per_iteration_x = 32;
                break;
            }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            default:
            {
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
            }
        }

        // Configure kernel window
        win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x));

        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration_x);

        window_changed = update_window_and_padding(win,
                                                   AccessWindowStatic(input0, 0, 0, input0->tensor_shape().x(), 1),
                                                   AccessWindowHorizontal(input1, 0, num_elems_processed_per_iteration_x),
                                                   output_access);

        Coordinates coord;
        coord.set_num_dimensions(output->num_dimensions());
        output_access.set_valid_region(win, ValidRegion(coord, output->tensor_shape()));
    }
    else
    {
        switch(input0->data_type())
        {
            case DataType::F32:
            {
                num_elems_processed_per_iteration_x = 8;
                break;
            }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
            {
                num_elems_processed_per_iteration_x = 8;
                break;
            }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            default:
            {
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
            }
        }

        // Configure kernel window
        win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

        AccessWindowRectangle output_access(output, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

        window_changed = update_window_and_padding(win,
                                                   AccessWindowRectangle(input0, 0, 0, 4, 1, 1.f, 0.25f),
                                                   AccessWindowStatic(input1, 0, 0, input1->tensor_shape().x(), ceil_to_multiple(input1->tensor_shape().y(), 4)),
                                                   output_access);

        output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->tensor_shape()));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEGEMMMatrixMultiplyKernel::NEGEMMMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr), _alpha(1.0f)
{
}

void NEGEMMMatrixMultiplyKernel::configure(const ITensor *input0, const ITensor *input1, ITensor *output, float alpha, bool is_interleaved, const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input0, input1, output);

    // Output tensor auto inizialitation if not yet initialized
    TensorShape tensor_shape{ input0->info()->tensor_shape() };
    tensor_shape.set(0, is_interleaved ? reshape_info.n() : input1->info()->dimension(0));
    tensor_shape.set(1, is_interleaved ? reshape_info.m() : input0->info()->dimension(1));

    auto_init_if_empty(*output->info(), input0->info()->clone()->set_tensor_shape(tensor_shape));

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input0->info(), input1->info(), output->info(), alpha, is_interleaved, reshape_info));

    _input0 = input0;
    _input1 = input1;
    _output = output;
    _alpha  = alpha;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input0->info(), input1->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEGEMMMatrixMultiplyKernel::validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, float alpha, bool is_interleaved,
                                            const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input0, input1, output, alpha, is_interleaved, reshape_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input0->clone().get(), input1->clone().get(), output->clone().get()).first);

    return Status{};
}

void NEGEMMMatrixMultiplyKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const bool multiply_alpha = !(helpers::float_ops::is_one(_alpha));

    // Check if the output tensor is a vector. If so,the kernel runs the vector-matrix multiplication
    if((_output->info()->dimension(1) == 1))
    {
        switch(_input0->info()->data_type())
        {
            case DataType::F32:
            {
                multiply_alpha ? vector_matrix_multiply_f32<true>(_input0, _input1, _output, window, info, _alpha) :
                vector_matrix_multiply_f32<false>(_input0, _input1, _output, window, info, _alpha);
                break;
            }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
            {
                multiply_alpha ? vector_matrix_multiply_f16<true>(_input0, _input1, _output, window, info, _alpha) :
                vector_matrix_multiply_f16<false>(_input0, _input1, _output, window, info, _alpha);
                break;
            }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            default:
            {
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
            }
        }
    }
    else
    {
        switch(_input0->info()->data_type())
        {
            case DataType::F32:
            {
                multiply_alpha ? matrix_matrix_multiply_f32<true>(_input0, _input1, _output, window, _alpha) :
                matrix_matrix_multiply_f32<false>(_input0, _input1, _output, window, _alpha);
                break;
            }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
            {
                multiply_alpha ? matrix_matrix_multiply_f16<true>(_input0, _input1, _output, window, _alpha) :
                matrix_matrix_multiply_f16<false>(_input0, _input1, _output, window, _alpha);
                break;
            }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            default:
            {
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
            }
        }
    }
}
