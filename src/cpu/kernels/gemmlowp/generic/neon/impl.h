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
#ifndef ACL_SRC_CPU_KERNELS_GEMMLOWP_GENERIC_NEON_IMPL_H
#define ACL_SRC_CPU_KERNELS_GEMMLOWP_GENERIC_NEON_IMPL_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEMath.h"

namespace arm_compute
{

template <typename T>
void convert_scale_store(int32x4x4_t &offset_term_s32, T scale, T *mm_result_pt);

template <typename T>
void add_contribution_boffset_store(const T b_offset_term_scalar, T *mm_result_ptr);

template <typename T>
void neon_run_offset_contribution_float(const Window  &window,
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
    Window collapsed_window = window.collapse_if_possible(window, Window::DimZ);
    collapsed_window.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int height_input = is_gemm3d ? mm_result->info()->dimension(1) : 0;
    const int depth_input  = is_gemm3d ? mm_result->info()->dimension(2) : 1;

    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();
    const int window_step_x  = 16;

    // if vector_sum_col is nullptr then stride_y is 0, else get stride_y
    const size_t sum_col_stride_y = (vector_sum_col != nullptr) ? (vector_sum_col->info()->strides_in_bytes().y()) : 0;
    Iterator     mm_result_it(mm_result, collapsed_window);

    if ((a_offset != 0) && (b_offset != 0) && (vector_sum_col != nullptr) && (vector_sum_row != nullptr)) // true, true
    {
        // Set window for vector_sum_col
        Window win_vector_sum_col(collapsed_window);
        win_vector_sum_col.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_vector_sum_col.set(Window::DimZ, Window::Dimension(0, 0, 0));

        // Set window for vector_sum_row
        Window win_vector_sum_row(collapsed_window);
        win_vector_sum_row.set(Window::DimX, Window::Dimension(0, 0, 0));
        win_vector_sum_row.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_vector_sum_row.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Iterator vector_sum_col_it(vector_sum_col, win_vector_sum_col);
        Iterator vector_sum_row_it(vector_sum_row, win_vector_sum_row);

        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

        // Offset in case vector_sum_col is batched
        const int vector_sum_col_batch_offset =
            slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

        execute_window_loop(
            collapsed_window,
            [&](const Coordinates &id)
            {
                const int    batch_id         = id.z() / depth_input;
                const size_t batch_offset_col = batch_id * sum_col_stride_y;
                auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(vector_sum_col_it.ptr() + batch_offset_col +
                                                                            batch_id * vector_sum_col_batch_offset);
                auto mm_result_ptr      = reinterpret_cast<T *>(mm_result_it.ptr());

                // Compute the leftover term due to b_offset.
                int32_t b_offset_term_s32 =
                    *(reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() + batch_id * sum_row_stride_y) +
                      id.y() + (id.z() % depth_input) * height_input);
                b_offset_term_s32 *= b_offset;

                const int32x4_t b_offset_term_s32_vec = vdupq_n_s32(b_offset_term_s32);

                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    // Compute the leftover term due to a_offset.
                    int32x4x4_t a_offset_term_s32 = {
                        {vld1q_s32(vector_sum_col_ptr + x + 0), vld1q_s32(vector_sum_col_ptr + x + 4),
                         vld1q_s32(vector_sum_col_ptr + x + 8), vld1q_s32(vector_sum_col_ptr + x + 12)}};

                    a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], a_offset);
                    a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], a_offset);
                    a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], a_offset);
                    a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], a_offset);

                    // Add a_offset_term_s32 and b_offset_term_s32
                    int32x4x4_t offset_term_s32 = {
                        {vdupq_n_s32(k_offset), vdupq_n_s32(k_offset), vdupq_n_s32(k_offset), vdupq_n_s32(k_offset)}};

                    offset_term_s32.val[0] =
                        vaddq_s32(offset_term_s32.val[0], vaddq_s32(a_offset_term_s32.val[0], b_offset_term_s32_vec));
                    offset_term_s32.val[1] =
                        vaddq_s32(offset_term_s32.val[1], vaddq_s32(a_offset_term_s32.val[1], b_offset_term_s32_vec));
                    offset_term_s32.val[2] =
                        vaddq_s32(offset_term_s32.val[2], vaddq_s32(a_offset_term_s32.val[2], b_offset_term_s32_vec));
                    offset_term_s32.val[3] =
                        vaddq_s32(offset_term_s32.val[3], vaddq_s32(a_offset_term_s32.val[3], b_offset_term_s32_vec));

                    convert_scale_store<T>(offset_term_s32, scale, mm_result_ptr + x);
                }

                // Left-overs loop
                for (; x < window_end_x; ++x)
                {
                    // Compute the leftover term due to a_offset.
                    int32_t a_offset_term_s32 = *(vector_sum_col_ptr + x);

                    a_offset_term_s32 *= a_offset;

                    // Add the offset terms to GEMM's result
                    // Store the result with the offset contribution
                    mm_result_ptr[x] += (k_offset + a_offset_term_s32 + b_offset_term_s32) * scale;
                }
            },
            vector_sum_col_it, vector_sum_row_it, mm_result_it);
    }
    else if ((a_offset == 0) && (b_offset != 0) && (vector_sum_row != nullptr)) // false, true
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(vector_sum_row);

        // Set window for vector_sum_row
        Window win_vector_sum_row(collapsed_window);
        win_vector_sum_row.set(Window::DimX, Window::Dimension(0, 0, 0));
        win_vector_sum_row.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_vector_sum_row.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Iterator vector_sum_row_it(vector_sum_row, win_vector_sum_row);

        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

        execute_window_loop(
            collapsed_window,
            [&](const Coordinates &id)
            {
                const int batch_id      = id.z() / depth_input;
                auto      mm_result_ptr = reinterpret_cast<T *>(mm_result_it.ptr());

                // Compute the leftover term due to b_offset.
                int32_t row_sum =
                    *(reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() + batch_id * sum_row_stride_y) +
                      id.y() + (id.z() % depth_input) * height_input);
                const T scaled_b_offset_term_scalar = row_sum * b_offset * scale;

                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    add_contribution_boffset_store<T>(scaled_b_offset_term_scalar, mm_result_ptr + x);
                }

                // Left-overs loop
                for (; x < window_end_x; ++x)
                {
                    // Add the offset terms to GEMM's result
                    // Store the result with the offset contribution
                    mm_result_ptr[x] += scaled_b_offset_term_scalar;
                }
            },
            vector_sum_row_it, mm_result_it);
    }
    else if ((a_offset != 0) && (b_offset == 0) && (vector_sum_col != nullptr)) // true, false
    {
        // Set window for vector_sum_col
        Window win_vector_sum_col(collapsed_window);
        win_vector_sum_col.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_vector_sum_col.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Iterator vector_sum_col_it(vector_sum_col, win_vector_sum_col);

        // Offset in case vector_sum_col is batched
        const int vector_sum_col_batch_offset =
            slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

        execute_window_loop(
            collapsed_window,
            [&](const Coordinates &id)
            {
                const int    batch_id = id.z() / depth_input;
                const size_t batch_offset_col =
                    batch_id *
                    sum_col_stride_y; // Value to offset vector_sum_col_ptr to allow for iteration of y values in tensor
                auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(vector_sum_col_it.ptr() + batch_offset_col +
                                                                            batch_id * vector_sum_col_batch_offset);
                auto mm_result_ptr      = reinterpret_cast<T *>(mm_result_it.ptr());

                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    // Compute the leftover term due to a_offset.
                    int32x4x4_t a_offset_term_s32 = {
                        {vld1q_s32(vector_sum_col_ptr + x + 0), vld1q_s32(vector_sum_col_ptr + x + 4),
                         vld1q_s32(vector_sum_col_ptr + x + 8), vld1q_s32(vector_sum_col_ptr + x + 12)}};

                    a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], a_offset);
                    a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], a_offset);
                    a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], a_offset);
                    a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], a_offset);

                    convert_scale_store<T>(a_offset_term_s32, scale, mm_result_ptr + x);
                }

                // Left-overs loop
                for (; x < window_end_x; ++x)
                {
                    // Compute the leftover term due to a_offset.
                    const int32_t a_offset_term_s32 = *(vector_sum_col_ptr + x);

                    // Add the offset terms to GEMM's result
                    // Store the result with the offset contribution
                    mm_result_ptr[x] += a_offset_term_s32 * a_offset * scale;
                }
            },
            vector_sum_col_it, mm_result_it);
    }
    else // false, false
    {
        // No offset contribution from matrix A and matrix B
        return;
    }
}

} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_GEMMLOWP_GENERIC_NEON_IMPL_H
