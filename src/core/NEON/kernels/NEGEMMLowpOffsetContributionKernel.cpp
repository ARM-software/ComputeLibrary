/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpOffsetContributionKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row,
                          int32_t a_offset, int32_t b_offset)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mm_result, 1, DataType::S32);

    // If a_offset == 0, vector_sum_col can be a nullptr
    if(a_offset != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_col, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(vector_sum_col->dimension(0) != mm_result->dimension(0));
    }

    // If b_offset == 0, vector_sum_row can be a nullptr
    if(b_offset != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_row, 1, DataType::S32);

        // Check if input is a 3D reinterpretation
        const bool reinterpret_as_3d = mm_result->num_dimensions() > 1 && mm_result->tensor_shape().y() != vector_sum_row->tensor_shape().x();

        // Validate input
        ARM_COMPUTE_RETURN_ERROR_ON(reinterpret_as_3d && vector_sum_row->dimension(0) != (mm_result->dimension(1) * mm_result->dimension(2)));
        ARM_COMPUTE_RETURN_ERROR_ON(!reinterpret_as_3d && vector_sum_row->dimension(0) != mm_result->dimension(1));

        TensorShape output_shape = mm_result->tensor_shape();
        if(output_shape.num_dimensions() > 1)
        {
            const unsigned int output_batch_idx = reinterpret_as_3d ? 3 : 2;

            TensorShape vector_sum_row_shape = vector_sum_row->tensor_shape();
            vector_sum_row_shape.collapse_from(1);
            output_shape.collapse_from(output_batch_idx);

            ARM_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_row_shape[1] != output_shape[output_batch_idx],
                                            "mm_result tensor must have the same number of batches of output tensor");

            if(a_offset != 0)
            {
                TensorShape vector_sum_col_shape = vector_sum_col->tensor_shape();
                vector_sum_col_shape.collapse_from(1);

                ARM_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_col_shape[1] != 1 && vector_sum_col_shape[1] != vector_sum_row_shape[1],
                                                "vector_sum_col tensor must have the same number of batches of vector_sum_row_shape or the number of batches must be set to 1");
            }
        }
    }

    return Status{};
}

void run_offset_contribution(const Window &window,
                             ITensor *mm_result, const ITensor *vector_sum_col, const ITensor *vector_sum_row,
                             int32_t a_offset, int32_t b_offset, int32_t k_offset, bool slide_vector_sum_col, bool is_gemm3d)
{
    Window collapsed_window = window.collapse_if_possible(window, Window::DimZ);
    collapsed_window.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int height_input = is_gemm3d ? mm_result->info()->dimension(1) : 0;
    const int depth_input  = is_gemm3d ? mm_result->info()->dimension(2) : 1;

    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();
    const int window_step_x  = 16;

    Iterator mm_result_it(mm_result, collapsed_window);

    if((a_offset != 0) && (b_offset != 0) && (vector_sum_col != nullptr) && (vector_sum_row != nullptr)) // true, true
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
        const int vector_sum_col_batch_offset = slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

        execute_window_loop(collapsed_window, [&](const Coordinates & id)
        {
            const int batch_id           = id.z() / depth_input;
            auto      vector_sum_col_ptr = reinterpret_cast<const int32_t *>(vector_sum_col_it.ptr() + batch_id * vector_sum_col_batch_offset);
            auto      mm_result_ptr      = reinterpret_cast<int32_t *>(mm_result_it.ptr());

            // Compute the leftover term due to b_offset.
            int32_t b_offset_term_s32 = *(reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() + batch_id * sum_row_stride_y) + id.y() + (id.z() % depth_input) * height_input);
            b_offset_term_s32 *= b_offset;

            const int32x4_t b_offset_term_s32_vec = vdupq_n_s32(b_offset_term_s32);

            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                // Compute the leftover term due to a_offset.
                int32x4x4_t a_offset_term_s32 =
                {
                    {
                        vld1q_s32(vector_sum_col_ptr + x + 0),
                        vld1q_s32(vector_sum_col_ptr + x + 4),
                        vld1q_s32(vector_sum_col_ptr + x + 8),
                        vld1q_s32(vector_sum_col_ptr + x + 12)
                    }
                };

                a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], a_offset);
                a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], a_offset);
                a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], a_offset);
                a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], a_offset);

                // Add a_offset_term_s32 and b_offset_term_s32
                int32x4x4_t offset_term_s32 =
                {
                    {
                        vdupq_n_s32(k_offset),
                        vdupq_n_s32(k_offset),
                        vdupq_n_s32(k_offset),
                        vdupq_n_s32(k_offset)
                    }
                };

                offset_term_s32.val[0] = vaddq_s32(offset_term_s32.val[0], vaddq_s32(a_offset_term_s32.val[0], b_offset_term_s32_vec));
                offset_term_s32.val[1] = vaddq_s32(offset_term_s32.val[1], vaddq_s32(a_offset_term_s32.val[1], b_offset_term_s32_vec));
                offset_term_s32.val[2] = vaddq_s32(offset_term_s32.val[2], vaddq_s32(a_offset_term_s32.val[2], b_offset_term_s32_vec));
                offset_term_s32.val[3] = vaddq_s32(offset_term_s32.val[3], vaddq_s32(a_offset_term_s32.val[3], b_offset_term_s32_vec));

                int32x4x4_t in_s32 =
                {
                    {
                        vld1q_s32(mm_result_ptr + x + 0),
                        vld1q_s32(mm_result_ptr + x + 4),
                        vld1q_s32(mm_result_ptr + x + 8),
                        vld1q_s32(mm_result_ptr + x + 12)
                    }
                };

                // Add the offset terms to GEMM's result
                in_s32.val[0] = vaddq_s32(in_s32.val[0], offset_term_s32.val[0]);
                in_s32.val[1] = vaddq_s32(in_s32.val[1], offset_term_s32.val[1]);
                in_s32.val[2] = vaddq_s32(in_s32.val[2], offset_term_s32.val[2]);
                in_s32.val[3] = vaddq_s32(in_s32.val[3], offset_term_s32.val[3]);

                // Store the result with the offset contribution
                vst1q_s32(mm_result_ptr + x + 0, in_s32.val[0]);
                vst1q_s32(mm_result_ptr + x + 4, in_s32.val[1]);
                vst1q_s32(mm_result_ptr + x + 8, in_s32.val[2]);
                vst1q_s32(mm_result_ptr + x + 12, in_s32.val[3]);
            }

            // Left-overs loop
            for(; x < window_end_x; ++x)
            {
                // Compute the leftover term due to a_offset.
                int32_t a_offset_term_s32 = *(vector_sum_col_ptr + x);

                a_offset_term_s32 *= a_offset;

                // Add the offset terms to GEMM's result
                // Store the result with the offset contribution
                mm_result_ptr[x] += k_offset + a_offset_term_s32 + b_offset_term_s32;
            }
        },
        vector_sum_col_it, vector_sum_row_it, mm_result_it);
    }
    else if((a_offset == 0) && (b_offset != 0) && (vector_sum_row != nullptr)) // false, true
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(vector_sum_row);

        // Set window for vector_sum_row
        Window win_vector_sum_row(collapsed_window);
        win_vector_sum_row.set(Window::DimX, Window::Dimension(0, 0, 0));
        win_vector_sum_row.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_vector_sum_row.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Iterator vector_sum_row_it(vector_sum_row, win_vector_sum_row);

        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

        execute_window_loop(collapsed_window, [&](const Coordinates & id)
        {
            const int batch_id      = id.z() / depth_input;
            auto      mm_result_ptr = reinterpret_cast<int32_t *>(mm_result_it.ptr());

            // Compute the leftover term due to b_offset.
            int32_t b_offset_term_s32 = *(reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() + batch_id * sum_row_stride_y) + id.y() + (id.z() % depth_input) * height_input);
            b_offset_term_s32 *= b_offset;

            const int32x4_t b_offset_term_s32_vec = vdupq_n_s32(b_offset_term_s32);

            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                int32x4x4_t in_s32 =
                {
                    {
                        vld1q_s32(mm_result_ptr + x + 0),
                        vld1q_s32(mm_result_ptr + x + 4),
                        vld1q_s32(mm_result_ptr + x + 8),
                        vld1q_s32(mm_result_ptr + x + 12)
                    }
                };

                // Add the offset terms to GEMM's result
                in_s32.val[0] = vaddq_s32(in_s32.val[0], b_offset_term_s32_vec);
                in_s32.val[1] = vaddq_s32(in_s32.val[1], b_offset_term_s32_vec);
                in_s32.val[2] = vaddq_s32(in_s32.val[2], b_offset_term_s32_vec);
                in_s32.val[3] = vaddq_s32(in_s32.val[3], b_offset_term_s32_vec);

                // Store the result with the offset contribution
                vst1q_s32(mm_result_ptr + x + 0, in_s32.val[0]);
                vst1q_s32(mm_result_ptr + x + 4, in_s32.val[1]);
                vst1q_s32(mm_result_ptr + x + 8, in_s32.val[2]);
                vst1q_s32(mm_result_ptr + x + 12, in_s32.val[3]);
            }

            // Left-overs loop
            for(; x < window_end_x; ++x)
            {
                // Add the offset terms to GEMM's result
                // Store the result with the offset contribution
                mm_result_ptr[x] += b_offset_term_s32;
            }
        },
        vector_sum_row_it, mm_result_it);
    }
    else if((a_offset != 0) && (b_offset == 0) && (vector_sum_col != nullptr)) // true, false
    {
        // Set window for vector_sum_col
        Window win_vector_sum_col(collapsed_window);
        win_vector_sum_col.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_vector_sum_col.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Iterator vector_sum_col_it(vector_sum_col, win_vector_sum_col);

        // Offset in case vector_sum_col is batched
        const int vector_sum_col_batch_offset = slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

        execute_window_loop(collapsed_window, [&](const Coordinates & id)
        {
            const int batch_id           = id.z() / depth_input;
            auto      vector_sum_col_ptr = reinterpret_cast<const int32_t *>(vector_sum_col_it.ptr() + batch_id * vector_sum_col_batch_offset);
            auto      mm_result_ptr      = reinterpret_cast<int32_t *>(mm_result_it.ptr());

            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                // Compute the leftover term due to a_offset.
                int32x4x4_t a_offset_term_s32 =
                {
                    {
                        vld1q_s32(vector_sum_col_ptr + x + 0),
                        vld1q_s32(vector_sum_col_ptr + x + 4),
                        vld1q_s32(vector_sum_col_ptr + x + 8),
                        vld1q_s32(vector_sum_col_ptr + x + 12)
                    }
                };

                a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], a_offset);
                a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], a_offset);
                a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], a_offset);
                a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], a_offset);

                int32x4x4_t in_s32 =
                {
                    {
                        vld1q_s32(mm_result_ptr + x + 0),
                        vld1q_s32(mm_result_ptr + x + 4),
                        vld1q_s32(mm_result_ptr + x + 8),
                        vld1q_s32(mm_result_ptr + x + 12)
                    }
                };

                // Add the offset terms to GEMM's result
                in_s32.val[0] = vaddq_s32(in_s32.val[0], a_offset_term_s32.val[0]);
                in_s32.val[1] = vaddq_s32(in_s32.val[1], a_offset_term_s32.val[1]);
                in_s32.val[2] = vaddq_s32(in_s32.val[2], a_offset_term_s32.val[2]);
                in_s32.val[3] = vaddq_s32(in_s32.val[3], a_offset_term_s32.val[3]);

                // Store the result with the offset contribution
                vst1q_s32(mm_result_ptr + x + 0, in_s32.val[0]);
                vst1q_s32(mm_result_ptr + x + 4, in_s32.val[1]);
                vst1q_s32(mm_result_ptr + x + 8, in_s32.val[2]);
                vst1q_s32(mm_result_ptr + x + 12, in_s32.val[3]);
            }

            // Left-overs loop
            for(; x < window_end_x; ++x)
            {
                // Compute the leftover term due to a_offset.
                const int32_t a_offset_term_s32 = *(vector_sum_col_ptr + x);

                // Add the offset terms to GEMM's result
                // Store the result with the offset contribution
                mm_result_ptr[x] += a_offset_term_s32 * a_offset;
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
} // namespace

NEGEMMLowpOffsetContributionKernel::NEGEMMLowpOffsetContributionKernel()
    : _vector_sum_col(nullptr), _vector_sum_row(nullptr), _mm_result(nullptr), _a_offset(0), _b_offset(0), _k_offset(0), _slide_vector_sum_col(true)
{
}

void NEGEMMLowpOffsetContributionKernel::configure(ITensor *mm_result, const ITensor *vector_sum_col, const ITensor *vector_sum_row, int32_t k, int32_t a_offset, int32_t b_offset)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(mm_result);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(mm_result->info(),
                                                  vector_sum_col != nullptr ? vector_sum_col->info() : nullptr, // NOLINT
                                                  vector_sum_row != nullptr ? vector_sum_row->info() : nullptr, // NOLINT
                                                  a_offset, b_offset));                                         // NOLINT

    _vector_sum_col = vector_sum_col;
    _vector_sum_row = vector_sum_row;
    _mm_result      = mm_result;
    _a_offset       = a_offset;
    _b_offset       = b_offset;
    _k_offset       = a_offset * b_offset * k;

    // If a_offset == 0, vector_sum_col can be a nullptr
    if(a_offset != 0)
    {
        // Check if vector_sum_col_shape should be slidden or not
        // Don't slide vector_sum_col_shape along the y dimension if vector_sum_col_shape has just 1 dimension and vector_sum_row_shape more than 1
        // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
        _slide_vector_sum_col = vector_sum_col->info()->tensor_shape().num_dimensions() > 1;
    }

    // Configure kernel window
    Window      win = calculate_max_window(*mm_result->info(), Steps());
    Coordinates coord;
    coord.set_num_dimensions(mm_result->info()->num_dimensions());
    mm_result->info()->set_valid_region(ValidRegion(coord, mm_result->info()->tensor_shape()));
    INEKernel::configure(win);
}

Status NEGEMMLowpOffsetContributionKernel::validate(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row,
                                                    int32_t a_offset, int32_t b_offset)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(mm_result, vector_sum_col, vector_sum_row, a_offset, b_offset));

    return Status{};
}

void NEGEMMLowpOffsetContributionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    // Check if input is a 3D reinterpretation
    const bool reinterpret_as_3d = _vector_sum_row != nullptr
                                   && _mm_result->info()->num_dimensions() > 1
                                   && _mm_result->info()->tensor_shape().y() != _vector_sum_row->info()->tensor_shape().x();

    run_offset_contribution(window, _mm_result, _vector_sum_col, _vector_sum_row, _a_offset, _b_offset, _k_offset, _slide_vector_sum_col, reinterpret_as_3d);
}
} // namespace arm_compute
