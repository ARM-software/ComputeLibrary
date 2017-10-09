/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpFinalizeKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

template <bool add_a_offset, bool add_b_offset>
void NEGEMMLowpFinalizeKernel::finalize(const Window &window)
{
    const int32x4_t c_offset_s32 = vdupq_n_s32(_c_offset);
    const int32x4_t shift_s32    = vdupq_n_s32(-_shift);

    Window collapsed_window = window.collapse_if_possible(IKernel::window(), Window::DimZ);

    if(add_a_offset && add_b_offset) // true, true
    {
        // Set window for vector_sum_col
        Window win_vector_sum_col(collapsed_window);
        win_vector_sum_col.set(Window::DimY, Window::Dimension(0, 0, 0));
        if(!_slide_vector_sum_col)
        {
            win_vector_sum_col.set(Window::DimZ, Window::Dimension(0, 0, 0));
        }

        // Set window for vector_sum_row
        Window win_vector_sum_row(collapsed_window);
        win_vector_sum_row.set(Window::DimX, Window::Dimension(0, 0, 0));
        win_vector_sum_row.set(Window::DimY, Window::Dimension(0, 0, 0));

        Iterator vector_sum_col(_vector_sum_col, win_vector_sum_col);
        Iterator vector_sum_row(_vector_sum_row, win_vector_sum_row);
        Iterator mm_result(_mm_result, window);
        Iterator out(_output, window);

        execute_window_loop(window, [&](const Coordinates & id)
        {
            // Compute the leftover term due to a_offset.
            int32x4x4_t a_offset_term_s32 =
            {
                {
                    vld1q_s32(reinterpret_cast<const int32_t *>(vector_sum_col.ptr()) + 0),
                    vld1q_s32(reinterpret_cast<const int32_t *>(vector_sum_col.ptr()) + 4),
                    vld1q_s32(reinterpret_cast<const int32_t *>(vector_sum_col.ptr()) + 8),
                    vld1q_s32(reinterpret_cast<const int32_t *>(vector_sum_col.ptr()) + 12)
                }
            };

            a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], _a_offset);
            a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], _a_offset);
            a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], _a_offset);
            a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], _a_offset);

            // Compute the leftover term due to b_offset.
            int32x4_t b_offset_term_s32 = vld1q_dup_s32(reinterpret_cast<const int32_t *>(vector_sum_row.ptr()) + id.y());
            b_offset_term_s32           = vmulq_n_s32(b_offset_term_s32, _b_offset);

            // Add a_offset_term_s32 and b_offset_term_s32
            int32x4x4_t offset_term_s32 =
            {
                {
                    vdupq_n_s32(_k_offset),
                    vdupq_n_s32(_k_offset),
                    vdupq_n_s32(_k_offset),
                    vdupq_n_s32(_k_offset)
                }
            };

            offset_term_s32.val[0] = vaddq_s32(offset_term_s32.val[0], vaddq_s32(a_offset_term_s32.val[0], b_offset_term_s32));
            offset_term_s32.val[1] = vaddq_s32(offset_term_s32.val[1], vaddq_s32(a_offset_term_s32.val[1], b_offset_term_s32));
            offset_term_s32.val[2] = vaddq_s32(offset_term_s32.val[2], vaddq_s32(a_offset_term_s32.val[2], b_offset_term_s32));
            offset_term_s32.val[3] = vaddq_s32(offset_term_s32.val[3], vaddq_s32(a_offset_term_s32.val[3], b_offset_term_s32));

            // Add c_offset
            offset_term_s32.val[0] = vaddq_s32(offset_term_s32.val[0], c_offset_s32);
            offset_term_s32.val[1] = vaddq_s32(offset_term_s32.val[1], c_offset_s32);
            offset_term_s32.val[2] = vaddq_s32(offset_term_s32.val[2], c_offset_s32);
            offset_term_s32.val[3] = vaddq_s32(offset_term_s32.val[3], c_offset_s32);

            int32x4x4_t in_s32 =
            {
                {
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 0),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 4),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 8),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 12)
                }
            };

            // Add the offset terms to GEMM's result
            in_s32.val[0] = vaddq_s32(in_s32.val[0], offset_term_s32.val[0]);
            in_s32.val[1] = vaddq_s32(in_s32.val[1], offset_term_s32.val[1]);
            in_s32.val[2] = vaddq_s32(in_s32.val[2], offset_term_s32.val[2]);
            in_s32.val[3] = vaddq_s32(in_s32.val[3], offset_term_s32.val[3]);

            // Multiply by c_mult_int
            in_s32.val[0] = vmulq_n_s32(in_s32.val[0], _c_mult_int);
            in_s32.val[1] = vmulq_n_s32(in_s32.val[1], _c_mult_int);
            in_s32.val[2] = vmulq_n_s32(in_s32.val[2], _c_mult_int);
            in_s32.val[3] = vmulq_n_s32(in_s32.val[3], _c_mult_int);

            // Shift final result (negative value shift right)
            in_s32.val[0] = vshlq_s32(in_s32.val[0], shift_s32);
            in_s32.val[1] = vshlq_s32(in_s32.val[1], shift_s32);
            in_s32.val[2] = vshlq_s32(in_s32.val[2], shift_s32);
            in_s32.val[3] = vshlq_s32(in_s32.val[3], shift_s32);

            // Convert S32 to U16
            const int16x8x2_t in_u16 =
            {
                {
                    vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                    vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3])),
                }
            };

            // Convert U16 to U8
            const uint8x16_t out_u8 = vcombine_u8(vqmovun_s16(in_u16.val[0]), vqmovun_s16(in_u16.val[1]));

            vst1q_u8(out.ptr(), out_u8);
        },
        vector_sum_col, vector_sum_row, mm_result, out);
    }
    else if(!add_a_offset && add_b_offset) // false, true
    {
        // Set window for vector_sum_row
        Window win_vector_sum_row(collapsed_window);
        win_vector_sum_row.set(Window::DimX, Window::Dimension(0, 0, 0));
        win_vector_sum_row.set(Window::DimY, Window::Dimension(0, 0, 0));

        Iterator vector_sum_row(_vector_sum_row, win_vector_sum_row);
        Iterator mm_result(_mm_result, window);
        Iterator out(_output, window);

        execute_window_loop(window, [&](const Coordinates & id)
        {
            // Compute the leftover term due to b_offset.
            int32x4_t b_offset_term_s32 = vld1q_dup_s32(reinterpret_cast<const int32_t *>(vector_sum_row.ptr()) + id.y());
            b_offset_term_s32           = vmulq_n_s32(b_offset_term_s32, _b_offset);

            // Add b_offset_term_s32 and c_offset_term_s32
            int32x4_t offset_term_s32 = vaddq_s32(b_offset_term_s32, c_offset_s32);

            int32x4x4_t in_s32 =
            {
                {
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 0),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 4),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 8),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 12)
                }
            };

            // Add the offset terms to GEMM's result
            in_s32.val[0] = vaddq_s32(in_s32.val[0], offset_term_s32);
            in_s32.val[1] = vaddq_s32(in_s32.val[1], offset_term_s32);
            in_s32.val[2] = vaddq_s32(in_s32.val[2], offset_term_s32);
            in_s32.val[3] = vaddq_s32(in_s32.val[3], offset_term_s32);

            // Multiply by c_mult_int
            in_s32.val[0] = vmulq_n_s32(in_s32.val[0], _c_mult_int);
            in_s32.val[1] = vmulq_n_s32(in_s32.val[1], _c_mult_int);
            in_s32.val[2] = vmulq_n_s32(in_s32.val[2], _c_mult_int);
            in_s32.val[3] = vmulq_n_s32(in_s32.val[3], _c_mult_int);

            // Shift final result (negative value shift right)
            in_s32.val[0] = vshlq_s32(in_s32.val[0], shift_s32);
            in_s32.val[1] = vshlq_s32(in_s32.val[1], shift_s32);
            in_s32.val[2] = vshlq_s32(in_s32.val[2], shift_s32);
            in_s32.val[3] = vshlq_s32(in_s32.val[3], shift_s32);

            // Convert S32 to U16
            const int16x8x2_t in_u16 =
            {
                {
                    vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                    vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3])),
                }
            };

            // Convert U16 to U8
            const uint8x16_t out_u8 = vcombine_u8(vqmovun_s16(in_u16.val[0]), vqmovun_s16(in_u16.val[1]));

            vst1q_u8(out.ptr(), out_u8);
        },
        vector_sum_row, mm_result, out);
    }
    else if(add_a_offset && !add_b_offset) // true, false
    {
        // Set window for vector_sum_col
        Window win_vector_sum_col(collapsed_window);
        win_vector_sum_col.set(Window::DimY, Window::Dimension(0, 0, 0));
        if(!_slide_vector_sum_col)
        {
            win_vector_sum_col.set(Window::DimZ, Window::Dimension(0, 0, 0));
        }

        Iterator vector_sum_col(_vector_sum_col, win_vector_sum_col);
        Iterator mm_result(_mm_result, window);
        Iterator out(_output, window);

        execute_window_loop(window, [&](const Coordinates & id)
        {
            // Compute the leftover term due to a_offset.
            int32x4x4_t a_offset_term_s32 =
            {
                {
                    vld1q_s32(reinterpret_cast<const int32_t *>(vector_sum_col.ptr()) + 0),
                    vld1q_s32(reinterpret_cast<const int32_t *>(vector_sum_col.ptr()) + 4),
                    vld1q_s32(reinterpret_cast<const int32_t *>(vector_sum_col.ptr()) + 8),
                    vld1q_s32(reinterpret_cast<const int32_t *>(vector_sum_col.ptr()) + 12)
                }
            };

            a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], _a_offset);
            a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], _a_offset);
            a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], _a_offset);
            a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], _a_offset);

            // Add a_offset_term_s32 and b_offset_term_s32
            int32x4x4_t offset_term_s32 =
            {
                {
                    vaddq_s32(c_offset_s32, a_offset_term_s32.val[0]),
                    vaddq_s32(c_offset_s32, a_offset_term_s32.val[1]),
                    vaddq_s32(c_offset_s32, a_offset_term_s32.val[2]),
                    vaddq_s32(c_offset_s32, a_offset_term_s32.val[3])
                }
            };

            int32x4x4_t in_s32 =
            {
                {
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 0),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 4),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 8),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 12)
                }
            };

            // Add the offset terms to GEMM's result
            in_s32.val[0] = vaddq_s32(in_s32.val[0], offset_term_s32.val[0]);
            in_s32.val[1] = vaddq_s32(in_s32.val[1], offset_term_s32.val[1]);
            in_s32.val[2] = vaddq_s32(in_s32.val[2], offset_term_s32.val[2]);
            in_s32.val[3] = vaddq_s32(in_s32.val[3], offset_term_s32.val[3]);

            // Multiply by c_mult_int
            in_s32.val[0] = vmulq_n_s32(in_s32.val[0], _c_mult_int);
            in_s32.val[1] = vmulq_n_s32(in_s32.val[1], _c_mult_int);
            in_s32.val[2] = vmulq_n_s32(in_s32.val[2], _c_mult_int);
            in_s32.val[3] = vmulq_n_s32(in_s32.val[3], _c_mult_int);

            // Shift final result (negative value shift right)
            in_s32.val[0] = vshlq_s32(in_s32.val[0], shift_s32);
            in_s32.val[1] = vshlq_s32(in_s32.val[1], shift_s32);
            in_s32.val[2] = vshlq_s32(in_s32.val[2], shift_s32);
            in_s32.val[3] = vshlq_s32(in_s32.val[3], shift_s32);

            // Convert S32 to U16
            const int16x8x2_t in_u16 =
            {
                {
                    vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                    vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
                }
            };

            // Convert U16 to U8
            const uint8x16_t out_u8 = vcombine_u8(vqmovun_s16(in_u16.val[0]), vqmovun_s16(in_u16.val[1]));

            vst1q_u8(out.ptr(), out_u8);
        },
        vector_sum_col, mm_result, out);
    }
    else // false, false
    {
        Iterator mm_result(_mm_result, window);
        Iterator out(_output, window);

        execute_window_loop(window, [&](const Coordinates & id)
        {
            int32x4x4_t in_s32 =
            {
                {
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 0),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 4),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 8),
                    vld1q_s32(reinterpret_cast<const int32_t *>(mm_result.ptr()) + 12)
                }
            };

            // Add the offset terms to GEMM's result
            in_s32.val[0] = vaddq_s32(in_s32.val[0], c_offset_s32);
            in_s32.val[1] = vaddq_s32(in_s32.val[1], c_offset_s32);
            in_s32.val[2] = vaddq_s32(in_s32.val[2], c_offset_s32);
            in_s32.val[3] = vaddq_s32(in_s32.val[3], c_offset_s32);

            // Multiply by c_mult_int
            in_s32.val[0] = vmulq_n_s32(in_s32.val[0], _c_mult_int);
            in_s32.val[1] = vmulq_n_s32(in_s32.val[1], _c_mult_int);
            in_s32.val[2] = vmulq_n_s32(in_s32.val[2], _c_mult_int);
            in_s32.val[3] = vmulq_n_s32(in_s32.val[3], _c_mult_int);

            // Shift final result (negative value shift right)
            in_s32.val[0] = vshlq_s32(in_s32.val[0], shift_s32);
            in_s32.val[1] = vshlq_s32(in_s32.val[1], shift_s32);
            in_s32.val[2] = vshlq_s32(in_s32.val[2], shift_s32);
            in_s32.val[3] = vshlq_s32(in_s32.val[3], shift_s32);

            // Convert S32 to U16
            const int16x8x2_t in_u16 =
            {
                {
                    vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                    vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
                }
            };

            // Convert U16 to U8
            const uint8x16_t out_u8 = vcombine_u8(vqmovun_s16(in_u16.val[0]), vqmovun_s16(in_u16.val[1]));

            vst1q_u8(out.ptr(), out_u8);
        },
        mm_result, out);
    }
}

NEGEMMLowpFinalizeKernel::NEGEMMLowpFinalizeKernel()
    : _func(nullptr), _vector_sum_col(nullptr), _vector_sum_row(nullptr), _mm_result(nullptr), _output(nullptr), _a_offset(0), _b_offset(0), _c_offset(0), _k_offset(0), _c_mult_int(0), _shift(0),
      _slide_vector_sum_col(true)
{
}

void NEGEMMLowpFinalizeKernel::configure(const ITensor *vector_sum_col, const ITensor *vector_sum_row, const ITensor *mm_result, ITensor *output, int32_t num_mtx_a_cols, int32_t a_offset,
                                         int32_t b_offset,
                                         int32_t c_offset, int32_t c_mult_int, int32_t shift)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mm_result, 1, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);

    TensorShape mm_result_shape = mm_result->info()->tensor_shape();
    TensorShape output_shape    = output->info()->tensor_shape();

    mm_result_shape.collapse(2);
    output_shape.collapse(2);

    ARM_COMPUTE_ERROR_ON_MSG(mm_result_shape[2] != output_shape[2], "mm_result tensor must have the same number of batches of output tensor");

    // If a_offset == 0, vector_sum_col can be a nullptr
    if(a_offset != 0)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_col, 1, DataType::S32);
        ARM_COMPUTE_ERROR_ON(vector_sum_col->info()->dimension(0) != mm_result->info()->dimension(0));

        TensorShape vector_sum_col_shape = vector_sum_col->info()->tensor_shape();
        vector_sum_col_shape.collapse(1);

        // Check if vector_sum_col_shape should be slidden or not
        // Don't slide vector_sum_col_shape along the y dimension if vector_sum_col_shape has just 1 dimension and vector_sum_row_shape more than 1
        // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
        _slide_vector_sum_col = vector_sum_col_shape[1] != 1;
    }

    // If b_offset == 0, vector_sum_row can be a nullptr
    if(b_offset != 0)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_row, 1, DataType::S32);
        ARM_COMPUTE_ERROR_ON(vector_sum_row->info()->dimension(0) != mm_result->info()->dimension(1));

        TensorShape vector_sum_row_shape = vector_sum_row->info()->tensor_shape();
        vector_sum_row_shape.collapse(1);

        ARM_COMPUTE_ERROR_ON_MSG(vector_sum_row_shape[1] != output_shape[2], "mm_result tensor must have the same number of batches of output tensor");

        if(a_offset != 0)
        {
            TensorShape vector_sum_col_shape = vector_sum_col->info()->tensor_shape();
            vector_sum_col_shape.collapse(1);

            ARM_COMPUTE_ERROR_ON_MSG(vector_sum_col_shape[1] != 1
                                     && vector_sum_col_shape[1] != vector_sum_row_shape[1],
                                     "vector_sum_col tensor must have the same number of batches of vector_sum_row_shape or the number of batches must be set to 1");
        }
    }

    _vector_sum_col = vector_sum_col;
    _vector_sum_row = vector_sum_row;
    _mm_result      = mm_result;
    _output         = output;
    _a_offset       = a_offset;
    _b_offset       = b_offset;
    _k_offset       = a_offset * b_offset * num_mtx_a_cols;
    _c_offset       = c_offset;
    _c_mult_int     = c_mult_int;
    _shift          = shift;

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal mm_result_access(mm_result->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_result_access(output->info(), 0, num_elems_processed_per_iteration);

    // Accordingly with a_offset and b_offset, we can have 4 cases:
    // a_offset != 0 && b_offset != 0
    // a_offset  = 0 && b_offset != 0
    // a_offset != 0 && b_offset  = 0
    // a_offset  = 0 && b_offset  = 0
    if(a_offset != 0 && b_offset != 0)
    {
        // Set the function to use
        _func = &NEGEMMLowpFinalizeKernel::finalize<true, true>;

        AccessWindowStatic     vector_sum_row_access(vector_sum_row->info(), 0, 0, vector_sum_row->info()->dimension(0), 0);
        AccessWindowHorizontal vector_sum_col_access(vector_sum_col->info(), 0, num_elems_processed_per_iteration);

        update_window_and_padding(win,
                                  vector_sum_col_access,
                                  vector_sum_row_access,
                                  mm_result_access,
                                  output_result_access);
    }
    else if(a_offset == 0 && b_offset != 0)
    {
        // Set the function to use
        _func = &NEGEMMLowpFinalizeKernel::finalize<false, true>;

        AccessWindowStatic vector_sum_row_access(vector_sum_row->info(), 0, 0, vector_sum_row->info()->dimension(0), 0);

        update_window_and_padding(win,
                                  vector_sum_row_access,
                                  mm_result_access,
                                  output_result_access);
    }
    else if(a_offset != 0 && b_offset == 0)
    {
        // Set the function to use
        _func = &NEGEMMLowpFinalizeKernel::finalize<true, false>;

        AccessWindowHorizontal vector_sum_col_access(vector_sum_col->info(), 0, num_elems_processed_per_iteration);

        update_window_and_padding(win,
                                  vector_sum_col_access,
                                  mm_result_access,
                                  output_result_access);
    }
    else
    {
        // Set the function to use
        _func = &NEGEMMLowpFinalizeKernel::finalize<false, false>;

        update_window_and_padding(win,
                                  mm_result_access,
                                  output_result_access);
    }

    output_result_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEGEMMLowpFinalizeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (this->*_func)(window);
}
