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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpReductionKernel.h"

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

INEGEMMLowpReductionKernel::INEGEMMLowpReductionKernel()
    : _input(), _output(), _k(0), _is_reshaped(false)
{
}

void NEGEMMLowpMatrixAReductionKernel::configure(const ITensor *mtx_a_interleaved4x4, ITensor *vector_sum_row, int32_t num_mtx_a_cols, bool is_interleaved4x4)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mtx_a_interleaved4x4, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_row, 1, DataType::S32);

    _input       = mtx_a_interleaved4x4;
    _output      = vector_sum_row;
    _k           = num_mtx_a_cols;
    _is_reshaped = is_interleaved4x4;

    const unsigned int num_elems_processed_per_iteration = _is_reshaped ? 4 : 1;

    // Configure kernel window
    Window win = calculate_max_window(*_output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowStatic     input_access(_input->info(), 0, 0, ceil_to_multiple(_input->info()->dimension(0), 16), _input->info()->dimension(1));
    AccessWindowHorizontal output_access(_output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              input_access,
                              output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), _output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEGEMMLowpMatrixAReductionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window collapsed_window = window.collapse_if_possible(IKernel::window(), Window::DimY);

    Window win_input(collapsed_window);
    win_input.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_input.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_input.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Iterator in(_input, win_input);
    Iterator out(_output, collapsed_window);

    if(_is_reshaped)
    {
        execute_window_loop(collapsed_window, [&](const Coordinates & id)
        {
            // Note: Since the input is unsigned char, we can safely use unsigned int for the accumulation
            uint32x4_t sum_row = vdupq_n_u32(0);

            const uint8_t *matrix_a = in.ptr() + (id.x() / 4) * _input->info()->strides_in_bytes()[1] + id.y() * _input->info()->strides_in_bytes()[2];

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_a));
#endif /* __arm__ */

            int i = 0;
            // This for loop performs 4 accumulations
            for(; i <= (_k - 4); i += 4)
            {
                const uint8x16_t a0_u8 = vld1q_u8(matrix_a + i * 4);

                // Convert U8 to U16
                uint16x4x4_t a0_u16 =
                {
                    {
                        vget_low_u16(vmovl_u8(vget_low_u8(a0_u8))),
                        vget_high_u16(vmovl_u8(vget_low_u8(a0_u8))),
                        vget_low_u16(vmovl_u8(vget_high_u8(a0_u8))),
                        vget_high_u16(vmovl_u8(vget_high_u8(a0_u8)))
                    }
                };

                // Accumulate to U16
                a0_u16.val[0] = vadd_u16(a0_u16.val[0], a0_u16.val[1]);
                a0_u16.val[0] = vadd_u16(a0_u16.val[0], a0_u16.val[2]);
                a0_u16.val[0] = vadd_u16(a0_u16.val[0], a0_u16.val[3]);

                // Accumulate to U32
                sum_row = vaddw_u16(sum_row, a0_u16.val[0]);
            }

            // This for loop performs the leftover accumulations
            for(; i < _k; ++i)
            {
                const uint8x8_t a0_u8 = vld1_u8(matrix_a + i * 4);

                // Convert U8 to U16
                const uint16x4_t a0_u16 = vget_low_u16(vmovl_u8(a0_u8));

                // Accumulate to U32
                sum_row = vaddw_u16(sum_row, a0_u16);
            }

            auto vector_sum_row = reinterpret_cast<int32_t *>(out.ptr());

            vst1q_s32(vector_sum_row, vreinterpretq_s32_u32(sum_row));
        },
        in, out);
    }
    else // it is not reshaped
    {
        execute_window_loop(collapsed_window, [&](const Coordinates & id)
        {
            // Note: Since the input is unsigned char, we can safely use unsigned int for the accumulation
            uint32x4_t   sum_row_s32 = vdupq_n_u32(0);
            unsigned int sum_row     = 0;

            const uint8_t *matrix_a = in.ptr() + id.x() * _input->info()->strides_in_bytes()[1] + +id.y() * _input->info()->strides_in_bytes()[2];

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_a));
#endif /* __arm__ */

            int i = 0;
            // This for loop performs 16 accumulations
            for(; i <= (_k - 16); i += 16)
            {
                const uint8x16_t a0_u8 = vld1q_u8(matrix_a + i);

                // Partial accumulations in U16
                const uint16x8_t tmp_sum0 = vaddl_u8(vget_low_u8(a0_u8), vget_high_u8(a0_u8));

                // Accumulate to U32
                sum_row_s32 = vaddq_u32(sum_row_s32, vpaddlq_u16(tmp_sum0));
            }

            // This for loop performs the leftover accumulations
            for(; i < _k; ++i)
            {
                sum_row += static_cast<unsigned int>(matrix_a[i]);
            }

#if defined(__aarch64__)
            // Reduction operation available on 64 bit architectures only
            sum_row += vaddvq_u32(sum_row_s32);
#else  // __aarch64__
            uint32x2_t tmp = vpadd_u32(vget_high_u32(sum_row_s32), vget_low_u32(sum_row_s32));
            tmp            = vpadd_u32(tmp, tmp);

            sum_row += vget_lane_u32(tmp, 0);
#endif // __aarch64__

            *(reinterpret_cast<int *>(out.ptr())) = static_cast<int>(sum_row);
        },
        in, out);
    }
}

void NEGEMMLowpMatrixBReductionKernel::configure(const ITensor *mtx_b_transposed1xW, ITensor *vector_sum_col, int32_t num_mtx_b_rows, bool is_transposed1xW)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mtx_b_transposed1xW, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_col, 1, DataType::S32);

    _input       = mtx_b_transposed1xW;
    _output      = vector_sum_col;
    _k           = num_mtx_b_rows;
    _is_reshaped = is_transposed1xW;

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*vector_sum_col->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowStatic     input_access(_input->info(), 0, 0, ceil_to_multiple(_input->info()->dimension(0), 16), _input->info()->dimension(1));
    AccessWindowHorizontal output_access(_output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              input_access,
                              output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), _output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEGEMMLowpMatrixBReductionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window collapsed_window = window.collapse_if_possible(IKernel::window(), Window::DimY);

    if(_is_reshaped)
    {
        Window win_input(collapsed_window);
        win_input.set(Window::DimX, Window::Dimension(0, 0, 0));
        win_input.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_input.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Iterator in(_input, win_input);
        Iterator out(_output, collapsed_window);

        execute_window_loop(collapsed_window, [&](const Coordinates & id)
        {
            // Note: Since the input is unsigned char, we can safely use unsigned int for the accumulation
            uint32x4x4_t sum_col =
            {
                {
                    vdupq_n_u32(0),
                    vdupq_n_u32(0),
                    vdupq_n_u32(0),
                    vdupq_n_u32(0)
                }
            };

            const uint8_t *matrix_b = in.ptr() + (id.x() / 16) * _input->info()->strides_in_bytes()[1] + id.y() * _input->info()->strides_in_bytes()[2];

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_b));
#endif /* __arm__ */

            int i = 0;
            for(; i < _k; ++i)
            {
                const uint8x16_t b0_u8 = vld1q_u8(matrix_b + i * 16);

                // Convert U8 to U16
                const uint16x8x2_t b0_u16 =
                {
                    {
                        vmovl_u8(vget_low_u8(b0_u8)),
                        vmovl_u8(vget_high_u8(b0_u8))
                    }
                };

                // Accumulate to U32
                sum_col =
                {
                    {
                        vaddw_u16(sum_col.val[0], vget_low_u16(b0_u16.val[0])),
                        vaddw_u16(sum_col.val[1], vget_high_u16(b0_u16.val[0])),
                        vaddw_u16(sum_col.val[2], vget_low_u16(b0_u16.val[1])),
                        vaddw_u16(sum_col.val[3], vget_high_u16(b0_u16.val[1]))
                    }
                };
            }

            auto vector_sum_col = reinterpret_cast<int32_t *>(out.ptr());

            vst1q_s32(vector_sum_col + 0, vreinterpretq_s32_u32(sum_col.val[0]));
            vst1q_s32(vector_sum_col + 4, vreinterpretq_s32_u32(sum_col.val[1]));
            vst1q_s32(vector_sum_col + 8, vreinterpretq_s32_u32(sum_col.val[2]));
            vst1q_s32(vector_sum_col + 12, vreinterpretq_s32_u32(sum_col.val[3]));
        },
        in, out);
    }
    else // it is not reshaped
    {
        const auto width_matrix_b = static_cast<int>(_input->info()->dimension(0));
        const auto in_b_stride    = static_cast<int>(_input->info()->strides_in_bytes()[1]);

        // The implementation computes 16 elements per iteration
        const int window_start_x = 16 * info.thread_id;
        const int window_step_x  = 16 * info.num_threads;
        // Make sure (window_end_x - window_start_x) is a multiple of window_step_x
        const int window_end_x = ceil_to_multiple(width_matrix_b - window_start_x, window_step_x) + window_start_x;

        Window win_out(collapsed_window);
        win_out.set(Window::DimX, Window::Dimension(window_start_x, window_end_x, window_step_x));

        Window win_in(win_out);
        win_in.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Iterator inb(_input, win_in);
        Iterator out(_output, win_out);

        execute_window_loop(win_out, [&](const Coordinates & id)
        {
            if(id.x() > width_matrix_b)
            {
                return;
            }

            // Note: Since the input is unsigned char, we can safely use unsigned int for the accumulation
            uint32x4x4_t sum_col =
            {
                {
                    vdupq_n_u32(0),
                    vdupq_n_u32(0),
                    vdupq_n_u32(0),
                    vdupq_n_u32(0)
                }
            };

            const uint8_t *matrix_b = inb.ptr() + id.y() * _input->info()->strides_in_bytes()[2];

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_b));
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_b + in_b_stride));
#endif /* __arm__ */

            int i = 0;
            // This for loop performs 4 accumulations
            for(; i <= (_k - 4); i += 4)
            {
                const uint8x16_t b0_u8 = vld1q_u8(matrix_b + 0 * in_b_stride);
                const uint8x16_t b1_u8 = vld1q_u8(matrix_b + 1 * in_b_stride);
                const uint8x16_t b2_u8 = vld1q_u8(matrix_b + 2 * in_b_stride);
                const uint8x16_t b3_u8 = vld1q_u8(matrix_b + 3 * in_b_stride);

#if __arm__
                asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 1 * in_b_stride));
                asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 2 * in_b_stride));
                asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 3 * in_b_stride));
                asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 4 * in_b_stride));
#endif /* __arm__ */

                // Partial accumulation in u16
                uint16x8x2_t tmp_sum =
                {
                    {
                        vdupq_n_u16(0),
                        vdupq_n_u16(0)
                    }
                };

                tmp_sum.val[0] = vaddw_u8(tmp_sum.val[0], vget_low_u8(b0_u8));
                tmp_sum.val[0] = vaddw_u8(tmp_sum.val[0], vget_low_u8(b1_u8));
                tmp_sum.val[0] = vaddw_u8(tmp_sum.val[0], vget_low_u8(b2_u8));
                tmp_sum.val[0] = vaddw_u8(tmp_sum.val[0], vget_low_u8(b3_u8));
                tmp_sum.val[1] = vaddw_u8(tmp_sum.val[1], vget_high_u8(b0_u8));
                tmp_sum.val[1] = vaddw_u8(tmp_sum.val[1], vget_high_u8(b1_u8));
                tmp_sum.val[1] = vaddw_u8(tmp_sum.val[1], vget_high_u8(b2_u8));
                tmp_sum.val[1] = vaddw_u8(tmp_sum.val[1], vget_high_u8(b3_u8));

                // Accumulate to U32
                sum_col =
                {
                    {
                        vaddw_u16(sum_col.val[0], vget_low_u16(tmp_sum.val[0])),
                        vaddw_u16(sum_col.val[1], vget_high_u16(tmp_sum.val[0])),
                        vaddw_u16(sum_col.val[2], vget_low_u16(tmp_sum.val[1])),
                        vaddw_u16(sum_col.val[3], vget_high_u16(tmp_sum.val[1]))
                    }
                };

                matrix_b += 4 * in_b_stride;
            }

            // This for loop perfoms the leftover accumulations
            for(; i < _k; ++i)
            {
                const uint8x16_t b0_u8 = vld1q_u8(matrix_b + 0 * in_b_stride);

                // Convert U8 to U16
                const uint16x8x2_t b0_u16 =
                {
                    {
                        vmovl_u8(vget_low_u8(b0_u8)),
                        vmovl_u8(vget_high_u8(b0_u8))
                    }
                };

                // Accumulate to U32
                sum_col =
                {
                    {
                        vaddw_u16(sum_col.val[0], vget_low_u16(b0_u16.val[0])),
                        vaddw_u16(sum_col.val[1], vget_high_u16(b0_u16.val[0])),
                        vaddw_u16(sum_col.val[2], vget_low_u16(b0_u16.val[1])),
                        vaddw_u16(sum_col.val[3], vget_high_u16(b0_u16.val[1]))
                    }
                };

                matrix_b += in_b_stride;
            }

            auto vector_sum_col = reinterpret_cast<int32_t *>(out.ptr());

            vst1q_s32(vector_sum_col + 0, vreinterpretq_s32_u32(sum_col.val[0]));
            vst1q_s32(vector_sum_col + 4, vreinterpretq_s32_u32(sum_col.val[1]));
            vst1q_s32(vector_sum_col + 8, vreinterpretq_s32_u32(sum_col.val[2]));
            vst1q_s32(vector_sum_col + 12, vreinterpretq_s32_u32(sum_col.val[3]));
        },
        inb, out);
    }
}