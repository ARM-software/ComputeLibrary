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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"

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
#include <tuple>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

NEGEMMLowpMatrixMultiplyKernel::NEGEMMLowpMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr), _slide_matrix_b(true)
{
}

void NEGEMMLowpMatrixMultiplyKernel::configure(const ITensor *input0, const ITensor *input1, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::QASYMM8, DataType::S8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);

    // Check if matrix B should be slidden or not
    // Don't slide matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
    // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
    TensorShape in0_shape = input0->info()->tensor_shape();
    TensorShape in1_shape = input1->info()->tensor_shape();
    TensorShape out_shape = output->info()->tensor_shape();

    in0_shape.collapse(2);
    in1_shape.collapse(2);
    out_shape.collapse(2);

    ARM_COMPUTE_ERROR_ON_MSG(in0_shape[2] != out_shape[2], "Output tensor must have the same number of batches of input0 tensor");
    ARM_COMPUTE_ERROR_ON_MSG(in1_shape[2] != 1 && in0_shape[2] != in1_shape[2], "Input1 tensor must have the same number of batches of input0 or the number of batches must be set to 1");

    _input0         = input0;
    _input1         = input1;
    _output         = output;
    _slide_matrix_b = in1_shape[2] != 1;

    constexpr unsigned int num_elems_processed_per_iteration_x = 16;
    constexpr unsigned int num_elems_processed_per_iteration_y = 4;

    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    AccessWindowStatic     in0_access(input0->info(), 0, 0, ceil_to_multiple(input0->info()->dimension(0), 8), input0->info()->dimension(1));
    AccessWindowHorizontal in1_access(input1->info(), 0, num_elems_processed_per_iteration_x);
    AccessWindowRectangle  output_access(output->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

    update_window_and_padding(win, in0_access, in1_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));
    INEKernel::configure(win);
}

void inline matrix_multiply_u8(Iterator &ina, Iterator &inb, Iterator &out, int width_b, size_t out_stride, const Window &window)
{
    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8_t *mtx_a0 = ina.ptr();
        const uint8_t *mtx_b0 = inb.ptr();

        // Note: Since the input are all positives, we can use uint32_t
        // Accumulators for the block 0
        uint32x4x4_t c0 =
        {
            {
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0)
            }
        };

        // Accumulators for the block 1
        uint32x4x4_t c1 =
        {
            {
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0)
            }
        };

        // Accumulators for the block 2
        uint32x4x4_t c2 =
        {
            {
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0)
            }
        };

        // Accumulators for the block 3
        uint32x4x4_t c3 =
        {
            {
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0)
            }
        };

        for(int k = 0; k < width_b; k += 16, mtx_a0 += 4, mtx_b0 += 16)
        {
            const uint8x8_t  a00_u8 = vld1_u8(mtx_a0);
            const uint8x16_t b00_u8 = vld1q_u8(mtx_b0);

            // Convert a00_s8 to uint16_t and get the lower part
            const uint16x4_t a00_u16 = vget_low_u16(vmovl_u8(a00_u8));

            // Convert b00_s8 to uint16_t
            const uint16x4x4_t b00_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b00_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b00_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b00_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b00_u8)))
                }
            };

            // 4x4 block 0
            c0.val[0] = vmlal_lane_u16(c0.val[0], b00_u16.val[0], a00_u16, 0);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b00_u16.val[1], a00_u16, 0);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b00_u16.val[2], a00_u16, 0);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b00_u16.val[3], a00_u16, 0);

            // 4x4 block 1
            c1.val[0] = vmlal_lane_u16(c1.val[0], b00_u16.val[0], a00_u16, 1);
            c1.val[1] = vmlal_lane_u16(c1.val[1], b00_u16.val[1], a00_u16, 1);
            c1.val[2] = vmlal_lane_u16(c1.val[2], b00_u16.val[2], a00_u16, 1);
            c1.val[3] = vmlal_lane_u16(c1.val[3], b00_u16.val[3], a00_u16, 1);

            // 4x4 block 2
            c2.val[0] = vmlal_lane_u16(c2.val[0], b00_u16.val[0], a00_u16, 2);
            c2.val[1] = vmlal_lane_u16(c2.val[1], b00_u16.val[1], a00_u16, 2);
            c2.val[2] = vmlal_lane_u16(c2.val[2], b00_u16.val[2], a00_u16, 2);
            c2.val[3] = vmlal_lane_u16(c2.val[3], b00_u16.val[3], a00_u16, 2);

            // 4x4 block 3
            c3.val[0] = vmlal_lane_u16(c3.val[0], b00_u16.val[0], a00_u16, 3);
            c3.val[1] = vmlal_lane_u16(c3.val[1], b00_u16.val[1], a00_u16, 3);
            c3.val[2] = vmlal_lane_u16(c3.val[2], b00_u16.val[2], a00_u16, 3);
            c3.val[3] = vmlal_lane_u16(c3.val[3], b00_u16.val[3], a00_u16, 3);
        }

        auto mtx_out = reinterpret_cast<int32_t *>(out.ptr());
        vst1q_s32(mtx_out + 0 * out_stride + 0, vreinterpretq_s32_u32(c0.val[0]));
        vst1q_s32(mtx_out + 0 * out_stride + 4, vreinterpretq_s32_u32(c0.val[1]));
        vst1q_s32(mtx_out + 0 * out_stride + 8, vreinterpretq_s32_u32(c0.val[2]));
        vst1q_s32(mtx_out + 0 * out_stride + 12, vreinterpretq_s32_u32(c0.val[3]));
        vst1q_s32(mtx_out + 1 * out_stride + 0, vreinterpretq_s32_u32(c1.val[0]));
        vst1q_s32(mtx_out + 1 * out_stride + 4, vreinterpretq_s32_u32(c1.val[1]));
        vst1q_s32(mtx_out + 1 * out_stride + 8, vreinterpretq_s32_u32(c1.val[2]));
        vst1q_s32(mtx_out + 1 * out_stride + 12, vreinterpretq_s32_u32(c1.val[3]));
        vst1q_s32(mtx_out + 2 * out_stride + 0, vreinterpretq_s32_u32(c2.val[0]));
        vst1q_s32(mtx_out + 2 * out_stride + 4, vreinterpretq_s32_u32(c2.val[1]));
        vst1q_s32(mtx_out + 2 * out_stride + 8, vreinterpretq_s32_u32(c2.val[2]));
        vst1q_s32(mtx_out + 2 * out_stride + 12, vreinterpretq_s32_u32(c2.val[3]));
        vst1q_s32(mtx_out + 3 * out_stride + 0, vreinterpretq_s32_u32(c3.val[0]));
        vst1q_s32(mtx_out + 3 * out_stride + 4, vreinterpretq_s32_u32(c3.val[1]));
        vst1q_s32(mtx_out + 3 * out_stride + 8, vreinterpretq_s32_u32(c3.val[2]));
        vst1q_s32(mtx_out + 3 * out_stride + 12, vreinterpretq_s32_u32(c3.val[3]));
    },
    ina, inb, out);
}

void inline matrix_multiply_s8(Iterator &ina, Iterator &inb, Iterator &out, int width_b, size_t out_stride, const Window &window)
{
    // The implementation assumes that the matrix A and Matrix B have been reshaped respectively with NEGEMMInterleave4x4 and NEGEMMTranspose1xW
    // The reshaping of the matrices helps to have a cache friendly implementation and helps to avoid the data re-arrangements needed for computing 16x4 elements per iteration
    // All the values needed for computing a single 4x4 block will be read from consecutive memory positions
    execute_window_loop(window, [&](const Coordinates & id)
    {
        auto *mtx_a0 = reinterpret_cast<const int8_t *>(ina.ptr());
        auto *mtx_b0 = reinterpret_cast<const int8_t *>(inb.ptr());

        // Note: Since the input are all positives, we can use uint32_t
        // Accumulators for the block 0
        int32x4x4_t c0 =
        {
            {
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0)
            }
        };

        // Accumulators for the block 1
        int32x4x4_t c1 =
        {
            {
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0)
            }
        };

        // Accumulators for the block 2
        int32x4x4_t c2 =
        {
            {
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0)
            }
        };

        // Accumulators for the block 3
        int32x4x4_t c3 =
        {
            {
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0)
            }
        };

        for(int k = 0; k < width_b; k += 16, mtx_a0 += 4, mtx_b0 += 16)
        {
            const int8x8_t  a00_s8 = vld1_s8(mtx_a0);
            const int8x16_t b00_s8 = vld1q_s8(mtx_b0);

            // Convert a00_s8 to uint16_t and get the lower part
            const int16x4_t a00_s16 = vget_low_s16(vmovl_s8(a00_s8));

            // Convert b00_s8 to int16_t
            const int16x4x4_t b00_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b00_s8)))
                }
            };

            // 4x4 block 0
            c0.val[0] = vmlal_lane_s16(c0.val[0], b00_s16.val[0], a00_s16, 0);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b00_s16.val[1], a00_s16, 0);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b00_s16.val[2], a00_s16, 0);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b00_s16.val[3], a00_s16, 0);

            // 4x4 block 1
            c1.val[0] = vmlal_lane_s16(c1.val[0], b00_s16.val[0], a00_s16, 1);
            c1.val[1] = vmlal_lane_s16(c1.val[1], b00_s16.val[1], a00_s16, 1);
            c1.val[2] = vmlal_lane_s16(c1.val[2], b00_s16.val[2], a00_s16, 1);
            c1.val[3] = vmlal_lane_s16(c1.val[3], b00_s16.val[3], a00_s16, 1);

            // 4x4 block 2
            c2.val[0] = vmlal_lane_s16(c2.val[0], b00_s16.val[0], a00_s16, 2);
            c2.val[1] = vmlal_lane_s16(c2.val[1], b00_s16.val[1], a00_s16, 2);
            c2.val[2] = vmlal_lane_s16(c2.val[2], b00_s16.val[2], a00_s16, 2);
            c2.val[3] = vmlal_lane_s16(c2.val[3], b00_s16.val[3], a00_s16, 2);

            // 4x4 block 3
            c3.val[0] = vmlal_lane_s16(c3.val[0], b00_s16.val[0], a00_s16, 3);
            c3.val[1] = vmlal_lane_s16(c3.val[1], b00_s16.val[1], a00_s16, 3);
            c3.val[2] = vmlal_lane_s16(c3.val[2], b00_s16.val[2], a00_s16, 3);
            c3.val[3] = vmlal_lane_s16(c3.val[3], b00_s16.val[3], a00_s16, 3);
        }

        auto mtx_out = reinterpret_cast<int32_t *>(out.ptr());
        vst1q_s32(mtx_out + 0 * out_stride + 0, c0.val[0]);
        vst1q_s32(mtx_out + 0 * out_stride + 4, c0.val[1]);
        vst1q_s32(mtx_out + 0 * out_stride + 8, c0.val[2]);
        vst1q_s32(mtx_out + 0 * out_stride + 12, c0.val[3]);
        vst1q_s32(mtx_out + 1 * out_stride + 0, c1.val[0]);
        vst1q_s32(mtx_out + 1 * out_stride + 4, c1.val[1]);
        vst1q_s32(mtx_out + 1 * out_stride + 8, c1.val[2]);
        vst1q_s32(mtx_out + 1 * out_stride + 12, c1.val[3]);
        vst1q_s32(mtx_out + 2 * out_stride + 0, c2.val[0]);
        vst1q_s32(mtx_out + 2 * out_stride + 4, c2.val[1]);
        vst1q_s32(mtx_out + 2 * out_stride + 8, c2.val[2]);
        vst1q_s32(mtx_out + 2 * out_stride + 12, c2.val[3]);
        vst1q_s32(mtx_out + 3 * out_stride + 0, c3.val[0]);
        vst1q_s32(mtx_out + 3 * out_stride + 4, c3.val[1]);
        vst1q_s32(mtx_out + 3 * out_stride + 8, c3.val[2]);
        vst1q_s32(mtx_out + 3 * out_stride + 12, c3.val[3]);
    },
    ina, inb, out);
}

void NEGEMMLowpMatrixMultiplyKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const size_t in_b_stride = _input1->info()->strides_in_bytes()[1];
    const size_t out_stride  = _output->info()->strides_in_bytes()[1] / _output->info()->element_size();

    // Set step_x and step_y for matrix A. Scale by a factor of 4 the Y range as the input interleaved matrix A has 4 times less the rows of the output matrix
    Window win_a(window);
    win_a.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_a.set(Window::DimY, Window::Dimension(window.y().start() / 4, window.y().end() / 4, 1));

    // Set step_x and step_y for matrix B. Scale by a factor of 16 the X range as the input transposed matrix A has 16 times less the columns of the output matrix
    Window win_b;
    // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
    // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
    if(_slide_matrix_b)
    {
        win_b = window;
    }
    win_b.set(Window::DimX, Window::Dimension(window.x().start() / 16, window.x().end() / 16, in_b_stride));
    win_b.set(Window::DimY, Window::Dimension(0, 0, 0));

    // The step x and step y for the output matrix has been already set using in configure()
    Iterator ina(_input0, win_a);
    Iterator inb(_input1, win_b);
    Iterator out(_output, window);

    const int width_b = _input1->info()->dimension(0);
    switch(_input0->info()->data_type())
    {
        case DataType::S8:
        {
            matrix_multiply_s8(ina, inb, out, width_b, out_stride, window);
            break;
        }
        case DataType::U8:
        case DataType::QASYMM8:
        {
            matrix_multiply_u8(ina, inb, out, width_b, out_stride, window);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported");
            break;
        }
    }
}
