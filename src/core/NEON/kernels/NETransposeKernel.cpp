/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NETransposeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/AccessWindowTranspose.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
TensorShape transposed_tensor_shape(const TensorShape &in)
{
    TensorShape  output_shape{ in };
    const size_t w_out = in[1];
    const size_t h_out = in[0];
    output_shape.set(0, w_out);
    output_shape.set(1, h_out);

    return output_shape;
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use Neon FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);

    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info = input->clone()->set_tensor_shape(transposed_tensor_shape(input->tensor_shape()));

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
unsigned int num_elems_processed(size_t element_size)
{
    switch(element_size)
    {
        case 1:
            return 8;
        case 2:
        case 4:
            return 4;
        default:
            break;
    }

    ARM_COMPUTE_ERROR("Element size not supported");
}

void transpose_8bit_elements(const ITensor *in, ITensor *out, const Window &window)
{
    const int    window_step_x            = 8;
    const int    window_step_y            = 8;
    const int    window_start_x           = window.x().start();
    const int    window_end_x             = window.x().end();
    const int    window_start_y           = window.y().start();
    const int    window_end_y             = std::min(window.y().end(), static_cast<int>(in->info()->dimension(1)));
    const int    window_end_y_multiple_of = ((window_end_y - window_start_y) / window_step_y) * window_step_y;
    const size_t input_stride_in_bytes    = in->info()->strides_in_bytes()[1];
    const size_t output_stride_in_bytes   = out->info()->strides_in_bytes()[1];

    // Check if we need a left-over loop for the y dimension
    bool left_over_loop_y = (((window_end_y - window_start_y) % window_step_y) != 0);

    Window window_in(window);
    window_in.set(Window::DimX, Window::Dimension(0, 1, 1));
    if(left_over_loop_y)
    {
        // Check if window_end_y_multiple_of is greater than window_start_y
        if(window_end_y_multiple_of > window_start_y)
        {
            window_in.set(Window::DimY, Window::Dimension(window_start_y, window_end_y_multiple_of, window_step_y));
        }
        else
        {
            window_in.set(Window::DimY, Window::Dimension(0, 0, 1));
        }
    }

    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator output(out, window_out);

    // Run the Neon path if and only if the input is not a row-vector
    if(in->info()->dimension(1) != 1)
    {
        Iterator input(in, window_in);
        execute_window_loop(window_in, [&](const Coordinates & id)
        {
            // Compute 8x8 elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const uint8x8_t row0 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x + 0 * input_stride_in_bytes));
                const uint8x8_t row1 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x + 1 * input_stride_in_bytes));
                const uint8x8_t row2 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x + 2 * input_stride_in_bytes));
                const uint8x8_t row3 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x + 3 * input_stride_in_bytes));
                const uint8x8_t row4 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x + 4 * input_stride_in_bytes));
                const uint8x8_t row5 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x + 5 * input_stride_in_bytes));
                const uint8x8_t row6 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x + 6 * input_stride_in_bytes));
                const uint8x8_t row7 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x + 7 * input_stride_in_bytes));

                // Transpose 2x2
                const uint8x8x2_t k0_u8 = vtrn_u8(row0, row1);
                const uint8x8x2_t k1_u8 = vtrn_u8(row2, row3);
                const uint8x8x2_t k2_u8 = vtrn_u8(row4, row5);
                const uint8x8x2_t k3_u8 = vtrn_u8(row6, row7);

                // Transpose 4x4
                const uint16x4x2_t k0_u16 = vtrn_u16(vreinterpret_u16_u8(k0_u8.val[0]), vreinterpret_u16_u8(k1_u8.val[0]));
                const uint16x4x2_t k1_u16 = vtrn_u16(vreinterpret_u16_u8(k0_u8.val[1]), vreinterpret_u16_u8(k1_u8.val[1]));
                const uint16x4x2_t k2_u16 = vtrn_u16(vreinterpret_u16_u8(k2_u8.val[0]), vreinterpret_u16_u8(k3_u8.val[0]));
                const uint16x4x2_t k3_u16 = vtrn_u16(vreinterpret_u16_u8(k2_u8.val[1]), vreinterpret_u16_u8(k3_u8.val[1]));

                // Transpose 8x8
                const uint32x2x2_t k0_u32 = vtrn_u32(vreinterpret_u32_u16(k0_u16.val[0]), vreinterpret_u32_u16(k2_u16.val[0]));
                const uint32x2x2_t k1_u32 = vtrn_u32(vreinterpret_u32_u16(k0_u16.val[1]), vreinterpret_u32_u16(k2_u16.val[1]));
                const uint32x2x2_t k2_u32 = vtrn_u32(vreinterpret_u32_u16(k1_u16.val[0]), vreinterpret_u32_u16(k3_u16.val[0]));
                const uint32x2x2_t k3_u32 = vtrn_u32(vreinterpret_u32_u16(k1_u16.val[1]), vreinterpret_u32_u16(k3_u16.val[1]));

                // Compute destination address
                const size_t dst_offset_in_bytes = id.y() * sizeof(uint8_t) + x * output_stride_in_bytes;

                vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 0 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k0_u32.val[0])));
                vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 1 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k2_u32.val[0])));
                vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 2 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k1_u32.val[0])));
                vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 3 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k3_u32.val[0])));
                vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 4 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k0_u32.val[1])));
                vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 5 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k2_u32.val[1])));
                vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 6 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k1_u32.val[1])));
                vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 7 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k3_u32.val[1])));
            }

            // Compute left-over elements along the x dimension (1x8)
            for(; x < window_end_x; ++x)
            {
                const uint8_t val0 = *(input.ptr() + x + 0 * input_stride_in_bytes);
                const uint8_t val1 = *(input.ptr() + x + 1 * input_stride_in_bytes);
                const uint8_t val2 = *(input.ptr() + x + 2 * input_stride_in_bytes);
                const uint8_t val3 = *(input.ptr() + x + 3 * input_stride_in_bytes);
                const uint8_t val4 = *(input.ptr() + x + 4 * input_stride_in_bytes);
                const uint8_t val5 = *(input.ptr() + x + 5 * input_stride_in_bytes);
                const uint8_t val6 = *(input.ptr() + x + 6 * input_stride_in_bytes);
                const uint8_t val7 = *(input.ptr() + x + 7 * input_stride_in_bytes);

                uint8x8_t result = vdup_n_u8(0);
                result           = vset_lane_u8(val0, result, 0);
                result           = vset_lane_u8(val1, result, 1);
                result           = vset_lane_u8(val2, result, 2);
                result           = vset_lane_u8(val3, result, 3);
                result           = vset_lane_u8(val4, result, 4);
                result           = vset_lane_u8(val5, result, 5);
                result           = vset_lane_u8(val6, result, 6);
                result           = vset_lane_u8(val7, result, 7);

                // Compute destination address
                const size_t dst_offset_in_bytes = id.y() * sizeof(uint8_t) + x * output_stride_in_bytes;

                vst1_u8(output.ptr() + dst_offset_in_bytes, result);
            }
        },
        input, output);
    }

    if(left_over_loop_y)
    {
        window_in.set(Window::DimX, Window::Dimension(window.x().start(), window.x().end(), 1));
        window_in.set(Window::DimY, Window::Dimension(window_end_y_multiple_of, window_end_y, 1));

        Iterator input(in, window_in);
        Iterator output(out, window_out);

        // Compute left-over elements along the y dimension (1x1)
        execute_window_loop(window_in, [&](const Coordinates & id)
        {
            const uint8_t val0 = *input.ptr();

            // Compute destination address
            const size_t dst_offset_in_bytes = id.y() * sizeof(uint8_t) + id.x() * output_stride_in_bytes;

            *(output.ptr() + dst_offset_in_bytes) = val0;
        },
        input, output);
    }
}

void transpose_16bit_elements(const ITensor *in, ITensor *out, const Window &window)
{
    const int    window_step_x            = 4;
    const int    window_step_y            = 4;
    const int    window_start_x           = window.x().start();
    const int    window_end_x             = window.x().end();
    const int    window_start_y           = window.y().start();
    const int    window_end_y             = std::min(window.y().end(), static_cast<int>(in->info()->dimension(1)));
    const int    window_end_y_multiple_of = ((window_end_y - window_start_y) / window_step_y) * window_step_y;
    const size_t input_stride_in_bytes    = in->info()->strides_in_bytes()[1];
    const size_t output_stride_in_bytes   = out->info()->strides_in_bytes()[1];

    // Check if we need a left-over loop for the y dimension
    bool left_over_loop_y = (((window_end_y - window_start_y) % window_step_y) != 0);

    Window window_in(window);
    window_in.set(Window::DimX, Window::Dimension(0, 1, 1));
    if(left_over_loop_y)
    {
        // Check if window_end_y_multiple_of is greater than window_start_y
        if(window_end_y_multiple_of > window_start_y)
        {
            window_in.set(Window::DimY, Window::Dimension(window_start_y, window_end_y_multiple_of, window_step_y));
        }
        else
        {
            window_in.set(Window::DimY, Window::Dimension(0, 0, 1));
        }
    }

    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator output(out, window_out);

    // Run the Neon path if and only if the input is not a row-vector
    if(in->info()->dimension(1) != 1)
    {
        Iterator input(in, window_in);
        execute_window_loop(window_in, [&](const Coordinates & id)
        {
            // Compute 4x4 elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const uint16x4_t row0 = vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() + 0 * input_stride_in_bytes) + x);
                const uint16x4_t row1 = vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() + 1 * input_stride_in_bytes) + x);
                const uint16x4_t row2 = vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() + 2 * input_stride_in_bytes) + x);
                const uint16x4_t row3 = vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() + 3 * input_stride_in_bytes) + x);

                // Transpose 2x2
                const uint16x4x2_t k0_u16 = vtrn_u16(row0, row1);
                const uint16x4x2_t k1_u16 = vtrn_u16(row2, row3);

                // Transpose 4x4
                const uint32x2x2_t k0_u32 = vtrn_u32(vreinterpret_u32_u16(k0_u16.val[0]), vreinterpret_u32_u16(k1_u16.val[0]));
                const uint32x2x2_t k1_u32 = vtrn_u32(vreinterpret_u32_u16(k0_u16.val[1]), vreinterpret_u32_u16(k1_u16.val[1]));

                // Compute destination address
                const size_t dst_offset_in_bytes = id.y() * sizeof(uint16_t) + x * output_stride_in_bytes;

                vst1_u16(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes + 0 * output_stride_in_bytes), vreinterpret_u16_u32(k0_u32.val[0]));
                vst1_u16(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes + 1 * output_stride_in_bytes), vreinterpret_u16_u32(k1_u32.val[0]));
                vst1_u16(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes + 2 * output_stride_in_bytes), vreinterpret_u16_u32(k0_u32.val[1]));
                vst1_u16(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes + 3 * output_stride_in_bytes), vreinterpret_u16_u32(k1_u32.val[1]));
            }

            // Compute left-over elements (1x4)
            for(; x < window_end_x; ++x)
            {
                const uint16_t val0 = *(reinterpret_cast<uint16_t *>(input.ptr() + 0 * input_stride_in_bytes) + x);
                const uint16_t val1 = *(reinterpret_cast<uint16_t *>(input.ptr() + 1 * input_stride_in_bytes) + x);
                const uint16_t val2 = *(reinterpret_cast<uint16_t *>(input.ptr() + 2 * input_stride_in_bytes) + x);
                const uint16_t val3 = *(reinterpret_cast<uint16_t *>(input.ptr() + 3 * input_stride_in_bytes) + x);

                uint16x4_t result = vdup_n_u16(0);
                result            = vset_lane_u16(val0, result, 0);
                result            = vset_lane_u16(val1, result, 1);
                result            = vset_lane_u16(val2, result, 2);
                result            = vset_lane_u16(val3, result, 3);

                // Compute destination address
                const size_t dst_offset_in_bytes = id.y() * sizeof(uint16_t) + x * output_stride_in_bytes;

                vst1_u16(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes), result);
            }
        },
        input, output);
    }

    if(left_over_loop_y)
    {
        window_in.set(Window::DimX, Window::Dimension(window.x().start(), window.x().end(), 1));
        window_in.set(Window::DimY, Window::Dimension(window_end_y_multiple_of, window_end_y, 1));

        Iterator input(in, window_in);
        Iterator output(out, window_out);

        // Compute left-over elements along the y dimension (1x1)
        execute_window_loop(window_in, [&](const Coordinates & id)
        {
            const uint16_t val0 = *(reinterpret_cast<uint16_t *>(input.ptr()));

            // Compute destination address
            const size_t dst_offset_in_bytes = id.y() * sizeof(uint16_t) + id.x() * output_stride_in_bytes;

            *(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes)) = val0;
        },
        input, output);
    }
}

void transpose_32bit_elements(const ITensor *in, ITensor *out, const Window &window)
{
    const int    window_step_x            = 4;
    const int    window_step_y            = 4;
    const int    window_start_x           = window.x().start();
    const int    window_end_x             = window.x().end();
    const int    window_start_y           = window.y().start();
    const int    window_end_y             = std::min(window.y().end(), static_cast<int>(in->info()->dimension(1)));
    const int    window_end_y_multiple_of = ((window_end_y - window_start_y) / window_step_y) * window_step_y;
    const size_t input_stride_in_bytes    = in->info()->strides_in_bytes()[1];
    const size_t output_stride_in_bytes   = out->info()->strides_in_bytes()[1];

    // Check if we need a left-over loop for the y dimension
    bool left_over_loop_y = (((window_end_y - window_start_y) % window_step_y) != 0);

    Window window_in(window);
    window_in.set(Window::DimX, Window::Dimension(0, 1, 1));
    if(left_over_loop_y)
    {
        // Check if window_end_y_multiple_of is greater than window_start_y
        if(window_end_y_multiple_of > window_start_y)
        {
            window_in.set(Window::DimY, Window::Dimension(window_start_y, window_end_y_multiple_of, window_step_y));
        }
        else
        {
            window_in.set(Window::DimY, Window::Dimension(0, 0, 1));
        }
    }

    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator output(out, window_out);

    // Run the Neon path if and only if the input is not a row-vector
    if(in->info()->dimension(1) != 1)
    {
        Iterator input(in, window_in);
        execute_window_loop(window_in, [&](const Coordinates & id)
        {
            // Compute 4x4 elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const uint32x4_t row0 = vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 0 * input_stride_in_bytes) + x);
                const uint32x4_t row1 = vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 1 * input_stride_in_bytes) + x);
                const uint32x4_t row2 = vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 2 * input_stride_in_bytes) + x);
                const uint32x4_t row3 = vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 3 * input_stride_in_bytes) + x);

                // Transpose 2x2
                const uint32x2x2_t k0_u32 = vtrn_u32(vget_low_u32(row0), vget_low_u32(row1));
                const uint32x2x2_t k1_u32 = vtrn_u32(vget_high_u32(row2), vget_high_u32(row3));
                const uint32x2x2_t k2_u32 = vtrn_u32(vget_high_u32(row0), vget_high_u32(row1));
                const uint32x2x2_t k3_u32 = vtrn_u32(vget_low_u32(row2), vget_low_u32(row3));

                // Compute destination address
                const size_t dst_offset_in_bytes = id.y() * sizeof(uint32_t) + x * output_stride_in_bytes;

                // Swap block 01 with block 10 and store
                vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 0 * output_stride_in_bytes), vcombine_u32(k0_u32.val[0], k3_u32.val[0]));
                vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 1 * output_stride_in_bytes), vcombine_u32(k0_u32.val[1], k3_u32.val[1]));
                vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 2 * output_stride_in_bytes), vcombine_u32(k2_u32.val[0], k1_u32.val[0]));
                vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 3 * output_stride_in_bytes), vcombine_u32(k2_u32.val[1], k1_u32.val[1]));
            }

            // Compute left-over elements (1x4)
            for(; x < window_end_x; ++x)
            {
                const uint32_t val0 = *(reinterpret_cast<uint32_t *>(input.ptr() + 0 * input_stride_in_bytes) + x);
                const uint32_t val1 = *(reinterpret_cast<uint32_t *>(input.ptr() + 1 * input_stride_in_bytes) + x);
                const uint32_t val2 = *(reinterpret_cast<uint32_t *>(input.ptr() + 2 * input_stride_in_bytes) + x);
                const uint32_t val3 = *(reinterpret_cast<uint32_t *>(input.ptr() + 3 * input_stride_in_bytes) + x);

                uint32x4_t result = vdupq_n_u32(0);
                result            = vsetq_lane_u32(val0, result, 0);
                result            = vsetq_lane_u32(val1, result, 1);
                result            = vsetq_lane_u32(val2, result, 2);
                result            = vsetq_lane_u32(val3, result, 3);

                // Compute destination address
                const size_t dst_offset_in_bytes = id.y() * sizeof(uint32_t) + x * output_stride_in_bytes;

                vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes), result);
            }
        },
        input, output);
    }

    if(left_over_loop_y)
    {
        window_in.set(Window::DimX, Window::Dimension(window.x().start(), window.x().end(), 1));
        window_in.set(Window::DimY, Window::Dimension(window_end_y_multiple_of, window_end_y, 1));

        Iterator input(in, window_in);
        Iterator output(out, window_out);

        // Compute left-over elements along the y dimension (1x1)
        execute_window_loop(window_in, [&](const Coordinates & id)
        {
            const uint32_t val0 = *(reinterpret_cast<uint32_t *>(input.ptr()));

            // Compute destination address
            const size_t dst_offset_in_bytes = id.y() * sizeof(uint32_t) + id.x() * output_stride_in_bytes;

            *(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes)) = val0;
        },
        input, output);
    }
}
} // namespace

Status NETransposeKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    return Status{};
}

NETransposeKernel::NETransposeKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr)
{
}

void NETransposeKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(transposed_tensor_shape(input->info()->tensor_shape())));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    switch(input->info()->element_size())
    {
        case 1:
            _func = &transpose_8bit_elements;
            break;
        case 2:
            _func = &transpose_16bit_elements;
            break;
        case 4:
            _func = &transpose_32bit_elements;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }

    // Configure kernel window
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    // Note: This kernel performs 16 elements per iteration.
    // However, since we use a left-over for loop on both dimensions (X and Y), we cannot have any read or write out of memory
    // For this reason num_elems_processed_per_iteration_x is set to 1
    const unsigned int num_elems_processed_per_iteration_x = 1;
    const unsigned int num_elems_processed_per_iteration_y = num_elems_processed(input->info()->element_size());

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    INEKernel::configure(win);
}

void NETransposeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _output, window);
}
