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
#include "arm_compute/core/NEON/kernels/NETransposeKernel.h"

#include "arm_compute/core/AccessWindowTranspose.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
void transpose_8bit_elements(const ITensor *in, ITensor *out, const Window &window)
{
    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator input(in, window);
    Iterator output(out, window_out);

    const size_t input_stride_in_bytes  = in->info()->strides_in_bytes()[1];
    const size_t output_stride_in_bytes = out->info()->strides_in_bytes()[1];

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8x8_t row0 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + 0 * input_stride_in_bytes));
        const uint8x8_t row1 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + 1 * input_stride_in_bytes));
        const uint8x8_t row2 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + 2 * input_stride_in_bytes));
        const uint8x8_t row3 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + 3 * input_stride_in_bytes));
        const uint8x8_t row4 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + 4 * input_stride_in_bytes));
        const uint8x8_t row5 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + 5 * input_stride_in_bytes));
        const uint8x8_t row6 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + 6 * input_stride_in_bytes));
        const uint8x8_t row7 = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + 7 * input_stride_in_bytes));

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
        const size_t dst_offset_in_bytes = id.y() * sizeof(uint8_t) + id.x() * output_stride_in_bytes;

        vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 0 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k0_u32.val[0])));
        vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 1 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k2_u32.val[0])));
        vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 2 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k1_u32.val[0])));
        vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 3 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k3_u32.val[0])));
        vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 4 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k0_u32.val[1])));
        vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 5 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k2_u32.val[1])));
        vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 6 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k1_u32.val[1])));
        vst1_u8(reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes + 7 * output_stride_in_bytes), vreinterpret_u8_u16(vreinterpret_u16_u32(k3_u32.val[1])));
    },
    input, output);
}

void transpose_16bit_elements(const ITensor *in, ITensor *out, const Window &window)
{
    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator input(in, window);
    Iterator output(out, window_out);

    const size_t input_stride_in_bytes  = in->info()->strides_in_bytes()[1];
    const size_t output_stride_in_bytes = out->info()->strides_in_bytes()[1];

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint16x4_t row0 = vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() + 0 * input_stride_in_bytes));
        const uint16x4_t row1 = vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() + 1 * input_stride_in_bytes));
        const uint16x4_t row2 = vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() + 2 * input_stride_in_bytes));
        const uint16x4_t row3 = vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() + 3 * input_stride_in_bytes));

        // Transpose 2x2
        const uint16x4x2_t k0_u16 = vtrn_u16(row0, row1);
        const uint16x4x2_t k1_u16 = vtrn_u16(row2, row3);

        // Transpose 4x4
        const uint32x2x2_t k0_u32 = vtrn_u32(vreinterpret_u32_u16(k0_u16.val[0]), vreinterpret_u32_u16(k1_u16.val[0]));
        const uint32x2x2_t k1_u32 = vtrn_u32(vreinterpret_u32_u16(k0_u16.val[1]), vreinterpret_u32_u16(k1_u16.val[1]));

        // Compute destination address
        const size_t dst_offset_in_bytes = id.y() * sizeof(uint16_t) + id.x() * output_stride_in_bytes;

        vst1_u16(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes + 0 * output_stride_in_bytes), vreinterpret_u16_u32(k0_u32.val[0]));
        vst1_u16(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes + 1 * output_stride_in_bytes), vreinterpret_u16_u32(k1_u32.val[0]));
        vst1_u16(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes + 2 * output_stride_in_bytes), vreinterpret_u16_u32(k0_u32.val[1]));
        vst1_u16(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes + 3 * output_stride_in_bytes), vreinterpret_u16_u32(k1_u32.val[1]));
    },
    input, output);
}

void transpose_32bit_elements(const ITensor *in, ITensor *out, const Window &window)
{
    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator input(in, window);
    Iterator output(out, window_out);

    const size_t input_stride_in_bytes  = in->info()->strides_in_bytes()[1];
    const size_t output_stride_in_bytes = out->info()->strides_in_bytes()[1];

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint32x4_t row0 = vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 0 * input_stride_in_bytes));
        const uint32x4_t row1 = vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 1 * input_stride_in_bytes));
        const uint32x4_t row2 = vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 2 * input_stride_in_bytes));
        const uint32x4_t row3 = vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 3 * input_stride_in_bytes));

        // Transpose 2x2
        const uint32x2x2_t k0_u32 = vtrn_u32(vget_low_u32(row0), vget_low_u32(row1));
        const uint32x2x2_t k1_u32 = vtrn_u32(vget_high_u32(row2), vget_high_u32(row3));
        const uint32x2x2_t k2_u32 = vtrn_u32(vget_high_u32(row0), vget_high_u32(row1));
        const uint32x2x2_t k3_u32 = vtrn_u32(vget_low_u32(row2), vget_low_u32(row3));

        // Compute destination address
        const size_t dst_offset_in_bytes = id.y() * sizeof(uint32_t) + id.x() * output_stride_in_bytes;

        // Swap block 01 with block 10 and store
        vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 0 * output_stride_in_bytes), vcombine_u32(k0_u32.val[0], k3_u32.val[0]));
        vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 1 * output_stride_in_bytes), vcombine_u32(k0_u32.val[1], k3_u32.val[1]));
        vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 2 * output_stride_in_bytes), vcombine_u32(k2_u32.val[0], k1_u32.val[0]));
        vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 3 * output_stride_in_bytes), vcombine_u32(k2_u32.val[1], k1_u32.val[1]));
    },
    input, output);
}
} // namespace

NETransposeKernel::NETransposeKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr)
{
}

void NETransposeKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QS8, DataType::U16, DataType::S16, DataType::U32, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    TensorShape  output_shape{ input->info()->tensor_shape() };
    const size_t w_out = input->info()->dimension(1);
    const size_t h_out = input->info()->dimension(0);
    output_shape.set(0, w_out);
    output_shape.set(1, h_out);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);

    _input  = input;
    _output = output;

    unsigned int num_elems_processed_per_iteration = 0;

    switch(input->info()->element_size())
    {
        case 1:
            _func                             = &transpose_8bit_elements;
            num_elems_processed_per_iteration = 8;
            break;
        case 2:
            _func                             = &transpose_16bit_elements;
            num_elems_processed_per_iteration = 4;
            break;
        case 4:
            _func                             = &transpose_32bit_elements;
            num_elems_processed_per_iteration = 4;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }

    // Configure kernel window
    Window                win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration, num_elems_processed_per_iteration));
    AccessWindowTranspose output_access(output->info(), 0, 0, num_elems_processed_per_iteration, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), 0, 0, num_elems_processed_per_iteration, num_elems_processed_per_iteration),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

void NETransposeKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _output, window);
}
