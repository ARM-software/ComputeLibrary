/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"

#include "arm_compute/core/AccessWindowTranspose.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstring>

using namespace arm_compute;

void NEGEMMTranspose1xWKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::U8, DataType::S8, DataType::U16, DataType::S16, DataType::U32, DataType::S32, DataType::F16,
                                                  DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    TensorShape  output_shape{ input->info()->tensor_shape() };
    const size_t transpose_w = 16 / input->info()->element_size();
    output_shape.set(0, input->info()->dimension(1) * transpose_w);
    output_shape.set(1, static_cast<size_t>(std::ceil((input->info()->dimension(0) / static_cast<float>(transpose_w)))));

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();
    const int          scale_x                           = num_elems_processed_per_iteration;

    _input  = input;
    _output = output;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    ARM_COMPUTE_ERROR_ON_MSG((win.x().end() / scale_x) == 0, "Transposed shape would be 0 in the second dimension");

    AccessWindowTranspose output_access(output->info(), 0, 0, num_elems_processed_per_iteration, 1, scale_x, 1.f / scale_x);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration),
                              output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), input->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEGEMMTranspose1xWKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    /*
     * Following an example of how the transposition1xW works when the input data type is F32
     *
     *         |a00 a01 a02 a03|
     *         |a10 a11 a12 a13|
     *         |a20 a21 a22 a23| = | a00 a01 a02 a03 || a10 a11 a12 a13 || a20 a21 a22 a23 || a30 a31 a32 a33 |
     *         |a30 a31 a32 a33|
     *
     * The output matrix will have the following shape: [ height * W, ceil(width / W) ], where W = (16 / element size of the tensor)
     */

    // Set window for output tensor. Set to 0 the X and Y dimensions in order to allow multi-threading implementation and future batched matrix multiplications
    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, window);
    Iterator out(_output, win_out);

    switch(_input->info()->element_size())
    {
        case 1:
        {
            const size_t out_stride = _output->info()->strides_in_bytes()[1];
            execute_window_loop(window, [&](const Coordinates & id)
            {
                // Output address = base addr + (y * 16) + (x / 16 ) * stride
                const uint8_t *in_ptr  = in.ptr();
                uint8_t *const out_ptr = out.ptr() + (id.y() << 4) + (id.x() >> 4) * out_stride;
                vst1q_u8(out_ptr, vld1q_u8(in_ptr));
            },
            in, out);
            break;
        }
        case 2:
        {
            const size_t out_stride = _output->info()->strides_in_bytes()[1] / sizeof(int16_t);
            execute_window_loop(window, [&](const Coordinates & id)
            {
                // Output address = base addr + (y * 8) + (x / 8 ) * stride
                const auto in_ptr  = reinterpret_cast<const uint16_t *>(in.ptr());
                const auto out_ptr = reinterpret_cast<uint16_t *>(out.ptr()) + (id.y() << 3) + (id.x() >> 3) * out_stride;
                vst1q_u16(out_ptr, vld1q_u16(in_ptr));
            },
            in, out);
            break;
        }
        case 4:
        {
            const size_t out_stride = _output->info()->strides_in_bytes()[1] / sizeof(float);
            execute_window_loop(window, [&](const Coordinates & id)
            {
                // Output address = base addr + (y * 4) + (x / 4 ) * stride
                const auto in_ptr  = reinterpret_cast<const uint32_t *>(in.ptr());
                const auto out_ptr = reinterpret_cast<uint32_t *>(out.ptr()) + (id.y() << 2) + (id.x() >> 2) * out_stride;
                vst1q_u32(out_ptr, vld1q_u32(in_ptr));
            },
            in, out);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
        }
    }
}
