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
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstring>

using namespace arm_compute;

void NEGEMMTranspose1xWKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(1) != std::ceil(input->info()->dimension(0) / 8.0f)) && (input->info()->data_type() == DataType::F16));
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(1) != std::ceil(input->info()->dimension(0) / 4.0f)) && (input->info()->data_type() == DataType::F32));
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(1) != std::ceil(input->info()->dimension(0) / 4.0f)) && (input->info()->data_type() == DataType::U32));

    unsigned int num_elems_processed_per_iteration = 0;
    switch(input->info()->data_type())
    {
        case DataType::F32:
        case DataType::U8:
            num_elems_processed_per_iteration = 4;
            break;
        case DataType::F16:
#ifdef ARM_COMPUTE_ENABLE_FP16
            num_elems_processed_per_iteration = 8;
            break;
#endif
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            break;
    }

    _input  = input;
    _output = output;

    // Configure kernel window
    Window                win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowTranspose output_access(output->info(), 0, 0, num_elems_processed_per_iteration, 1);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration),
                              output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));

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
     * If the input data type is F32, the output matrix will have the following shape: [ height * 4, width / 4 ]
     * If the input data type is F16, the output matrix will have the following shape: [ height * 8, width / 8 ]
     */

    /* Set window for output tensor. Set to 0 the X and Y dimensions in order to allow multi-threading implementation and future batched matrix multiplications. */
    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, window);
    Iterator out(_output, win_out);

    switch(_input->info()->data_type())
    {
        case DataType::F32:
        {
            const size_t out_stride = _output->info()->strides_in_bytes()[1] / sizeof(float);

            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto        in_ptr = reinterpret_cast<const float *>(in.ptr());
                const float32x4_t data   = vld1q_f32(in_ptr);
                /* Output address = base addr + (y * 4) + (x / 4 ) * stride */
                const auto out_ptr = reinterpret_cast<float *>(out.ptr()) + (id.y() << 2) + (id.x() >> 2) * out_stride;
                vst1q_f32(out_ptr, data);
            },
            in, out);
            break;
        }
        case DataType::U8:
        {
            const size_t out_stride = _output->info()->strides_in_bytes()[1] / sizeof(uint8_t);
            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto in_ptr = reinterpret_cast<const uint8_t *>(in.ptr());
                /* Output address = base addr + (y * 4) + (x / 4 ) * stride */
                const auto out_ptr = reinterpret_cast<uint8_t *>(out.ptr()) + (id.y() << 2) + (id.x() >> 2) * out_stride;
                std::copy_n(in_ptr, 4, out_ptr);
            },
            in, out);
            break;
        }

        case DataType::F16:
#ifdef ARM_COMPUTE_ENABLE_FP16
            {
                const size_t out_stride = _output->info()->strides_in_bytes()[1] / sizeof(float16_t);

                execute_window_loop(window, [&](const Coordinates & id)
                {
                    const auto in_ptr = reinterpret_cast<const float16_t *>(in.ptr());
                    // Output address = base addr + (y * 8) + (x / 8 ) * stride
                    float16_t *out_ptr = reinterpret_cast<float16_t *>(out.ptr()) + (id.y() << 3) + (id.x() >> 3) * out_stride;
                    vst1q_f16(out_ptr, vld1q_f16(in_ptr));
                },
                in, out);
                break;
            }
#endif
        default:
        {
            ARM_COMPUTE_ERROR("Data type not supported");
            break;
        }
    }
}
