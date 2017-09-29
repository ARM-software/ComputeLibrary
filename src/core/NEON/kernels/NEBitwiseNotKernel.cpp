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
#include "arm_compute/core/NEON/kernels/NEBitwiseNotKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
inline void bitwise_not_U8_U8(const uint8_t *__restrict input, uint8_t *__restrict output)
{
    const uint8x16_t val0 = vld1q_u8(input);

    vst1q_u8(output, vmvnq_u8(val0));
}
} // namespace

NEBitwiseNotKernel::NEBitwiseNotKernel()
    : _input(nullptr), _output(nullptr)
{
}

void NEBitwiseNotKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    set_format_if_unknown(*output->info(), Format::U8);
    set_format_if_unknown(*input->info(), Format::U8);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input  = input;
    _output = output;

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration), output_access);
    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

void NEBitwiseNotKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        bitwise_not_U8_U8(input.ptr(), output.ptr());
    },
    input, output);
}
