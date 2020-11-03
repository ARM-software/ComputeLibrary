/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "src/core/NEON/kernels/NEBitwiseXorKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
inline void bitwise_xor_U8_U8_U8(const uint8_t *__restrict input1, const uint8_t *__restrict input2, uint8_t *__restrict output)
{
    const uint8x16_t val1 = vld1q_u8(input1);
    const uint8x16_t val2 = vld1q_u8(input2);

    vst1q_u8(output, veorq_u8(val1, val2));
}
} // namespace

NEBitwiseXorKernel::NEBitwiseXorKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void NEBitwiseXorKernel::configure(const ITensor *input1, const ITensor *input2, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    set_shape_if_empty(*output->info(), input1->info()->tensor_shape());

    set_format_if_unknown(*output->info(), Format::U8);
    set_format_if_unknown(*input1->info(), Format::U8);
    set_format_if_unknown(*input2->info(), Format::U8);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input1, input2, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input1, input2, output);

    _input1 = input1;
    _input2 = input2;
    _output = output;

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    Window                 win = calculate_max_window(*input1->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, AccessWindowHorizontal(input1->info(), 0, num_elems_processed_per_iteration),
                              AccessWindowHorizontal(input2->info(), 0, num_elems_processed_per_iteration), output_access);

    const ValidRegion valid_region = intersect_valid_regions(input1->info()->valid_region(), input2->info()->valid_region());

    output_access.set_valid_region(win, valid_region);

    INEKernel::configure(win);
}

void NEBitwiseXorKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    Iterator input1(_input1, window);
    Iterator input2(_input2, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        bitwise_xor_U8_U8_U8(input1.ptr(), input2.ptr(), output.ptr());
    },
    input1, input2, output);
}
