/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEThresholdKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>

namespace arm_compute
{
class Coordinates;

NEThresholdKernel::NEThresholdKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _threshold(0), _false_value(0), _true_value(0), _upper(0)
{
}

void NEThresholdKernel::configure(const ITensor *input, ITensor *output, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);

    _input       = input;
    _output      = output;
    _threshold   = threshold;
    _false_value = false_value;
    _true_value  = true_value;
    _upper       = upper;

    switch(type)
    {
        case ThresholdType::BINARY:
            _func = &NEThresholdKernel::run_binary;
            break;
        case ThresholdType::RANGE:
            _func = &NEThresholdKernel::run_range;
            break;
        default:
            ARM_COMPUTE_ERROR("Thresholding type not recognized");
            break;
    }

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration), output_access);
    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

inline void NEThresholdKernel::run_binary(const Window &window)
{
    const uint8x16_t threshold   = vdupq_n_u8(_threshold);
    const uint8x16_t true_value  = vdupq_n_u8(_true_value);
    const uint8x16_t false_value = vdupq_n_u8(_false_value);

    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());
        const uint8x16_t mask = vcgtq_u8(data, threshold);

        vst1q_u8(output.ptr(), vbslq_u8(mask, true_value, false_value));
    },
    input, output);
}

inline void NEThresholdKernel::run_range(const Window &window)
{
    const uint8x16_t lower_threshold = vdupq_n_u8(_threshold);
    const uint8x16_t upper_threshold = vdupq_n_u8(_upper);
    const uint8x16_t true_value      = vdupq_n_u8(_true_value);
    const uint8x16_t false_value     = vdupq_n_u8(_false_value);

    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());

        uint8x16_t mask = vcleq_u8(data, upper_threshold);

        mask = vandq_u8(vcgeq_u8(data, lower_threshold), mask);

        vst1q_u8(output.ptr(), vbslq_u8(mask, true_value, false_value));
    },
    input, output);
}

void NEThresholdKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
} // namespace arm_compute
