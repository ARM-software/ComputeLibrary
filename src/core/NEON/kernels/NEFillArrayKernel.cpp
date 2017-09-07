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
#include "arm_compute/core/NEON/kernels/NEFillArrayKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

NEFillArrayKernel::NEFillArrayKernel()
    : _input(nullptr), _output(nullptr), _threshold(0)
{
}

void NEFillArrayKernel::configure(const IImage *input, uint8_t threshold, IKeyPointArray *output)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    _input     = input;
    _output    = output;
    _threshold = threshold;

    constexpr unsigned int num_elems_processed_per_iteration = 1;
    constexpr unsigned int num_elems_read_per_iteration      = 1;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, num_elems_read_per_iteration));

    INEKernel::configure(win);
}

bool NEFillArrayKernel::is_parallelisable() const
{
    return false;
}

void NEFillArrayKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Iterator input(_input, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8_t value = *input.ptr();

        if(value >= _threshold)
        {
            KeyPoint p;
            p.x               = id.x();
            p.y               = id.y();
            p.strength        = value;
            p.tracking_status = 1;

            if(!_output->push_back(p))
            {
                return; //Overflowed: stop trying to add more points
            }
        }
    },
    input);
}
