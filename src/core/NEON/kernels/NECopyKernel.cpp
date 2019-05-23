/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NECopyKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

NECopyKernel::NECopyKernel()
    : _input(nullptr), _output(nullptr)
{
}

void NECopyKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _input  = input;
    _output = output;

    INEKernel::configure(calculate_max_window(*output->info()));
}

Status NECopyKernel::validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    return Status{};
}

void NECopyKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window output_window{ window };
    output_window.set(Window::DimX, Window::Dimension(output_window.x().start(), output_window.x().end(), _input->info()->dimension(0)));

    Window out_slice = output_window.first_slice_window_1D();

    do
    {
        Iterator input_it(_input, out_slice);
        Iterator output_it(_output, out_slice);

        execute_window_loop(out_slice, [&](const Coordinates &)
        {
            memcpy(output_it.ptr(), input_it.ptr(), _output->info()->dimension(0) * _output->info()->element_size());
        },
        input_it, output_it);

    }
    while(output_window.slide_window_slice_1D(out_slice));
}
