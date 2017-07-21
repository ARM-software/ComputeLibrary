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
#include "arm_compute/runtime/CL/functions/CLMeanStdDev.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLMeanStdDev::CLMeanStdDev()
    : _mean_stddev_kernel(),
      _fill_border_kernel(),
      _global_sum(),
      _global_sum_squared()
{
}

void CLMeanStdDev::configure(ICLImage *input, float *mean, float *stddev)
{
    _global_sum = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_ulong));

    if(stddev != nullptr)
    {
        _global_sum_squared = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_ulong));
    }

    _mean_stddev_kernel.configure(input, mean, &_global_sum, stddev, &_global_sum_squared);
    _fill_border_kernel.configure(input, _mean_stddev_kernel.border_size(), BorderMode::CONSTANT, PixelValue(static_cast<uint8_t>(0)));
}

void CLMeanStdDev::run()
{
    CLScheduler::get().enqueue(_fill_border_kernel);
    CLScheduler::get().enqueue(_mean_stddev_kernel);
}
