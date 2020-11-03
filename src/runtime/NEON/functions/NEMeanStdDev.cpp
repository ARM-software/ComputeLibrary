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
#include "arm_compute/runtime/NEON/functions/NEMeanStdDev.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "src/core/NEON/kernels/NEMeanStdDevKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
NEMeanStdDev::~NEMeanStdDev() = default;

NEMeanStdDev::NEMeanStdDev()
    : _mean_stddev_kernel(), _fill_border_kernel(), _global_sum(0), _global_sum_squared(0)
{
}

void NEMeanStdDev::configure(IImage *input, float *mean, float *stddev)
{
    _mean_stddev_kernel = arm_compute::support::cpp14::make_unique<NEMeanStdDevKernel>();
    _fill_border_kernel = arm_compute::support::cpp14::make_unique<NEFillBorderKernel>();

    _mean_stddev_kernel->configure(input, mean, &_global_sum, stddev, &_global_sum_squared);
    _fill_border_kernel->configure(input, _mean_stddev_kernel->border_size(), BorderMode::CONSTANT, PixelValue(static_cast<uint8_t>(0)));
}

void NEMeanStdDev::run()
{
    _global_sum         = 0;
    _global_sum_squared = 0;

    NEScheduler::get().schedule(_fill_border_kernel.get(), Window::DimZ);
    NEScheduler::get().schedule(_mean_stddev_kernel.get(), Window::DimY);
}
} // namespace arm_compute
