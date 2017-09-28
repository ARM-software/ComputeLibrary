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
#include "arm_compute/runtime/NEON/functions/NEHistogram.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/IDistribution1D.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

NEHistogram::NEHistogram()
    : _histogram_kernel(), _local_hist(), _window_lut(arm_compute::support::cpp14::make_unique<uint32_t[]>(window_lut_default_size)), _local_hist_size(0)
{
}

void NEHistogram::configure(const IImage *input, IDistribution1D *output)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    // Allocate space for threads local histograms
    _local_hist_size = output->num_bins() * NEScheduler::get().num_threads();
    _local_hist      = arm_compute::support::cpp14::make_unique<uint32_t[]>(_local_hist_size);

    // Configure kernel
    _histogram_kernel.configure(input, output, _local_hist.get(), _window_lut.get());
}

void NEHistogram::run()
{
    // Calculate histogram of input.
    NEScheduler::get().schedule(&_histogram_kernel, Window::DimY);
}
