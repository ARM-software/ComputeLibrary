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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IDistribution1D.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEHistogramKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

using namespace arm_compute;

NEHistogram::NEHistogram()
    : _histogram_kernel(), _border_histogram_kernel(), _local_hist(), _window_lut(arm_compute::cpp14::make_unique<uint32_t[]>(window_lut_default_size)), _local_hist_size(0), _run_border_hist(false)
{
}

void NEHistogram::configure(const IImage *input, IDistribution1D *output)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    ARM_COMPUTE_ERROR_ON_MSG((input->info()->dimension(0) % 16) != 0, "Currently the width of the image must be a multiple of 16");

    // Allocate space for threads local histograms
    _local_hist_size = output->num_bins() * NEScheduler::get().num_threads();
    _local_hist      = arm_compute::cpp14::make_unique<uint32_t[]>(_local_hist_size);

    // Configure kernels
    _histogram_kernel.configure(input, output, _local_hist.get(), _window_lut.get());

    //COMPMID-196: Figure out how to handle the border part
    ARM_COMPUTE_UNUSED(_run_border_hist);
#if 0
    _run_border_hist = (input->info()->dimension(0) % _histogram_kernel.num_elems_processed_per_iteration()) != 0U;

    if(_run_border_hist)
    {
        _border_histogram_kernel.configure(input, output, _window_lut.get(), _histogram_kernel.num_elems_processed_per_iteration());
    }
#endif
}

void NEHistogram::run()
{
    // Calculate histogram of input.
    NEScheduler::get().multithread(&_histogram_kernel);

    // Calculate remaining pixels when image is not multiple of the elements of histogram kernel
    //COMPMID-196: Figure out how to handle the border part
#if 0
    if(_run_border_hist)
    {
        _border_histogram_kernel.run(_border_histogram_kernel.window());
    }
#endif
}
