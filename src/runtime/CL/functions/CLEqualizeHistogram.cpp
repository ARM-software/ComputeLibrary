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
#include "arm_compute/runtime/CL/functions/CLEqualizeHistogram.h"

#include "arm_compute/core/CL/ICLDistribution1D.h"
#include "arm_compute/core/CL/ICLLut.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

using namespace arm_compute;

namespace
{
void calculate_cum_dist_and_lut(CLDistribution1D &dist, CLDistribution1D &cum_dist, CLLut &lut)
{
    dist.map(true);
    cum_dist.map(true);
    lut.map(true);

    const uint32_t *dist_ptr     = dist.buffer();
    uint32_t       *cum_dist_ptr = cum_dist.buffer();
    uint8_t        *lut_ptr      = lut.buffer();

    ARM_COMPUTE_ERROR_ON(dist_ptr == nullptr);
    ARM_COMPUTE_ERROR_ON(cum_dist_ptr == nullptr);
    ARM_COMPUTE_ERROR_ON(lut_ptr == nullptr);

    // Calculate cumulative distribution
    std::partial_sum(dist_ptr, dist_ptr + 256, cum_dist_ptr);

    // Get the number of pixels that have the lowest value in the input image
    const uint32_t num_lowest_pixels = *std::find_if(dist_ptr, dist_ptr + 256, [](const uint32_t &v)
    {
        return v > 0;
    });
    const size_t image_size = cum_dist_ptr[255];

    if(image_size == num_lowest_pixels)
    {
        std::iota(lut_ptr, lut_ptr + 256, 0);
    }
    else
    {
        const float diff = image_size - num_lowest_pixels;

        for(size_t i = 0; i < 256; ++i)
        {
            lut_ptr[i] = lround((cum_dist_ptr[i] - num_lowest_pixels) / diff * 255.f);
        }
    }

    dist.unmap();
    cum_dist.unmap();
    lut.unmap();
}
} // namespace

CLEqualizeHistogram::CLEqualizeHistogram()
    : _histogram_kernel(), _border_histogram_kernel(), _map_histogram_kernel(), _hist(nr_bins, 0, max_range), _cum_dist(nr_bins, 0, max_range), _cd_lut(nr_bins, DataType::U8)
{
}

void CLEqualizeHistogram::configure(const ICLImage *input, ICLImage *output)
{
    _histogram_kernel.configure(input, &_hist);
    _border_histogram_kernel.configure(input, &_hist);
    _map_histogram_kernel.configure(input, &_cd_lut, output);
}

void CLEqualizeHistogram::run()
{
    // Calculate histogram of input.
    CLScheduler::get().enqueue(_histogram_kernel, false);

    // Calculate remaining pixels when image is not multiple of the elements of histogram kernel
    CLScheduler::get().enqueue(_border_histogram_kernel, false);

    // Calculate cumulative distribution of histogram and create LUT.
    calculate_cum_dist_and_lut(_hist, _cum_dist, _cd_lut);

    // Map input to output using created LUT.
    CLScheduler::get().enqueue(_map_histogram_kernel);
}
