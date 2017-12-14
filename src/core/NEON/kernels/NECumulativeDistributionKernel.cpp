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
#include "arm_compute/core/NEON/kernels/NECumulativeDistributionKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IDistribution1D.h"
#include "arm_compute/core/ILut.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace arm_compute;

NECumulativeDistributionKernel::NECumulativeDistributionKernel()
    : _input(nullptr), _distribution(nullptr), _cumulative_sum(nullptr), _output(nullptr)
{
}

bool NECumulativeDistributionKernel::is_parallelisable() const
{
    return false;
}

void NECumulativeDistributionKernel::configure(const IImage *input, const IDistribution1D *distribution, IDistribution1D *cumulative_sum, ILut *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, distribution, cumulative_sum, output);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);

    set_format_if_unknown(*input->info(), Format::U8);

    ARM_COMPUTE_ERROR_ON(distribution->num_bins() != cumulative_sum->num_bins());
    ARM_COMPUTE_ERROR_ON(distribution->num_bins() != output->num_elements());
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(input->info()->data_type() != output->type());

    _input          = input;
    _distribution   = distribution;
    _cumulative_sum = cumulative_sum;
    _output         = output;

    INEKernel::configure(calculate_max_window(*input->info()));
}

void NECumulativeDistributionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_distribution->buffer() == nullptr);
    ARM_COMPUTE_ERROR_ON(_cumulative_sum->buffer() == nullptr);
    ARM_COMPUTE_ERROR_ON(_output->buffer() == nullptr);
    ARM_COMPUTE_ERROR_ON_MSG(_distribution->num_bins() < 256, "Distribution must have 256 bins");

    // Calculate the cumulative distribution (summed histogram).
    const uint32_t *hist           = _distribution->buffer();
    uint32_t       *cumulative_sum = _cumulative_sum->buffer();
    uint8_t        *output         = _output->buffer();

    // Calculate cumulative distribution
    std::partial_sum(hist, hist + _histogram_size, cumulative_sum);

    // Get the number of pixels that have the lowest value in the input image
    const uint32_t cd_min = *std::find_if(hist, hist + _histogram_size, [](const uint32_t &v)
    {
        return v > 0;
    });
    const uint32_t image_size = cumulative_sum[_histogram_size - 1];

    ARM_COMPUTE_ERROR_ON(cd_min > image_size);

    // Create mapping lookup table
    if(image_size == cd_min)
    {
        std::iota(output, output + _histogram_size, 0);
    }
    else
    {
        const float diff = image_size - cd_min;

        for(unsigned int x = 0; x < _histogram_size; ++x)
        {
            output[x] = lround((cumulative_sum[x] - cd_min) / diff * 255.0f);
        }
    }
}
