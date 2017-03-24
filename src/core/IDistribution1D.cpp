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
#include "arm_compute/core/IDistribution1D.h"

#include "arm_compute/core/Error.h"

using namespace arm_compute;

IDistribution1D::IDistribution1D(size_t num_bins, int32_t offset, uint32_t range)
    : _num_bins(num_bins), _offset(offset), _range(range)
{
    ARM_COMPUTE_ERROR_ON_MSG(0 == _num_bins, "Invalid number of bins, it should be greater than 0");
}

size_t IDistribution1D::num_bins() const
{
    return _num_bins;
}

int32_t IDistribution1D::offset() const
{
    return _offset;
}

uint32_t IDistribution1D::range() const
{
    return _range;
}

uint32_t IDistribution1D::window() const
{
    return _range / _num_bins;
}

size_t IDistribution1D::size() const
{
    return _num_bins * sizeof(uint32_t);
}

void IDistribution1D::set_range(uint32_t range)
{
    _range = range;
}

size_t IDistribution1D::dimensions() const
{
    return 1;
}
