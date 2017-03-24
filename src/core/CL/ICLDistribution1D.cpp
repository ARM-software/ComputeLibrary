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
#include "arm_compute/core/CL/ICLDistribution1D.h"

#include "arm_compute/core/Error.h"

using namespace arm_compute;

ICLDistribution1D::ICLDistribution1D(size_t num_bins, int32_t offset, uint32_t range)
    : IDistribution1D(num_bins, offset, range), _mapping(nullptr)
{
}

void ICLDistribution1D::map(cl::CommandQueue &q, bool blocking)
{
    ARM_COMPUTE_ERROR_ON(_mapping != nullptr);
    _mapping = do_map(q, blocking);
}

void ICLDistribution1D::unmap(cl::CommandQueue &q)
{
    ARM_COMPUTE_ERROR_ON(_mapping == nullptr);
    do_unmap(q);
    _mapping = nullptr;
}

uint32_t *ICLDistribution1D::buffer() const
{
    return _mapping;
}
