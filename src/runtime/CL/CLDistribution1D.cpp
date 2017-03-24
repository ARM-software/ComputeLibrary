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
#include "arm_compute/runtime/CL/CLDistribution1D.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLDistribution1D::CLDistribution1D(size_t num_bins, int32_t offset, uint32_t range)
    : ICLDistribution1D(num_bins, offset, range), _mem(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, num_bins * sizeof(int32_t))
{
}

void CLDistribution1D::map(bool blocking)
{
    ICLDistribution1D::map(CLScheduler::get().queue(), blocking);
}

void CLDistribution1D::unmap()
{
    ICLDistribution1D::unmap(CLScheduler::get().queue());
}

uint32_t *CLDistribution1D::do_map(cl::CommandQueue &q, bool blocking)
{
    ARM_COMPUTE_ERROR_ON(_mem.get() == nullptr);
    return static_cast<uint32_t *>(q.enqueueMapBuffer(_mem, blocking ? CL_TRUE : CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, size()));
}

void CLDistribution1D::do_unmap(cl::CommandQueue &q)
{
    ARM_COMPUTE_ERROR_ON(_mem.get() == nullptr);
    q.enqueueUnmapMemObject(_mem, _mapping);
}

cl::Buffer &CLDistribution1D::cl_buffer()
{
    return _mem;
}
