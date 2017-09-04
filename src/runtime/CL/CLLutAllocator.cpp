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
#include "arm_compute/runtime/CL/CLLutAllocator.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLLutAllocator::CLLutAllocator()
    : _buffer(), _mapping(nullptr)
{
}

uint8_t *CLLutAllocator::data()
{
    return _mapping;
}

const cl::Buffer &CLLutAllocator::cl_data() const
{
    return _buffer;
}

uint8_t *CLLutAllocator::map(cl::CommandQueue &q, bool blocking)
{
    ARM_COMPUTE_ERROR_ON(_buffer.get() == nullptr);
    return static_cast<uint8_t *>(q.enqueueMapBuffer(_buffer, blocking ? CL_TRUE : CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, size()));
}

void CLLutAllocator::unmap(cl::CommandQueue &q, uint8_t *mapping)
{
    ARM_COMPUTE_ERROR_ON(_buffer.get() == nullptr);
    q.enqueueUnmapMemObject(_buffer, mapping);
}

void CLLutAllocator::allocate()
{
    _buffer = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, size());
}

uint8_t *CLLutAllocator::lock()
{
    ARM_COMPUTE_ERROR_ON(_mapping != nullptr);
    cl::CommandQueue q = CLScheduler::get().queue();
    _mapping           = map(q, true);
    return _mapping;
}

void CLLutAllocator::unlock()
{
    ARM_COMPUTE_ERROR_ON(_mapping == nullptr);
    cl::CommandQueue q = CLScheduler::get().queue();
    unmap(q, _mapping);
    _mapping = nullptr;
}
