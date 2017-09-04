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
#include "arm_compute/runtime/CL/CLLut.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include <cstring>

using namespace arm_compute;

CLLut::CLLut()
    : _allocator()
{
}

CLLut::CLLut(size_t num_elements, DataType data_type)
    : _allocator()
{
    _allocator.init(num_elements, data_type);
}

size_t CLLut::num_elements() const
{
    return _allocator.num_elements();
}

uint32_t CLLut::index_offset() const
{
    return (DataType::S16 == _allocator.type()) ? num_elements() / 2 : 0;
}

size_t CLLut::size_in_bytes() const
{
    return _allocator.size();
}

DataType CLLut::type() const
{
    return _allocator.type();
}

const cl::Buffer &CLLut::cl_buffer() const
{
    return _allocator.cl_data();
}

void CLLut::clear()
{
    cl::CommandQueue &q    = CLScheduler::get().queue();
    uint8_t          *data = _allocator.map(q, true /* blocking */);
    std::memset(data, 0, size_in_bytes());
    _allocator.unmap(q, data);
}

ILutAllocator *CLLut::allocator()
{
    return &_allocator;
}

void CLLut::map(bool blocking)
{
    ICLLut::map(CLScheduler::get().queue(), blocking);
}

void CLLut::unmap()
{
    ICLLut::unmap(CLScheduler::get().queue());
}

uint8_t *CLLut::do_map(cl::CommandQueue &q, bool blocking)
{
    return _allocator.map(q, blocking);
}

void CLLut::do_unmap(cl::CommandQueue &q)
{
    _allocator.unmap(q, buffer());
}
