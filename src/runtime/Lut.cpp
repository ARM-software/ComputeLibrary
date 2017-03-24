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
#include "arm_compute/runtime/Lut.h"

#include <cstring>

using namespace arm_compute;

Lut::Lut()
    : _allocator()
{
}

Lut::Lut(size_t num_elements, DataType data_type)
    : _allocator()
{
    _allocator.init(num_elements, data_type);
}

size_t Lut::num_elements() const
{
    return _allocator.num_elements();
}

uint32_t Lut::index_offset() const
{
    return (DataType::S16 == _allocator.type()) ? num_elements() / 2 : 0;
}

size_t Lut::size_in_bytes() const
{
    return _allocator.size();
}

DataType Lut::type() const
{
    return _allocator.type();
}

uint8_t *Lut::buffer() const
{
    return _allocator.data();
}

void Lut::clear()
{
    ARM_COMPUTE_ERROR_ON(this->buffer() == nullptr);
    std::memset(this->buffer(), 0, this->size_in_bytes());
}

ILutAllocator *Lut::allocator()
{
    return &_allocator;
}
