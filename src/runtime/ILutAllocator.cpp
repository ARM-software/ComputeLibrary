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
#include "arm_compute/runtime/ILutAllocator.h"

#include "arm_compute/core/Utils.h"

using namespace arm_compute;

ILutAllocator::ILutAllocator()
    : _num_elements(0), _data_type(DataType::U8)
{
}

void ILutAllocator::init(size_t num_elements, DataType data_type)
{
    // Init internal metadata
    _num_elements = num_elements;
    _data_type    = data_type;

    // Allocate the image's memory
    allocate();
}

size_t ILutAllocator::num_elements() const
{
    return _num_elements;
}

DataType ILutAllocator::type() const
{
    return _data_type;
}

size_t ILutAllocator::size() const
{
    return data_size_from_type(_data_type) * num_elements();
}
