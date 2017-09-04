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
#include "arm_compute/core/CL/ICLTensor.h"

#include <cstring>

using namespace arm_compute;

ICLTensor::ICLTensor()
    : _mapping(nullptr)
{
}

void ICLTensor::map(cl::CommandQueue &q, bool blocking)
{
    _mapping = do_map(q, blocking);
}

void ICLTensor::unmap(cl::CommandQueue &q)
{
    do_unmap(q);
    _mapping = nullptr;
}

void ICLTensor::clear(cl::CommandQueue &q)
{
    this->map(q);
    std::memset(static_cast<void *>(_mapping), 0, this->info()->total_size());
    this->unmap(q);
}

uint8_t *ICLTensor::buffer() const
{
    return _mapping;
}
