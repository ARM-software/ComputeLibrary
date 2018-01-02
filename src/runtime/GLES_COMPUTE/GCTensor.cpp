/*
 * Copyright (c) 2017 ARM Limited.
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

#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"

using namespace arm_compute;

GCTensor::GCTensor()
    : _allocator()
{
}

ITensorAllocator *GCTensor::allocator()
{
    return &_allocator;
}

TensorInfo *GCTensor::info() const
{
    return &_allocator.info();
}

TensorInfo *GCTensor::info()
{
    return &_allocator.info();
}

uint8_t *GCTensor::buffer() const
{
    return _allocator.data();
}

GLuint GCTensor::gc_buffer() const
{
    return _allocator.get_gl_ssbo_name();
}

void GCTensor::map(bool blocking)
{
    IGCTensor::map(blocking);
}

void GCTensor::unmap()
{
    IGCTensor::unmap();
}

uint8_t *GCTensor::do_map(bool blocking)
{
    return _allocator.map(blocking);
}

void GCTensor::do_unmap()
{
    _allocator.unmap();
}