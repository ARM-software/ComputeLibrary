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

#include "arm_compute/runtime/CL/CLHOG.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLHOG::CLHOG()
    : _info(), _buffer()
{
}

void CLHOG::init(const HOGInfo &input)
{
    ARM_COMPUTE_ERROR_ON(_buffer.get() != nullptr);
    _info   = input;
    _buffer = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, info()->descriptor_size() * sizeof(float));
}

void CLHOG::free()
{
    ARM_COMPUTE_ERROR_ON(_buffer.get() == nullptr);

    _buffer = cl::Buffer();
}

const HOGInfo *CLHOG::info() const
{
    return &_info;
}

const cl::Buffer &CLHOG::cl_buffer() const
{
    return _buffer;
}

void CLHOG::map(bool blocking)
{
    ARM_COMPUTE_ERROR_ON(descriptor() != nullptr);
    ICLHOG::map(CLScheduler::get().queue(), blocking);
}

void CLHOG::unmap()
{
    ARM_COMPUTE_ERROR_ON(descriptor() == nullptr);
    ICLHOG::unmap(CLScheduler::get().queue());
}

uint8_t *CLHOG::do_map(cl::CommandQueue &q, bool blocking)
{
    ARM_COMPUTE_ERROR_ON(_buffer.get() == nullptr);
    return static_cast<uint8_t *>(q.enqueueMapBuffer(_buffer, blocking ? CL_TRUE : CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, info()->descriptor_size()));
}

void CLHOG::do_unmap(cl::CommandQueue &q)
{
    ARM_COMPUTE_ERROR_ON(_buffer.get() == nullptr);
    q.enqueueUnmapMemObject(_buffer, descriptor());
}