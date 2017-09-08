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
#include "arm_compute/runtime/CL/CLTensorAllocator.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLTensorAllocator::CLTensorAllocator(CLTensor *owner)
    : _associated_memory_group(nullptr), _buffer(), _mapping(nullptr), _owner(owner)
{
}

CLTensorAllocator::~CLTensorAllocator()
{
    _buffer = cl::Buffer();
}

uint8_t *CLTensorAllocator::data()
{
    return _mapping;
}

const cl::Buffer &CLTensorAllocator::cl_data() const
{
    return _buffer;
}

void CLTensorAllocator::allocate()
{
    ARM_COMPUTE_ERROR_ON(_buffer.get() != nullptr);
    if(_associated_memory_group == nullptr)
    {
        _buffer = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, info().total_size());
    }
    else
    {
        _associated_memory_group->finalize_memory(_owner, reinterpret_cast<void **>(&_buffer()), info().total_size());
    }
    info().set_is_resizable(false);
}

void CLTensorAllocator::free()
{
    if(_associated_memory_group == nullptr)
    {
        _buffer = cl::Buffer();
        info().set_is_resizable(true);
    }
}

void CLTensorAllocator::set_associated_memory_group(CLMemoryGroup *associated_memory_group)
{
    ARM_COMPUTE_ERROR_ON(associated_memory_group == nullptr);
    ARM_COMPUTE_ERROR_ON(_associated_memory_group != nullptr);
    ARM_COMPUTE_ERROR_ON(_buffer.get() != nullptr);
    _associated_memory_group = associated_memory_group;
}

uint8_t *CLTensorAllocator::lock()
{
    ARM_COMPUTE_ERROR_ON(_mapping != nullptr);
    _mapping = map(CLScheduler::get().queue(), true);
    return _mapping;
}

void CLTensorAllocator::unlock()
{
    ARM_COMPUTE_ERROR_ON(_mapping == nullptr);
    unmap(CLScheduler::get().queue(), _mapping);
    _mapping = nullptr;
}

uint8_t *CLTensorAllocator::map(cl::CommandQueue &q, bool blocking)
{
    ARM_COMPUTE_ERROR_ON(_buffer.get() == nullptr);
    return static_cast<uint8_t *>(q.enqueueMapBuffer(_buffer, blocking ? CL_TRUE : CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, info().total_size()));
}

void CLTensorAllocator::unmap(cl::CommandQueue &q, uint8_t *mapping)
{
    ARM_COMPUTE_ERROR_ON(_buffer.get() == nullptr);
    q.enqueueUnmapMemObject(_buffer, mapping);
}
