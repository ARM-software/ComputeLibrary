/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCMemoryRegion.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

GCTensorAllocator::GCTensorAllocator(GCTensor *owner)
    : _associated_memory_group(nullptr), _memory(), _mapping(nullptr), _owner(owner)
{
}

uint8_t *GCTensorAllocator::data()
{
    return _mapping;
}

void GCTensorAllocator::allocate()
{
    if(_associated_memory_group == nullptr)
    {
        _memory.set_owned_region(support::cpp14::make_unique<GCBufferMemoryRegion>(info().total_size()));
    }
    else
    {
        _associated_memory_group->finalize_memory(_owner, _memory, info().total_size());
    }
    info().set_is_resizable(false);
}

void GCTensorAllocator::free()
{
    _mapping = nullptr;
    _memory.set_region(nullptr);
    info().set_is_resizable(true);
}

void GCTensorAllocator::set_associated_memory_group(GCMemoryGroup *associated_memory_group)
{
    ARM_COMPUTE_ERROR_ON(associated_memory_group == nullptr);
    ARM_COMPUTE_ERROR_ON(_associated_memory_group != nullptr && _associated_memory_group != associated_memory_group);
    ARM_COMPUTE_ERROR_ON(_memory.region() != nullptr && _memory.gc_region()->gc_ssbo_name() != 0);

    _associated_memory_group = associated_memory_group;
}

uint8_t *GCTensorAllocator::lock()
{
    return map(true);
}

void GCTensorAllocator::unlock()
{
    unmap();
}

GLuint GCTensorAllocator::get_gl_ssbo_name() const
{
    return (_memory.region() == nullptr) ? static_cast<GLuint>(0) : _memory.gc_region()->gc_ssbo_name();
}

uint8_t *GCTensorAllocator::map(bool blocking)
{
    ARM_COMPUTE_ERROR_ON(_mapping != nullptr);
    ARM_COMPUTE_ERROR_ON(_memory.region() == nullptr);

    _mapping = reinterpret_cast<uint8_t *>(_memory.gc_region()->map(blocking));
    return _mapping;
}

void GCTensorAllocator::unmap()
{
    ARM_COMPUTE_ERROR_ON(_mapping == nullptr);
    ARM_COMPUTE_ERROR_ON(_memory.region() == nullptr);

    _memory.gc_region()->unmap();
    _mapping = nullptr;
}