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
#include <algorithm>

#include "arm_compute/runtime/OffsetMemoryPool.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/IMemoryPool.h"
#include "arm_compute/runtime/Types.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

OffsetMemoryPool::OffsetMemoryPool(IAllocator *allocator, size_t blob_size)
    : _allocator(allocator), _blob(), _blob_size(blob_size)
{
    ARM_COMPUTE_ERROR_ON(!allocator);
    _blob = _allocator->allocate(_blob_size, 0);
}

OffsetMemoryPool::~OffsetMemoryPool()
{
    ARM_COMPUTE_ERROR_ON(!_allocator);
    _allocator->free(_blob);
    _blob = nullptr;
}

void OffsetMemoryPool::acquire(MemoryMappings &handles)
{
    ARM_COMPUTE_ERROR_ON(_blob == nullptr);

    // Set memory to handlers
    for(auto &handle : handles)
    {
        ARM_COMPUTE_ERROR_ON(handle.first == nullptr);
        *handle.first = reinterpret_cast<uint8_t *>(_blob) + handle.second;
    }
}

void OffsetMemoryPool::release(MemoryMappings &handles)
{
    for(auto &handle : handles)
    {
        ARM_COMPUTE_ERROR_ON(handle.first == nullptr);
        *handle.first = nullptr;
    }
}

MappingType OffsetMemoryPool::mapping_type() const
{
    return MappingType::OFFSETS;
}

std::unique_ptr<IMemoryPool> OffsetMemoryPool::duplicate()
{
    ARM_COMPUTE_ERROR_ON(!_allocator);
    return support::cpp14::make_unique<OffsetMemoryPool>(_allocator, _blob_size);
}