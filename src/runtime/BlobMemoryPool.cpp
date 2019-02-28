/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/BlobMemoryPool.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/IMemoryPool.h"
#include "arm_compute/runtime/Types.h"
#include "support/ToolchainSupport.h"

#include <vector>

using namespace arm_compute;

BlobMemoryPool::BlobMemoryPool(IAllocator *allocator, std::vector<BlobInfo> blob_info)
    : _allocator(allocator), _blobs(), _blob_info(std::move(blob_info))
{
    ARM_COMPUTE_ERROR_ON(!allocator);
    allocate_blobs(_blob_info);
}

BlobMemoryPool::~BlobMemoryPool()
{
    ARM_COMPUTE_ERROR_ON(!_allocator);
    free_blobs();
}

void BlobMemoryPool::acquire(MemoryMappings &handles)
{
    // Set memory to handlers
    for(auto &handle : handles)
    {
        ARM_COMPUTE_ERROR_ON(handle.first == nullptr);
        handle.first->set_region(_blobs[handle.second].get());
    }
}

void BlobMemoryPool::release(MemoryMappings &handles)
{
    for(auto &handle : handles)
    {
        ARM_COMPUTE_ERROR_ON(handle.first == nullptr);
        handle.first->set_region(nullptr);
    }
}

MappingType BlobMemoryPool::mapping_type() const
{
    return MappingType::BLOBS;
}

std::unique_ptr<IMemoryPool> BlobMemoryPool::duplicate()
{
    ARM_COMPUTE_ERROR_ON(!_allocator);
    return support::cpp14::make_unique<BlobMemoryPool>(_allocator, _blob_info);
}

void BlobMemoryPool::allocate_blobs(const std::vector<BlobInfo> &blob_info)
{
    ARM_COMPUTE_ERROR_ON(!_allocator);

    for(const auto &bi : blob_info)
    {
        _blobs.push_back(_allocator->make_region(bi.size, bi.alignment));
    }
}

void BlobMemoryPool::free_blobs()
{
    _blobs.clear();
}