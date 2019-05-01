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
#include "arm_compute/runtime/BlobLifetimeManager.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/BlobMemoryPool.h"
#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/IMemoryGroup.h"
#include "support/ToolchainSupport.h"

#include <algorithm>
#include <cmath>
#include <map>

using namespace arm_compute;

BlobLifetimeManager::BlobLifetimeManager()
    : _blobs()
{
}

std::unique_ptr<IMemoryPool> BlobLifetimeManager::create_pool(IAllocator *allocator)
{
    ARM_COMPUTE_ERROR_ON(allocator == nullptr);
    return support::cpp14::make_unique<BlobMemoryPool>(allocator, _blobs);
}

MappingType BlobLifetimeManager::mapping_type() const
{
    return MappingType::BLOBS;
}

void BlobLifetimeManager::update_blobs_and_mappings()
{
    ARM_COMPUTE_ERROR_ON(!are_all_finalized());
    ARM_COMPUTE_ERROR_ON(_active_group == nullptr);

    // Sort free blobs requirements in descending order.
    _free_blobs.sort([](const Blob & ba, const Blob & bb)
    {
        return ba.max_size > bb.max_size;
    });

    // Create group sizes vector
    std::vector<BlobInfo> group_sizes;
    std::transform(std::begin(_free_blobs), std::end(_free_blobs), std::back_inserter(group_sizes), [](const Blob & b)
    {
        return BlobInfo{ b.max_size, b.max_alignment };
    });

    // Update blob sizes
    size_t max_size = std::max(_blobs.size(), group_sizes.size());
    _blobs.resize(max_size);
    group_sizes.resize(max_size);
    std::transform(std::begin(_blobs), std::end(_blobs), std::begin(group_sizes), std::begin(_blobs), [](BlobInfo lhs, BlobInfo rhs)
    {
        return BlobInfo{ std::max(lhs.size, rhs.size), std::max(lhs.alignment, rhs.alignment) };
    });

    // Calculate group mappings
    auto &group_mappings = _active_group->mappings();
    int   blob_idx       = 0;
    for(auto &free_blob : _free_blobs)
    {
        for(auto &bound_element_id : free_blob.bound_elements)
        {
            ARM_COMPUTE_ERROR_ON(_active_elements.find(bound_element_id) == std::end(_active_elements));
            Element &bound_element               = _active_elements[bound_element_id];
            group_mappings[bound_element.handle] = blob_idx;
        }
        ++blob_idx;
    }
}
