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
#include "arm_compute/runtime/OffsetLifetimeManager.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/IMemoryGroup.h"
#include "arm_compute/runtime/OffsetMemoryPool.h"
#include "support/ToolchainSupport.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

using namespace arm_compute;

OffsetLifetimeManager::OffsetLifetimeManager()
    : _blob(0)
{
}

std::unique_ptr<IMemoryPool> OffsetLifetimeManager::create_pool(IAllocator *allocator)
{
    ARM_COMPUTE_ERROR_ON(allocator == nullptr);
    return support::cpp14::make_unique<OffsetMemoryPool>(allocator, _blob);
}

MappingType OffsetLifetimeManager::mapping_type() const
{
    return MappingType::OFFSETS;
}

void OffsetLifetimeManager::update_blobs_and_mappings()
{
    ARM_COMPUTE_ERROR_ON(!are_all_finalized());
    ARM_COMPUTE_ERROR_ON(_active_group == nullptr);

    // Update blob size
    size_t max_group_size = std::accumulate(std::begin(_free_blobs), std::end(_free_blobs), static_cast<size_t>(0), [](size_t s, const Blob & b)
    {
        return s + b.max_size;
    });
    _blob = std::max(_blob, max_group_size);

    // Calculate group mappings
    auto &group_mappings = _active_group->mappings();
    size_t offset         = 0;
    for(auto &free_blob : _free_blobs)
    {
        for(auto &bound_element_id : free_blob.bound_elements)
        {
            ARM_COMPUTE_ERROR_ON(_active_elements.find(bound_element_id) == std::end(_active_elements));
            Element &bound_element               = _active_elements[bound_element_id];
            group_mappings[bound_element.handle] = offset;
        }
        offset += free_blob.max_size;
        ARM_COMPUTE_ERROR_ON(offset > _blob);
    }
}
