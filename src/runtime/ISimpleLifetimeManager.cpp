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
#include "arm_compute/runtime/ISimpleLifetimeManager.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/IMemory.h"
#include "arm_compute/runtime/IMemoryGroup.h"
#include "arm_compute/runtime/IMemoryPool.h"
#include "support/ToolchainSupport.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

using namespace arm_compute;

ISimpleLifetimeManager::ISimpleLifetimeManager()
    : _active_group(nullptr), _active_elements(), _free_blobs(), _occupied_blobs(), _finalized_groups()
{
}

void ISimpleLifetimeManager::register_group(IMemoryGroup *group)
{
    if(_active_group == nullptr)
    {
        ARM_COMPUTE_ERROR_ON(group == nullptr);
        _active_group = group;
    }
}

void ISimpleLifetimeManager::start_lifetime(void *obj)
{
    ARM_COMPUTE_ERROR_ON(obj == nullptr);
    ARM_COMPUTE_ERROR_ON_MSG(_active_elements.find(obj) != std::end(_active_elements), "Memory object is already registered!");

    // Check if there is a free blob
    if(_free_blobs.empty())
    {
        _occupied_blobs.emplace_front(Blob{ obj, 0, 0, { obj } });
    }
    else
    {
        _occupied_blobs.splice(std::begin(_occupied_blobs), _free_blobs, std::begin(_free_blobs));
        _occupied_blobs.front().id = obj;
    }

    // Insert object in groups and mark its finalized state to false
    _active_elements.insert(std::make_pair(obj, obj));
}

void ISimpleLifetimeManager::end_lifetime(void *obj, IMemory &obj_memory, size_t size, size_t alignment)
{
    ARM_COMPUTE_ERROR_ON(obj == nullptr);

    // Find object
    auto active_object_it = _active_elements.find(obj);
    ARM_COMPUTE_ERROR_ON(active_object_it == std::end(_active_elements));

    // Update object fields and mark object as complete
    Element &el  = active_object_it->second;
    el.handle    = &obj_memory;
    el.size      = size;
    el.alignment = alignment;
    el.status    = true;

    // Find object in the occupied lists
    auto occupied_blob_it = std::find_if(std::begin(_occupied_blobs), std::end(_occupied_blobs), [&obj](const Blob & b)
    {
        return obj == b.id;
    });
    ARM_COMPUTE_ERROR_ON(occupied_blob_it == std::end(_occupied_blobs));

    // Update occupied blob and return as free
    occupied_blob_it->bound_elements.insert(obj);
    occupied_blob_it->max_size      = std::max(occupied_blob_it->max_size, size);
    occupied_blob_it->max_alignment = std::max(occupied_blob_it->max_alignment, alignment);
    occupied_blob_it->id            = nullptr;
    _free_blobs.splice(std::begin(_free_blobs), _occupied_blobs, occupied_blob_it);

    // Check if all object are finalized and reset active group
    if(are_all_finalized())
    {
        ARM_COMPUTE_ERROR_ON(!_occupied_blobs.empty());

        // Update blobs and group mappings
        update_blobs_and_mappings();

        // Update finalized groups
        _finalized_groups[_active_group] = std::move(_active_elements);

        // Reset state
        _active_elements.clear();
        _active_group = nullptr;
        _free_blobs.clear();
    }
}

bool ISimpleLifetimeManager::are_all_finalized() const
{
    return !std::any_of(std::begin(_active_elements), std::end(_active_elements), [](const std::pair<void *, Element> &e)
    {
        return !e.second.status;
    });
}
