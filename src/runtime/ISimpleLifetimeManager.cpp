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
#include "arm_compute/runtime/ISimpleLifetimeManager.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/IMemoryGroup.h"
#include "arm_compute/runtime/IMemoryPool.h"
#include "support/ToolchainSupport.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

using namespace arm_compute;

ISimpleLifetimeManager::ISimpleLifetimeManager()
    : _active_group(nullptr), _active_elements(), _finalized_groups()
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
    ARM_COMPUTE_ERROR_ON_MSG(std::find_if(std::begin(_active_elements), std::end(_active_elements), [&obj](const Element & e)
    {
        return obj == e.id;
    }) != std::end(_active_elements),
    "Memory object is already registered!");

    // Insert object in groups and mark its finalized state to false
    _active_elements.emplace_back(obj);
}

void ISimpleLifetimeManager::end_lifetime(void *obj, void **handle, size_t size)
{
    ARM_COMPUTE_ERROR_ON(obj == nullptr);

    // Find object
    auto it = std::find_if(std::begin(_active_elements), std::end(_active_elements), [&obj](const Element & e)
    {
        return obj == e.id;
    });
    ARM_COMPUTE_ERROR_ON(it == std::end(_active_elements));

    // Update object fields and mark object as complete
    it->handle = handle;
    it->size   = size;
    it->status = true;

    // Check if all object are finalized and reset active group
    if(are_all_finalized())
    {
        // Update finalized groups
        _finalized_groups[_active_group].insert(std::end(_finalized_groups[_active_group]), std::begin(_active_elements), std::end(_active_elements));

        // Update blobs and group mappings
        update_blobs_and_mappings();

        // Reset state
        _active_elements.clear();
        _active_group = nullptr;
    }
}

bool ISimpleLifetimeManager::are_all_finalized() const
{
    return !std::any_of(std::begin(_active_elements), std::end(_active_elements), [](const Element e)
    {
        return !e.status;
    });
}
