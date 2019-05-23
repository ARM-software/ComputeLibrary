/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/GLES_COMPUTE/GCMemory.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCMemoryRegion.h"

namespace arm_compute
{
GCMemory::GCMemory()
    : _region(nullptr), _region_owned(nullptr)
{
}

GCMemory::GCMemory(const std::shared_ptr<IGCMemoryRegion> &memory)
    : _region(nullptr), _region_owned(memory)
{
    _region_owned = memory;
    _region       = _region_owned.get();
}

GCMemory::GCMemory(IGCMemoryRegion *memory)
    : _region(memory), _region_owned(nullptr)
{
    _region = memory;
}

IGCMemoryRegion *GCMemory::gc_region()
{
    return _region;
}

IGCMemoryRegion *GCMemory::gc_region() const
{
    return _region;
}

IMemoryRegion *GCMemory::region()
{
    return _region;
}

IMemoryRegion *GCMemory::region() const
{
    return _region;
}

void GCMemory::set_region(IMemoryRegion *region)
{
    auto gc_region = utils::cast::polymorphic_downcast<IGCMemoryRegion *>(region);
    _region_owned  = nullptr;
    _region        = gc_region;
}

void GCMemory::set_owned_region(std::unique_ptr<IMemoryRegion> region)
{
    _region_owned = utils::cast::polymorphic_downcast_unique_ptr<IGCMemoryRegion>(std::move(region));
    _region       = _region_owned.get();
}
} // namespace arm_compute
