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
#include "arm_compute/runtime/Memory.h"

#include "arm_compute/runtime/MemoryRegion.h"

namespace arm_compute
{
Memory::Memory()
    : _region(nullptr), _region_owned(nullptr)
{
}

Memory::Memory(const std::shared_ptr<IMemoryRegion> &memory)
    : _region(nullptr), _region_owned(memory)
{
    _region_owned = memory;
    _region       = _region_owned.get();
}

Memory::Memory(IMemoryRegion *memory)
    : _region(memory), _region_owned(nullptr)
{
    _region = memory;
}

IMemoryRegion *Memory::region()
{
    return _region;
}

IMemoryRegion *Memory::region() const
{
    return _region;
}

void Memory::set_region(IMemoryRegion *region)
{
    _region_owned = nullptr;
    _region       = region;
}

void Memory::set_owned_region(std::unique_ptr<IMemoryRegion> region)
{
    _region_owned = std::move(region);
    _region       = _region_owned.get();
}
} // namespace arm_compute
