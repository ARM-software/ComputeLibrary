/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/CL/CLMemory.h"

#include "arm_compute/core/Error.h"

namespace arm_compute
{
CLMemory::CLMemory()
    : _region(nullptr), _region_owned(nullptr)
{
    create_empty_region();
}

CLMemory::CLMemory(std::shared_ptr<ICLMemoryRegion> memory)
    : _region(nullptr), _region_owned(std::move(memory))
{
    if(_region_owned == nullptr)
    {
        create_empty_region();
    }
    _region = _region_owned.get();
}

CLMemory::CLMemory(ICLMemoryRegion *memory)
    : _region(memory), _region_owned(nullptr)
{
    _region = memory;
}

ICLMemoryRegion *CLMemory::region()
{
    return _region;
}

ICLMemoryRegion *CLMemory::region() const
{
    return _region;
}

void CLMemory::create_empty_region()
{
    _region_owned = std::make_shared<CLBufferMemoryRegion>(cl::Context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, 0);
    _region       = _region_owned.get();
}
} // namespace arm_compute