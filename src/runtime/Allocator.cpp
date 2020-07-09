/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/MemoryRegion.h"

#include "arm_compute/core/Error.h"
#include "support/MemorySupport.h"

#include <cstddef>

using namespace arm_compute;

void *Allocator::allocate(size_t size, size_t alignment)
{
    ARM_COMPUTE_UNUSED(alignment);
    return ::operator new(size);
}

void Allocator::free(void *ptr)
{
    ::operator delete(ptr);
}

std::unique_ptr<IMemoryRegion> Allocator::make_region(size_t size, size_t alignment)
{
    return arm_compute::support::cpp14::make_unique<MemoryRegion>(size, alignment);
}
