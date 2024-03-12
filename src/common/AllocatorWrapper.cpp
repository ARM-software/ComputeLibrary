/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/common/AllocatorWrapper.h"

#include "arm_compute/core/Error.h"

namespace arm_compute
{
AllocatorWrapper::AllocatorWrapper(const AclAllocator &backing_allocator) noexcept
    : _backing_allocator(backing_allocator)
{
}

void *AllocatorWrapper::alloc(size_t size)
{
    ARM_COMPUTE_ERROR_ON(_backing_allocator.alloc == nullptr);
    return _backing_allocator.alloc(_backing_allocator.user_data, size);
}

void AllocatorWrapper::free(void *ptr)
{
    ARM_COMPUTE_ERROR_ON(_backing_allocator.free == nullptr);
    _backing_allocator.free(_backing_allocator.user_data, ptr);
}

void *AllocatorWrapper::aligned_alloc(size_t size, size_t alignment)
{
    ARM_COMPUTE_ERROR_ON(_backing_allocator.aligned_alloc == nullptr);
    return _backing_allocator.aligned_alloc(_backing_allocator.user_data, size, alignment);
}

void AllocatorWrapper::aligned_free(void *ptr)
{
    ARM_COMPUTE_ERROR_ON(_backing_allocator.aligned_free == nullptr);
    _backing_allocator.aligned_free(_backing_allocator.user_data, ptr);
}

void AllocatorWrapper::set_user_data(void *user_data)
{
    if (user_data != nullptr)
    {
        _backing_allocator.user_data = user_data;
    }
}
} // namespace arm_compute
