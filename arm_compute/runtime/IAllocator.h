/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_IALLOCATOR_H
#define ARM_COMPUTE_IALLOCATOR_H

#include "arm_compute/runtime/IMemoryRegion.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
/** Allocator interface */
class IAllocator
{
public:
    /** Default virtual destructor. */
    virtual ~IAllocator() = default;
    /** Interface to be implemented by the child class to allocate bytes
     *
     * @param[in] size      Size to allocate
     * @param[in] alignment Alignment that the returned pointer should comply with
     *
     * @return A pointer to the allocated memory
     */
    virtual void *allocate(size_t size, size_t alignment) = 0;
    /** Interface to be implemented by the child class to free the allocated tensor */
    virtual void free(void *ptr) = 0;
    /** Create self-managed memory region
     *
     * @param[in] size      Size of the memory region
     * @param[in] alignment Alignment of the memory region
     *
     * @return The memory region object
     */
    virtual std::unique_ptr<IMemoryRegion> make_region(size_t size, size_t alignment) = 0;
};
} // arm_compute
#endif /*ARM_COMPUTE_IALLOCATOR_H */
