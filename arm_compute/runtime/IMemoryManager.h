/*
 * Copyright (c) 2017-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_IMEMORYMANAGER_H
#define ARM_COMPUTE_IMEMORYMANAGER_H

#include "arm_compute/runtime/ILifetimeManager.h"
#include "arm_compute/runtime/IPoolManager.h"

#include <cstddef>

namespace arm_compute
{
// Forward declarations
class IAllocator;
class IMemoryGroup;

/** Memory manager interface to handle allocations of backing memory */
class IMemoryManager
{
public:
    /** Default virtual destructor */
    virtual ~IMemoryManager() = default;
    /** Returns the lifetime manager used by the memory manager
     *
     * @return The lifetime manager
     */
    virtual ILifetimeManager *lifetime_manager() = 0;
    /** Returns the pool manager used by the memory manager
     *
     * @return The pool manager
     */
    virtual IPoolManager *pool_manager() = 0;
    /** Populates the pool manager with the given number of pools
     *
     * @pre Pool manager must be empty
     *
     * @param[in] allocator Allocator to use for the backing allocations
     * @param[in] num_pools Number of pools to create
     */
    virtual void populate(IAllocator &allocator, size_t num_pools) = 0;
    /** Clears the pool manager
     *
     * @pre All pools must be unoccupied
     */
    virtual void clear() = 0;
};
} // arm_compute
#endif /*ARM_COMPUTE_IMEMORYMANAGER_H */
