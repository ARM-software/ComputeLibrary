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
#ifndef __ARM_COMPUTE_IMEMORYMANAGER_H__
#define __ARM_COMPUTE_IMEMORYMANAGER_H__

#include "arm_compute/runtime/ILifetimeManager.h"
#include "arm_compute/runtime/IPoolManager.h"

#include <cstddef>

namespace arm_compute
{
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
    /** Finalize memory manager */
    virtual void finalize() = 0;
};
} // arm_compute
#endif /*__ARM_COMPUTE_IMEMORYMANAGER_H__ */
