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
#ifndef __ARM_COMPUTE_IPOOLMANAGER_H__
#define __ARM_COMPUTE_IPOOLMANAGER_H__

#include <memory>

namespace arm_compute
{
class IMemoryPool;

/** Memory pool manager interface */
class IPoolManager
{
public:
    /** Default virtual destructor */
    virtual ~IPoolManager() = default;
    /** Locks a pool for execution
     *
     * @return Locked pool that workload will be mapped on
     */
    virtual IMemoryPool *lock_pool() = 0;
    /** Releases memory pool
     *
     * @param[in] pool Memory pool to release
     */
    virtual void unlock_pool(IMemoryPool *pool) = 0;
    /** Register pool to be managed by the pool
     *
     * @note Ownership of the pools is being transferred to the pool manager
     *
     * @param[in] pool Pool to be managed
     */
    virtual void register_pool(std::unique_ptr<IMemoryPool> pool) = 0;
};
} // arm_compute
#endif /*__ARM_COMPUTE_IPOOLMANAGER_H__ */
