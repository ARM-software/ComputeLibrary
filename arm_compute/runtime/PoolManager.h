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
#ifndef __ARM_COMPUTE_POOLMANAGER_H__
#define __ARM_COMPUTE_POOLMANAGER_H__

#include "arm_compute/runtime/IPoolManager.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IMemoryPool.h"
#include "support/Mutex.h"
#include "support/Semaphore.h"

#include <cstddef>
#include <list>
#include <memory>

namespace arm_compute
{
/** Memory pool manager */
class PoolManager : public IPoolManager
{
public:
    /** Default Constructor */
    PoolManager();
    /** Prevent instances of this class to be copy constructed */
    PoolManager(const PoolManager &) = delete;
    /** Prevent instances of this class to be copied */
    PoolManager &operator=(const PoolManager &) = delete;
    /** Allow instances of this class to be move constructed */
    PoolManager(PoolManager &&) = default;
    /** Allow instances of this class to be moved */
    PoolManager &operator=(PoolManager &&) = default;

    // Inherited methods overridden:
    IMemoryPool *lock_pool() override;
    void unlock_pool(IMemoryPool *pool) override;
    void register_pool(std::unique_ptr<IMemoryPool> pool) override;

private:
    std::list<std::unique_ptr<IMemoryPool>> _free_pools;     /**< List of free pools */
    std::list<std::unique_ptr<IMemoryPool>> _occupied_pools; /**< List of occupied pools */
    std::unique_ptr<arm_compute::Semaphore> _sem;            /**< Semaphore to control the queues */
    arm_compute::Mutex                      _mtx;            /**< Mutex to control access to the queues */
};
} // arm_compute
#endif /*__ARM_COMPUTE_POOLMANAGER_H__ */
