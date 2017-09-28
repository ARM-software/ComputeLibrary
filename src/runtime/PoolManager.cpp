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
#include "arm_compute/runtime/PoolManager.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IMemoryPool.h"
#include "support/ToolchainSupport.h"

#include <list>

using namespace arm_compute;

PoolManager::PoolManager()
    : _free_pools(), _occupied_pools(), _sem(), _mtx()
{
}

IMemoryPool *PoolManager::lock_pool()
{
    ARM_COMPUTE_ERROR_ON_MSG(_free_pools.empty() && _occupied_pools.empty(), "Haven't setup any pools!");

    _sem->wait();
    std::lock_guard<arm_compute::Mutex> lock(_mtx);
    ARM_COMPUTE_ERROR_ON_MSG(_free_pools.empty(), "Empty pool must exist as semaphore has been signalled");
    _occupied_pools.splice(std::begin(_occupied_pools), _free_pools, std::begin(_free_pools));
    return _occupied_pools.front().get();
}

void PoolManager::unlock_pool(IMemoryPool *pool)
{
    ARM_COMPUTE_ERROR_ON_MSG(_free_pools.empty() && _occupied_pools.empty(), "Haven't setup any pools!");

    std::lock_guard<arm_compute::Mutex> lock(_mtx);
    auto it = std::find_if(std::begin(_occupied_pools), std::end(_occupied_pools), [pool](const std::unique_ptr<IMemoryPool> &pool_it)
    {
        return pool_it.get() == pool;
    });
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_occupied_pools), "Pool to be unlocked couldn't be found!");
    _free_pools.splice(std::begin(_free_pools), _occupied_pools, it);
    _sem->signal();
}

void PoolManager::register_pool(std::unique_ptr<IMemoryPool> pool)
{
    std::lock_guard<arm_compute::Mutex> lock(_mtx);
    ARM_COMPUTE_ERROR_ON_MSG(!_occupied_pools.empty(), "All pools should be free in order to register a new one!");

    // Set pool
    _free_pools.push_front(std::move(pool));

    // Update semaphore
    _sem = arm_compute::support::cpp14::make_unique<arm_compute::Semaphore>(_free_pools.size());
}
