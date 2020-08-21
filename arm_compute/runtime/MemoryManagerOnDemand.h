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
#ifndef ARM_COMPUTE_MEMORY_MANAGER_ON_DEMAND_H
#define ARM_COMPUTE_MEMORY_MANAGER_ON_DEMAND_H

#include "arm_compute/runtime/IMemoryManager.h"

#include "arm_compute/runtime/ILifetimeManager.h"
#include "arm_compute/runtime/IMemoryGroup.h"
#include "arm_compute/runtime/IPoolManager.h"

#include <memory>

namespace arm_compute
{
/** On-demand memory manager */
class MemoryManagerOnDemand : public IMemoryManager
{
public:
    /** Default Constructor */
    MemoryManagerOnDemand(std::shared_ptr<ILifetimeManager> lifetime_manager, std::shared_ptr<IPoolManager> pool_manager);
    /** Prevent instances of this class to be copy constructed */
    MemoryManagerOnDemand(const MemoryManagerOnDemand &) = delete;
    /** Prevent instances of this class to be copied */
    MemoryManagerOnDemand &operator=(const MemoryManagerOnDemand &) = delete;
    /** Allow instances of this class to be move constructed */
    MemoryManagerOnDemand(MemoryManagerOnDemand &&) = default;
    /** Allow instances of this class to be moved */
    MemoryManagerOnDemand &operator=(MemoryManagerOnDemand &&) = default;

    // Inherited methods overridden:
    ILifetimeManager *lifetime_manager() override;
    IPoolManager     *pool_manager() override;
    void populate(IAllocator &allocator, size_t num_pools) override;
    void clear() override;

private:
    std::shared_ptr<ILifetimeManager> _lifetime_mgr; /**< Lifetime manager */
    std::shared_ptr<IPoolManager>     _pool_mgr;     /**< Memory pool manager */
};
} // arm_compute
#endif /*ARM_COMPUTE_MEMORY_MANAGER_ON_DEMAND_H */
