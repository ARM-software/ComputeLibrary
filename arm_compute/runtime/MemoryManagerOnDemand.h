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
#ifndef __ARM_COMPUTE_MEMORYMANAGERONDEMAND_H__
#define __ARM_COMPUTE_MEMORYMANAGERONDEMAND_H__

#include "arm_compute/runtime/IMemoryManager.h"

#include "IAllocator.h"
#include "arm_compute/runtime/ILifetimeManager.h"
#include "arm_compute/runtime/IMemoryGroup.h"
#include "arm_compute/runtime/IPoolManager.h"

#include <memory>
#include <set>

namespace arm_compute
{
class IAllocator;

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
    /** Sets the number of pools to create
     *
     * @param[in] num_pools Number of pools
     */
    void set_num_pools(unsigned int num_pools);
    /** Sets the allocator to be used for configuring the pools
     *
     * @param[in] allocator Allocator to use
     */
    void set_allocator(IAllocator *allocator);
    /** Checks if the memory manager has been finalized
     *
     * @return True if the memory manager has been finalized else false
     */
    bool is_finalized() const;

    // Inherited methods overridden:
    ILifetimeManager *lifetime_manager() override;
    IPoolManager     *pool_manager() override;
    void              finalize() override;

private:
    std::shared_ptr<ILifetimeManager> _lifetime_mgr; /**< Lifetime manager */
    std::shared_ptr<IPoolManager>     _pool_mgr;     /**< Memory pool manager */
    IAllocator                       *_allocator;    /**< Allocator used for backend allocations */
    bool                              _is_finalized; /**< Flag that notes if the memory manager has been finalized */
    unsigned int                      _num_pools;    /**< Number of pools to create */
};
} // arm_compute
#endif /*__ARM_COMPUTE_MEMORYMANAGERONDEMAND_H__ */
