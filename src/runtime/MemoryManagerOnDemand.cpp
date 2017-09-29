/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/runtime/MemoryManagerOnDemand.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/ILifetimeManager.h"
#include "arm_compute/runtime/IPoolManager.h"

#include <memory>

using namespace arm_compute;

MemoryManagerOnDemand::MemoryManagerOnDemand(std::shared_ptr<ILifetimeManager> lifetime_manager, std::shared_ptr<IPoolManager> pool_manager)
    : _lifetime_mgr(std::move(lifetime_manager)), _pool_mgr(std::move(pool_manager)), _allocator(nullptr), _is_finalized(false), _num_pools(1)
{
    ARM_COMPUTE_ERROR_ON_MSG(!_lifetime_mgr, "Lifetime manager not specified correctly!");
    ARM_COMPUTE_ERROR_ON_MSG(!_pool_mgr, "Pool manager not specified correctly!");
}

bool MemoryManagerOnDemand::is_finalized() const
{
    return _is_finalized;
}

void MemoryManagerOnDemand::set_num_pools(unsigned int num_pools)
{
    ARM_COMPUTE_ERROR_ON(num_pools == 0);
    _num_pools = num_pools;
}

void MemoryManagerOnDemand::set_allocator(IAllocator *allocator)
{
    ARM_COMPUTE_ERROR_ON_MSG(is_finalized(), "Memory manager is already finalized!");
    ARM_COMPUTE_ERROR_ON(allocator == nullptr);
    _allocator = allocator;
}

ILifetimeManager *MemoryManagerOnDemand::lifetime_manager()
{
    return _lifetime_mgr.get();
}

IPoolManager *MemoryManagerOnDemand::pool_manager()
{
    return _pool_mgr.get();
}

void MemoryManagerOnDemand::finalize()
{
    ARM_COMPUTE_ERROR_ON_MSG(is_finalized(), "Memory manager is already finalized!");
    ARM_COMPUTE_ERROR_ON(!_lifetime_mgr);
    ARM_COMPUTE_ERROR_ON(!_pool_mgr);
    ARM_COMPUTE_ERROR_ON_MSG(!_lifetime_mgr->are_all_finalized(), "All the objects have not been finalized! ");
    ARM_COMPUTE_ERROR_ON(_allocator == nullptr);

    // Create pools
    auto pool_template = _lifetime_mgr->create_pool(_allocator);
    for(int i = _num_pools; i > 1; --i)
    {
        auto pool = pool_template->duplicate();
        _pool_mgr->register_pool(std::move(pool));
    }
    _pool_mgr->register_pool(std::move(pool_template));

    // Set finalized to true
    _is_finalized = true;
}
