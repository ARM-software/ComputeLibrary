/*
 * Copyright (c) 2016-2018 Arm Limited.
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

namespace arm_compute
{
MemoryManagerOnDemand::MemoryManagerOnDemand(std::shared_ptr<ILifetimeManager> lifetime_manager,
                                             std::shared_ptr<IPoolManager>     pool_manager)
    : _lifetime_mgr(std::move(lifetime_manager)), _pool_mgr(std::move(pool_manager))
{
    ARM_COMPUTE_ERROR_ON_MSG(!_lifetime_mgr, "Lifetime manager not specified correctly!");
    ARM_COMPUTE_ERROR_ON_MSG(!_pool_mgr, "Pool manager not specified correctly!");
}

ILifetimeManager *MemoryManagerOnDemand::lifetime_manager()
{
    return _lifetime_mgr.get();
}

IPoolManager *MemoryManagerOnDemand::pool_manager()
{
    return _pool_mgr.get();
}

void MemoryManagerOnDemand::populate(arm_compute::IAllocator &allocator, size_t num_pools)
{
    ARM_COMPUTE_ERROR_ON(!_lifetime_mgr);
    ARM_COMPUTE_ERROR_ON(!_pool_mgr);
    ARM_COMPUTE_ERROR_ON_MSG(!_lifetime_mgr->are_all_finalized(), "All the objects have not been finalized!");
    ARM_COMPUTE_ERROR_ON_MSG(_pool_mgr->num_pools() != 0, "Pool manager already contains pools!");

    // Create pools
    auto pool_template = _lifetime_mgr->create_pool(&allocator);
    for (int i = num_pools; i > 1; --i)
    {
        auto pool = pool_template->duplicate();
        _pool_mgr->register_pool(std::move(pool));
    }
    _pool_mgr->register_pool(std::move(pool_template));
}

void MemoryManagerOnDemand::clear()
{
    ARM_COMPUTE_ERROR_ON_MSG(!_pool_mgr, "Pool manager not specified correctly!");
    _pool_mgr->clear_pools();
}
} //namespace arm_compute
