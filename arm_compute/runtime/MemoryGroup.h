/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_MEMORYGROUP_H
#define ARM_COMPUTE_MEMORYGROUP_H

#include "arm_compute/runtime/IMemoryGroup.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/misc/Macros.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IMemoryPool.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
// Forward declarations
class IMemory;

/** Memory group */
class MemoryGroup final : public IMemoryGroup
{
public:
    /** Default Constructor */
    MemoryGroup(std::shared_ptr<IMemoryManager> = nullptr) noexcept;
    /** Default destructor */
    ~MemoryGroup() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MemoryGroup(const MemoryGroup &) = delete;
    /** Prevent instances of this class from being copy assigned (As this class contains pointers) */
    MemoryGroup &operator=(const MemoryGroup &) = delete;
    /** Allow instances of this class to be moved */
    MemoryGroup(MemoryGroup &&) = default;
    /** Allow instances of this class to be moved */
    MemoryGroup &operator=(MemoryGroup &&) = default;

    // Inherited methods overridden:
    void manage(IMemoryManageable *obj) override;
    void finalize_memory(IMemoryManageable *obj, IMemory &obj_memory, size_t size, size_t alignment) override;
    void            acquire() override;
    void            release() override;
    MemoryMappings &mappings() override;

private:
    std::shared_ptr<IMemoryManager> _memory_manager; /**< Memory manager to be used by the group */
    IMemoryPool                    *_pool;           /**< Memory pool that the group is scheduled with */
    MemoryMappings                  _mappings;       /**< Memory mappings of the group */
};

inline MemoryGroup::MemoryGroup(std::shared_ptr<IMemoryManager> memory_manager) noexcept
    : _memory_manager(memory_manager),
      _pool(nullptr),
      _mappings()
{
}

inline void MemoryGroup::manage(IMemoryManageable *obj)
{
    if(_memory_manager && (obj != nullptr))
    {
        ARM_COMPUTE_ERROR_ON(!_memory_manager->lifetime_manager());

        // Defer registration to the first managed object
        _memory_manager->lifetime_manager()->register_group(this);

        // Associate this memory group with the tensor
        obj->associate_memory_group(this);

        // Start object lifetime
        _memory_manager->lifetime_manager()->start_lifetime(obj);
    }
}

inline void MemoryGroup::finalize_memory(IMemoryManageable *obj, IMemory &obj_memory, size_t size, size_t alignment)
{
    if(_memory_manager)
    {
        ARM_COMPUTE_ERROR_ON(!_memory_manager->lifetime_manager());
        _memory_manager->lifetime_manager()->end_lifetime(obj, obj_memory, size, alignment);
    }
}

inline void MemoryGroup::acquire()
{
    if(!_mappings.empty())
    {
        ARM_COMPUTE_ERROR_ON(!_memory_manager->pool_manager());
        _pool = _memory_manager->pool_manager()->lock_pool();
        _pool->acquire(_mappings);
    }
}

inline void MemoryGroup::release()
{
    if(_pool != nullptr)
    {
        ARM_COMPUTE_ERROR_ON(!_memory_manager->pool_manager());
        ARM_COMPUTE_ERROR_ON(_mappings.empty());
        _pool->release(_mappings);
        _memory_manager->pool_manager()->unlock_pool(_pool);
        _pool = nullptr;
    }
}

inline MemoryMappings &MemoryGroup::mappings()
{
    return _mappings;
}
} // arm_compute
#endif /*ARM_COMPUTE_MEMORYGROUP_H */
