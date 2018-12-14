/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_MEMORYGROUPBASE_H__
#define __ARM_COMPUTE_MEMORYGROUPBASE_H__

#include "arm_compute/runtime/IMemoryGroup.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IMemoryPool.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
// Forward declarations
class IMemory;

/** Memory group */
template <typename TensorType>
class MemoryGroupBase : public IMemoryGroup
{
public:
    /** Default Constructor */
    MemoryGroupBase(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Default destructor */
    ~MemoryGroupBase() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MemoryGroupBase(const MemoryGroupBase &) = delete;
    /** Prevent instances of this class from being copy assigned (As this class contains pointers) */
    MemoryGroupBase &operator=(const MemoryGroupBase &) = delete;
    /** Allow instances of this class to be moved */
    MemoryGroupBase(MemoryGroupBase &&) = default;
    /** Allow instances of this class to be moved */
    MemoryGroupBase &operator=(MemoryGroupBase &&) = default;
    /** Sets a object to be managed by the given memory group
     *
     * @note Manager must not be finalized
     *
     * @param[in] obj Object to be managed
     */
    void manage(TensorType *obj);
    /** Finalizes memory for a given object
     *
     * @note Manager must not be finalized
     *
     * @param[in, out] obj        Object to request memory for
     * @param[in, out] obj_memory Object's memory handling interface which can be used to alter the underlying memory
     *                            that is used by the object.
     * @param[in]      size       Size of memory to allocate
     * @param[in]      alignment  (Optional) Alignment to use
     */
    void finalize_memory(TensorType *obj, IMemory &obj_memory, size_t size, size_t alignment = 0);

    // Inherited methods overridden:
    void            acquire() override;
    void            release() override;
    MemoryMappings &mappings() override;

private:
    void associate_memory_group(TensorType *obj);

private:
    std::shared_ptr<IMemoryManager> _memory_manager; /**< Memory manager to be used by the group */
    IMemoryPool                    *_pool;           /**< Memory pool that the group is scheduled with */
    MemoryMappings                  _mappings;       /**< Memory mappings of the group */
};

template <typename TensorType>
inline MemoryGroupBase<TensorType>::MemoryGroupBase(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_manager(std::move(memory_manager)), _pool(nullptr), _mappings()
{
    if(_memory_manager)
    {
        ARM_COMPUTE_ERROR_ON(!_memory_manager->lifetime_manager());
    }
}

template <typename TensorType>
inline void MemoryGroupBase<TensorType>::manage(TensorType *obj)
{
    if(_memory_manager && _mappings.empty())
    {
        ARM_COMPUTE_ERROR_ON(!_memory_manager->lifetime_manager());

        // Defer registration to the first managed object
        _memory_manager->lifetime_manager()->register_group(this);

        // Associate this memory group with the tensor
        associate_memory_group(obj);

        // Start object lifetime
        _memory_manager->lifetime_manager()->start_lifetime(obj);
    }
}

template <typename TensorType>
inline void MemoryGroupBase<TensorType>::finalize_memory(TensorType *obj, IMemory &obj_memory, size_t size, size_t alignment)
{
    // TODO (geopin01) : Check size (track size in MemoryMappings)
    // Check if existing mapping is valid
    ARM_COMPUTE_ERROR_ON(!_mappings.empty() && (_mappings.find(&obj_memory) == std::end(_mappings)));

    if(_memory_manager && _mappings.empty())
    {
        ARM_COMPUTE_ERROR_ON(!_memory_manager->lifetime_manager());
        _memory_manager->lifetime_manager()->end_lifetime(obj, obj_memory, size, alignment);
    }
}

template <typename TensorType>
inline void MemoryGroupBase<TensorType>::acquire()
{
    if(!_mappings.empty())
    {
        ARM_COMPUTE_ERROR_ON(!_memory_manager->pool_manager());
        _pool = _memory_manager->pool_manager()->lock_pool();
        _pool->acquire(_mappings);
    }
}

template <typename TensorType>
inline void MemoryGroupBase<TensorType>::release()
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

template <typename TensorType>
inline MemoryMappings &MemoryGroupBase<TensorType>::mappings()
{
    return _mappings;
}

template <typename TensorType>
inline void MemoryGroupBase<TensorType>::associate_memory_group(TensorType *)
{
    ARM_COMPUTE_ERROR("Must be implemented by child class");
}
} // arm_compute
#endif /*__ARM_COMPUTE_MEMORYGROUPBASE_H__ */
