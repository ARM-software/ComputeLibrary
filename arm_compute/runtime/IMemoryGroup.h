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
#ifndef ARM_COMPUTE_IMEMORYGROUP_H
#define ARM_COMPUTE_IMEMORYGROUP_H

#include "arm_compute/runtime/IMemory.h"
#include "arm_compute/runtime/Types.h"

namespace arm_compute
{
// Forward declarations
class IMemoryGroup;
class IMemoryManageable;

/** Memory group interface */
class IMemoryGroup
{
public:
    /** Default virtual destructor */
    virtual ~IMemoryGroup() = default;
    /** Sets a object to be managed by the given memory group
     *
     * @note Manager must not be finalized
     *
     * @param[in] obj Object to be managed
     */
    virtual void manage(IMemoryManageable *obj) = 0;
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
    virtual void finalize_memory(IMemoryManageable *obj, IMemory &obj_memory, size_t size, size_t alignment) = 0;
    /** Acquires backing memory for the whole group */
    virtual void acquire() = 0;
    /** Releases backing memory of the whole group */
    virtual void release() = 0;
    /** Gets the memory mapping of the group */
    virtual MemoryMappings &mappings() = 0;
};

/** Interface of an object than can be memory managed */
class IMemoryManageable
{
public:
    /** Default virtual destructor */
    virtual ~IMemoryManageable() = default;
    /** Associates a memory managable object with the memory group that manages it
     *
     * @param[in] memory_group Memory group that manages the object.
     */
    virtual void associate_memory_group(IMemoryGroup *memory_group) = 0;
};

/** Memory group resources scope handling class */
class MemoryGroupResourceScope
{
public:
    /** Constructor
     *
     * @param[in] memory_group Memory group to handle
     */
    explicit MemoryGroupResourceScope(IMemoryGroup &memory_group)
        : _memory_group(memory_group)
    {
        _memory_group.acquire();
    }
    /** Destructor */
    ~MemoryGroupResourceScope()
    {
        _memory_group.release();
    }

private:
    IMemoryGroup &_memory_group;
};
} // arm_compute
#endif /*ARM_COMPUTE_IMEMORYGROUP_H */
