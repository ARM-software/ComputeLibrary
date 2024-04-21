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
#ifndef ARM_COMPUTE_ILIFETIMEMANAGER_H
#define ARM_COMPUTE_ILIFETIMEMANAGER_H

#include "arm_compute/runtime/IMemoryPool.h"
#include "arm_compute/runtime/Types.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
// Forward declarations
class IAllocator;
class IMemory;
class IMemoryGroup;

/** Interface for managing the lifetime of objects */
class ILifetimeManager
{
public:
    /** Virtual Destructor */
    virtual ~ILifetimeManager() = default;
    /** Registers a group to the lifetime manager and assigns a group id
     *
     * @param[in] group The group id of the group
     */
    virtual void register_group(IMemoryGroup *group) = 0;
    /** Unbound and release elements associated with a group
     *
     * @param[in] group Group to unbound its elements
     *
     * @return True if group was registered and released else false.
     */
    virtual bool release_group(IMemoryGroup *group) = 0;
    /** Registers and starts lifetime of an object
     *
     * @param[in] obj Object to register
     */
    virtual void start_lifetime(void *obj) = 0;
    /** Ends lifetime of an object
     *
     * @param[in] obj        Object
     * @param[in] obj_memory Object memory
     * @param[in] size       Size of the given object at given time
     * @param[in] alignment  Alignment requirements for the object
     */
    virtual void end_lifetime(void *obj, IMemory &obj_memory, size_t size, size_t alignment) = 0;
    /** Checks if the lifetime of the registered object is complete
     *
     * @return True if all object lifetimes are finalized else false.
     */
    virtual bool are_all_finalized() const = 0;
    /** Creates a memory pool depending on the memory requirements
     *
     * @param allocator Allocator to use
     *
     * @return A memory pool
     */
    virtual std::unique_ptr<IMemoryPool> create_pool(IAllocator *allocator) = 0;
    /** Returns the type of mappings that the lifetime manager returns
     *
     * @return Mapping type of the lifetime manager
     */
    virtual MappingType mapping_type() const = 0;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_ILIFETIMEMANAGER_H */
