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
#ifndef __ARM_COMPUTE_IMEMORYPOOL_H__
#define __ARM_COMPUTE_IMEMORYPOOL_H__

#include "arm_compute/runtime/Types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace arm_compute
{
/** Memory Pool Inteface */
class IMemoryPool
{
public:
    /** Default Virtual Destructor */
    virtual ~IMemoryPool() = default;
    /** Sets occupant to the memory pool
     *
     * @param[in] handles A vector of pairs (handle, index)
     */
    virtual void acquire(MemoryMappings &handles) = 0;
    /** Releases a memory block
     *
     * @param[in] handles A vector containing a pair of handles and indices
     */
    virtual void release(MemoryMappings &handles) = 0;
    /** Returns the mapping types that this pool accepts
     *
     * @return the mapping type of the memory
     */
    virtual MappingType mapping_type() const = 0;
    /** Duplicates the existing memory pool
     *
     * @return A duplicate of the existing pool
     */
    virtual std::unique_ptr<IMemoryPool> duplicate() = 0;
};
} // arm_compute
#endif /* __ARM_COMPUTE_IMEMORYPOOL_H__ */
