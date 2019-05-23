/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_IMEMORYGROUP_H__
#define __ARM_COMPUTE_IMEMORYGROUP_H__

#include "arm_compute/runtime/Types.h"

namespace arm_compute
{
/** Memory group interface */
class IMemoryGroup
{
public:
    /** Default virtual destructor */
    virtual ~IMemoryGroup() = default;
    /** Acquires backing memory for the whole group */
    virtual void acquire() = 0;
    /** Releases backing memory of the whole group */
    virtual void release() = 0;
    /** Gets the memory mapping of the group */
    virtual MemoryMappings &mappings() = 0;
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
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MemoryGroupResourceScope(const MemoryGroupResourceScope &) = delete;
    /** Default move constructor */
    MemoryGroupResourceScope(MemoryGroupResourceScope &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MemoryGroupResourceScope &operator=(const MemoryGroupResourceScope &) = delete;
    /** Default move assignment operator */
    MemoryGroupResourceScope &operator=(MemoryGroupResourceScope &&) = default;
    /** Destructor */
    ~MemoryGroupResourceScope()
    {
        _memory_group.release();
    }

private:
    IMemoryGroup &_memory_group;
};
} // arm_compute
#endif /*__ARM_COMPUTE_IMEMORYGROUP_H__ */
