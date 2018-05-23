/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_RUNTIME_MEMORY_REGION_H__
#define __ARM_COMPUTE_RUNTIME_MEMORY_REGION_H__

#include "arm_compute/runtime/IMemoryRegion.h"

#include "arm_compute/core/Error.h"

#include <cstddef>

namespace arm_compute
{
/** Memory region CPU implementation */
class MemoryRegion final : public IMemoryRegion
{
public:
    /** Default constructor
     *
     * @param[in] size Region size
     */
    MemoryRegion(size_t size)
        : IMemoryRegion(size), _mem(nullptr), _ptr(nullptr)
    {
        if(size != 0)
        {
            _mem = std::shared_ptr<uint8_t>(new uint8_t[size](), [](uint8_t *ptr)
            {
                delete[] ptr;
            });
            _ptr = _mem.get();
        }
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MemoryRegion(const MemoryRegion &) = delete;
    /** Default move constructor */
    MemoryRegion(MemoryRegion &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MemoryRegion &operator=(const MemoryRegion &) = delete;
    /** Default move assignment operator */
    MemoryRegion &operator=(MemoryRegion &&) = default;

    // Inherited methods overridden :
    void *buffer() final
    {
        return _mem.get();
    }
    void *buffer() const final
    {
        return _mem.get();
    }
    void **handle() final
    {
        return reinterpret_cast<void **>(&_mem);
    }

protected:
    std::shared_ptr<uint8_t> _mem;
    uint8_t                 *_ptr;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_RUNTIME_MEMORY_REGION_H__ */
