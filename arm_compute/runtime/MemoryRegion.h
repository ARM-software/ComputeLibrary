/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_RUNTIME_MEMORY_REGION_H
#define ARM_COMPUTE_RUNTIME_MEMORY_REGION_H

#include "arm_compute/runtime/IMemoryRegion.h"

#include "arm_compute/core/Error.h"

#include <cstddef>

namespace arm_compute
{
/** Memory region CPU implementation */
class MemoryRegion final : public IMemoryRegion
{
public:
    /** Constructor
     *
     * @param[in] size      Region size
     * @param[in] alignment Alignment in bytes of the base pointer. Defaults to 0
     */
    MemoryRegion(size_t size, size_t alignment = 0)
        : IMemoryRegion(size), _mem(nullptr), _ptr(nullptr)
    {
        if(size != 0)
        {
            // Allocate backing memory
            size_t space = size + alignment;
            _mem         = std::shared_ptr<uint8_t>(new uint8_t[space](), [](uint8_t *ptr)
            {
                delete[] ptr;
            });
            _ptr = _mem.get();

            // Calculate alignment offset
            if(alignment != 0)
            {
                void *aligned_ptr = _mem.get();
                std::align(alignment, size, aligned_ptr, space);
                _ptr = aligned_ptr;
            }
        }
    }
    MemoryRegion(void *ptr, size_t size)
        : IMemoryRegion(size), _mem(nullptr), _ptr(nullptr)
    {
        if(size != 0)
        {
            _ptr = ptr;
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
        return _ptr;
    }
    const void *buffer() const final
    {
        return _ptr;
    }
    std::unique_ptr<IMemoryRegion> extract_subregion(size_t offset, size_t size) final
    {
        if(_ptr != nullptr && (offset < _size) && (_size - offset >= size))
        {
            return std::make_unique<MemoryRegion>(static_cast<uint8_t *>(_ptr) + offset, size);
        }
        else
        {
            return nullptr;
        }
    }

protected:
    std::shared_ptr<uint8_t> _mem;
    void                    *_ptr;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_RUNTIME_MEMORY_REGION_H */
