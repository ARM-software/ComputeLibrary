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
#ifndef __ARM_COMPUTE_OFFSETMEMORYPOOL_H__
#define __ARM_COMPUTE_OFFSETMEMORYPOOL_H__

#include "arm_compute/runtime/IMemoryPool.h"

#include "arm_compute/runtime/IMemoryRegion.h"
#include "arm_compute/runtime/Types.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
class IAllocator;

/** Offset based memory pool */
class OffsetMemoryPool : public IMemoryPool
{
public:
    /** Default Constructor
     *
     * @note allocator should outlive the memory pool
     *
     * @param[in] allocator Backing memory allocator
     * @param[in] blob_info Configuration information of the blob to be allocated
     */
    OffsetMemoryPool(IAllocator *allocator, BlobInfo blob_info);
    /** Default Destructor */
    ~OffsetMemoryPool() = default;
    /** Prevent instances of this class to be copy constructed */
    OffsetMemoryPool(const OffsetMemoryPool &) = delete;
    /** Prevent instances of this class to be copy assigned */
    OffsetMemoryPool &operator=(const OffsetMemoryPool &) = delete;
    /** Allow instances of this class to be move constructed */
    OffsetMemoryPool(OffsetMemoryPool &&) = default;
    /** Allow instances of this class to be move assigned */
    OffsetMemoryPool &operator=(OffsetMemoryPool &&) = default;

    // Inherited methods overridden:
    void acquire(MemoryMappings &handles) override;
    void release(MemoryMappings &handles) override;
    MappingType                  mapping_type() const override;
    std::unique_ptr<IMemoryPool> duplicate() override;

private:
    IAllocator                    *_allocator; /**< Allocator to use for internal allocation */
    std::unique_ptr<IMemoryRegion> _blob;      /**< Memory blob */
    BlobInfo                       _blob_info; /**< Information for the blob to allocate */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_OFFSETMEMORYPOOL_H__ */
