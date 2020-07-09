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
#ifndef ARM_COMPUTE_BLOBMEMORYPOOL_H
#define ARM_COMPUTE_BLOBMEMORYPOOL_H

#include "arm_compute/runtime/IMemoryPool.h"

#include "arm_compute/runtime/IMemoryRegion.h"
#include "arm_compute/runtime/Types.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace arm_compute
{
// Forward declaration
class IAllocator;

/** Blob memory pool */
class BlobMemoryPool : public IMemoryPool
{
public:
    /** Default Constructor
     *
     * @note allocator should outlive the memory pool
     *
     * @param[in] allocator Backing memory allocator
     * @param[in] blob_info Configuration information of the blobs to be allocated
     */
    BlobMemoryPool(IAllocator *allocator, std::vector<BlobInfo> blob_info);
    /** Default Destructor */
    ~BlobMemoryPool();
    /** Prevent instances of this class to be copy constructed */
    BlobMemoryPool(const BlobMemoryPool &) = delete;
    /** Prevent instances of this class to be copy assigned */
    BlobMemoryPool &operator=(const BlobMemoryPool &) = delete;
    /** Allow instances of this class to be move constructed */
    BlobMemoryPool(BlobMemoryPool &&) = default;
    /** Allow instances of this class to be move assigned */
    BlobMemoryPool &operator=(BlobMemoryPool &&) = default;

    // Inherited methods overridden:
    void acquire(MemoryMappings &handles) override;
    void release(MemoryMappings &handles) override;
    MappingType                  mapping_type() const override;
    std::unique_ptr<IMemoryPool> duplicate() override;

private:
    /** Allocates internal blobs
     *
     * @param blob_info Size of each blob
     */
    void allocate_blobs(const std::vector<BlobInfo> &blob_info);
    /** Frees blobs **/
    void free_blobs();

private:
    IAllocator                                 *_allocator; /**< Allocator to use for internal allocation */
    std::vector<std::unique_ptr<IMemoryRegion>> _blobs;     /**< Vector holding all the memory blobs */
    std::vector<BlobInfo>                       _blob_info; /**< Information of each blob */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_BLOBMEMORYPOOL_H */
