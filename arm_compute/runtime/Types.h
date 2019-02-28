/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_RUNTIME_TYPES_H__
#define __ARM_COMPUTE_RUNTIME_TYPES_H__

#include "arm_compute/runtime/IMemory.h"

#include <map>

namespace arm_compute
{
/** Mapping type */
enum class MappingType
{
    BLOBS,  /**< Mappings are in blob granularity */
    OFFSETS /**< Mappings are in offset granularity in the same blob */
};

/** A map of (handle, index/offset), where handle is the memory handle of the object
 * to provide the memory for and index/offset is the buffer/offset from the pool that should be used
 *
 * @note All objects are pre-pinned to specific buffers to avoid any relevant overheads
 */
using MemoryMappings = std::map<IMemory *, size_t>;

/** A map of the groups and memory mappings */
using GroupMappings = std::map<size_t, MemoryMappings>;

/** Meta-data information for each blob */
struct BlobInfo
{
    BlobInfo(size_t size_ = 0, size_t alignment_ = 0, size_t owners_ = 1)
        : size(size_), alignment(alignment_), owners(owners_)
    {
    }
    size_t size;      /**< Blob size */
    size_t alignment; /**< Blob alignment */
    size_t owners;    /**< Number of owners in parallel of the blob */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_RUNTIME_TYPES_H__ */
