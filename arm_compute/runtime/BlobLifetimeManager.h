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
#ifndef __ARM_COMPUTE_BLOBLIFETIMEMANAGER_H__
#define __ARM_COMPUTE_BLOBLIFETIMEMANAGER_H__

#include "arm_compute/runtime/ISimpleLifetimeManager.h"

#include "arm_compute/runtime/IMemoryPool.h"
#include "arm_compute/runtime/Types.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace arm_compute
{
/** Concrete class that tracks the lifetime of registered tensors and
 *  calculates the systems memory requirements in terms of blobs */
class BlobLifetimeManager : public ISimpleLifetimeManager
{
public:
    /** Constructor */
    BlobLifetimeManager();
    /** Prevent instances of this class to be copy constructed */
    BlobLifetimeManager(const BlobLifetimeManager &) = delete;
    /** Prevent instances of this class to be copied */
    BlobLifetimeManager &operator=(const BlobLifetimeManager &) = delete;
    /** Allow instances of this class to be move constructed */
    BlobLifetimeManager(BlobLifetimeManager &&) = default;
    /** Allow instances of this class to be moved */
    BlobLifetimeManager &operator=(BlobLifetimeManager &&) = default;

    // Inherited methods overridden:
    std::unique_ptr<IMemoryPool> create_pool(IAllocator *allocator) override;
    MappingType mapping_type() const override;

private:
    // Inherited methods overridden:
    void update_blobs_and_mappings() override;

private:
    std::vector<size_t> _blobs; /**< Memory blobs' sizes */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_BLOBLIFETIMEMANAGER_H__ */
