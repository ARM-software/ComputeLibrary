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
#ifndef ARM_COMPUTE_BLOBLIFETIMEMANAGER_H
#define ARM_COMPUTE_BLOBLIFETIMEMANAGER_H

#include "arm_compute/runtime/ISimpleLifetimeManager.h"

#include "arm_compute/runtime/Types.h"

#include <memory>
#include <vector>

namespace arm_compute
{
// Forward declarations
class IMemoryPool;

/** Concrete class that tracks the lifetime of registered tensors and
 *  calculates the systems memory requirements in terms of blobs */
class BlobLifetimeManager : public ISimpleLifetimeManager
{
public:
    using info_type = std::vector<BlobInfo>;

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
    /** Accessor to the pool internal configuration meta-data
     *
     * @return Lifetime manager internal configuration meta-data
     */
    const info_type &info() const;

    // Inherited methods overridden:
    std::unique_ptr<IMemoryPool> create_pool(IAllocator *allocator) override;
    MappingType mapping_type() const override;

private:
    // Inherited methods overridden:
    void update_blobs_and_mappings() override;

private:
    std::vector<BlobInfo> _blobs; /**< Memory blobs */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_BLOBLIFETIMEMANAGER_H */
