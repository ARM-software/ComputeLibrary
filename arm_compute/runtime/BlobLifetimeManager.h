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

#include "arm_compute/runtime/ILifetimeManager.h"

#include "arm_compute/runtime/IMemoryGroup.h"
#include "arm_compute/runtime/Types.h"

#include <cstddef>
#include <map>
#include <vector>

namespace arm_compute
{
class IMemoryGroup;

/** Class that tracks the lifetime of registered tensors and calculates the systems memory requirements in terms of blobs */
class BlobLifetimeManager : public ILifetimeManager
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
    void register_group(IMemoryGroup *group) override;
    void start_lifetime(void *obj) override;
    void end_lifetime(void *obj, void **handle, size_t size) override;
    std::unique_ptr<IMemoryPool> create_pool(IAllocator *allocator) override;
    bool        are_all_finalized() const override;
    MappingType mapping_type() const override;

private:
    /** Update blobs and mappings */
    void update_blobs_and_mappings();

private:
    /** Element struct */
    struct Element
    {
        Element(void *id_ = nullptr, void **handle_ = nullptr, size_t size_ = 0, bool status_ = false)
            : id(id_), handle(handle_), size(size_), status(status_)
        {
        }
        void *id;      /**< Element id */
        void **handle; /**< Element's memory handle */
        size_t size;   /**< Element's size */
        bool   status; /**< Lifetime status */
    };

    IMemoryGroup        *_active_group;                               /**< Active group */
    std::vector<Element> _active_elements;                            /**< A map that contains the active elements */
    std::map<IMemoryGroup *, std::vector<Element>> _finalized_groups; /**< A map that contains the finalized groups */
    std::vector<size_t> _blobs;
};
} // arm_compute
#endif /* __ARM_COMPUTE_BLOBLIFETIMEMANAGER_H__ */
