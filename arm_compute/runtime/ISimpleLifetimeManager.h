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
#ifndef ARM_COMPUTE_ISIMPLELIFETIMEMANAGER_H
#define ARM_COMPUTE_ISIMPLELIFETIMEMANAGER_H

#include "arm_compute/runtime/ILifetimeManager.h"

#include "arm_compute/runtime/IMemoryPool.h"
#include "arm_compute/runtime/Types.h"

#include <cstddef>
#include <list>
#include <map>
#include <set>
#include <vector>

namespace arm_compute
{
class IAllocator;
class IMemoryGroup;

/** Abstract class of the simple lifetime manager interface */
class ISimpleLifetimeManager : public ILifetimeManager
{
public:
    /** Constructor */
    ISimpleLifetimeManager();
    /** Prevent instances of this class to be copy constructed */
    ISimpleLifetimeManager(const ISimpleLifetimeManager &) = delete;
    /** Prevent instances of this class to be copied */
    ISimpleLifetimeManager &operator=(const ISimpleLifetimeManager &) = delete;
    /** Allow instances of this class to be move constructed */
    ISimpleLifetimeManager(ISimpleLifetimeManager &&) = default;
    /** Allow instances of this class to be moved */
    ISimpleLifetimeManager &operator=(ISimpleLifetimeManager &&) = default;

    // Inherited methods overridden:
    void register_group(IMemoryGroup *group) override;
    bool release_group(IMemoryGroup *group) override;
    void start_lifetime(void *obj) override;
    void end_lifetime(void *obj, IMemory &obj_memory, size_t size, size_t alignment) override;
    bool are_all_finalized() const override;

protected:
    /** Update blobs and mappings */
    virtual void update_blobs_and_mappings() = 0;

protected:
    /** Element struct */
    struct Element
    {
        Element(void *id_ = nullptr, IMemory *handle_ = nullptr, size_t size_ = 0, size_t alignment_ = 0, bool status_ = false)
            : id(id_), handle(handle_), size(size_), alignment(alignment_), status(status_)
        {
        }
        void    *id;        /**< Element id */
        IMemory *handle;    /**< Element's memory handle */
        size_t   size;      /**< Element's size */
        size_t   alignment; /**< Alignment requirement */
        bool     status;    /**< Lifetime status */
    };

    /** Blob struct */
    struct Blob
    {
        void            *id;
        size_t           max_size;
        size_t           max_alignment;
        std::set<void *> bound_elements;
    };

    IMemoryGroup *_active_group;                                           /**< Active group */
    std::map<void *, Element> _active_elements;                            /**< A map that contains the active elements */
    std::list<Blob> _free_blobs;                                           /**< Free blobs */
    std::list<Blob> _occupied_blobs;                                       /**< Occupied blobs */
    std::map<IMemoryGroup *, std::map<void *, Element>> _finalized_groups; /**< A map that contains the finalized groups */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_ISIMPLELIFETIMEMANAGER_H */
