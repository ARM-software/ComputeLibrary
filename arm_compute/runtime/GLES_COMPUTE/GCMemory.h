/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_RUNTIME_GLES_COMPUTE_GCMEMORY_H
#define ARM_COMPUTE_RUNTIME_GLES_COMPUTE_GCMEMORY_H

#include "arm_compute/runtime/IMemory.h"

#include "arm_compute/runtime/GLES_COMPUTE/GCMemoryRegion.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
/** GLES implementation of memory object */
class GCMemory : public IMemory
{
public:
    /** Default Constructor */
    GCMemory();
    /** Default Constructor
     *
     * @param[in] memory Memory to be imported
     */
    GCMemory(const std::shared_ptr<IGCMemoryRegion> &memory);
    /** Default Constructor
     *
     * @note Ownership of the memory is not transferred to this object.
     *       Thus management (allocate/free) should be done by the client.
     *
     * @param[in] memory Memory to be imported
     */
    GCMemory(IGCMemoryRegion *memory);
    /** Allow instances of this class to be copied */
    GCMemory(const GCMemory &) = default;
    /** Allow instances of this class to be copy assigned */
    GCMemory &operator=(const GCMemory &) = default;
    /** Allow instances of this class to be moved */
    GCMemory(GCMemory &&) noexcept = default;
    /** Allow instances of this class to be move assigned */
    GCMemory &operator=(GCMemory &&) noexcept = default;
    /** GLES Region accessor
     *
     * @return Memory region
     */
    IGCMemoryRegion *gc_region();
    /** GLES Region accessor
     *
     * @return Memory region
     */
    IGCMemoryRegion *gc_region() const;

    // Inherited methods overridden:
    IMemoryRegion *region() final;
    IMemoryRegion *region() const final;
    void set_region(IMemoryRegion *region) final;
    void set_owned_region(std::unique_ptr<IMemoryRegion> region) final;

private:
    IGCMemoryRegion                 *_region;
    std::shared_ptr<IGCMemoryRegion> _region_owned;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_RUNTIME_GLES_COMPUTE_GCMEMORY_H */
