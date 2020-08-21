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
#ifndef ARM_COMPUTE_RUNTIME_GLES_COMPUTE_GC_MEMORY_REGION_H
#define ARM_COMPUTE_RUNTIME_GLES_COMPUTE_GC_MEMORY_REGION_H

#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/runtime/IMemoryRegion.h"

#include <cstddef>

namespace arm_compute
{
/** GLES memory region interface */
class IGCMemoryRegion : public IMemoryRegion
{
public:
    /** Constructor
     *
     * @param[in] size Region size
     */
    IGCMemoryRegion(size_t size);
    /** Default Destructor */
    virtual ~IGCMemoryRegion() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    IGCMemoryRegion(const IGCMemoryRegion &) = delete;
    /** Default move constructor */
    IGCMemoryRegion(IGCMemoryRegion &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    IGCMemoryRegion &operator=(const IGCMemoryRegion &) = delete;
    /** Default move assignment operator */
    IGCMemoryRegion &operator=(IGCMemoryRegion &&) = default;
    /** Returns the underlying CL buffer
     *
     * @return CL memory buffer object
     */
    const GLuint &gc_ssbo_name() const;
    /** Host/SVM pointer accessor
     *
     * @return Host/SVM pointer base
     */
    virtual void *ptr() = 0;
    /** Enqueue a map operation of the allocated buffer on the given queue.
     *
     * @param[in] blocking If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed before using the returned mapping pointer.
     *
     * @return The mapping address.
     */
    virtual void *map(bool blocking) = 0;
    /** Enqueue an unmap operation of the allocated buffer on the given queue.
     *
     * @note This method simply enqueue the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     *
     */
    virtual void unmap() = 0;

    // Inherited methods overridden :
    void       *buffer() override;
    const void *buffer() const override;

protected:
    void *_mapping;
    GLuint _ssbo_name;
};

/** GLES buffer memory region implementation */
class GCBufferMemoryRegion final : public IGCMemoryRegion
{
public:
    /** Constructor
     *
     * @param[in] size Region size
     */
    GCBufferMemoryRegion(size_t size);
    /** Destructor */
    ~GCBufferMemoryRegion();

    // Inherited methods overridden :
    void *ptr() final;
    void *map(bool blocking) final;
    void                           unmap() final;
    std::unique_ptr<IMemoryRegion> extract_subregion(size_t offset, size_t size) final;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_RUNTIME_GLES_COMPUTE_GC_MEMORY_REGION_H */
