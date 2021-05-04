/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_RUNTIME_CL_CL_MEMORY_REGION_H
#define ARM_COMPUTE_RUNTIME_CL_CL_MEMORY_REGION_H

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/IMemoryRegion.h"

#include <cstddef>

namespace arm_compute
{
/** OpenCL memory region interface */
class ICLMemoryRegion : public IMemoryRegion
{
public:
    /** Constructor
     *
     * @param[in] size Region size
     */
    ICLMemoryRegion(size_t size);
    /** Default Destructor */
    virtual ~ICLMemoryRegion() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ICLMemoryRegion(const ICLMemoryRegion &) = delete;
    /** Default move constructor */
    ICLMemoryRegion(ICLMemoryRegion &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ICLMemoryRegion &operator=(const ICLMemoryRegion &) = delete;
    /** Default move assignment operator */
    ICLMemoryRegion &operator=(ICLMemoryRegion &&) = default;
    /** Returns the underlying CL buffer
     *
     * @return CL memory buffer object
     */
    const cl::Buffer &cl_data() const;
    /** Host/SVM pointer accessor
     *
     * @return Host/SVM pointer base
     */
    virtual void *ptr() = 0;
    /** Enqueue a map operation of the allocated buffer on the given queue.
     *
     * @param[in,out] q        The CL command queue to use for the mapping operation.
     * @param[in]     blocking If true, then the mapping will be ready to use by the time
     *                         this method returns, else it is the caller's responsibility
     *                         to flush the queue and wait for the mapping operation to have completed before using the returned mapping pointer.
     *
     * @return The mapping address.
     */
    virtual void *map(cl::CommandQueue &q, bool blocking) = 0;
    /** Enqueue an unmap operation of the allocated buffer on the given queue.
     *
     * @note This method simply enqueue the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     *
     * @param[in,out] q The CL command queue to use for the mapping operation.
     */
    virtual void unmap(cl::CommandQueue &q) = 0;

    // Inherited methods overridden :
    void                          *buffer() override;
    const void                    *buffer() const override;
    std::unique_ptr<IMemoryRegion> extract_subregion(size_t offset, size_t size) override;

protected:
    cl::CommandQueue _queue;
    cl::Context      _ctx;
    void            *_mapping;
    cl::Buffer       _mem;
};

/** OpenCL buffer memory region implementation */
class CLBufferMemoryRegion final : public ICLMemoryRegion
{
public:
    /** Constructor
     *
     * @param[in] flags Memory flags
     * @param[in] size  Region size
     */
    CLBufferMemoryRegion(cl_mem_flags flags, size_t size);
    /** Constructor
     *
     * @param[in] buffer Buffer to be used as a memory region
     */
    CLBufferMemoryRegion(const cl::Buffer &buffer);

    // Inherited methods overridden :
    void *ptr() final;
    void *map(cl::CommandQueue &q, bool blocking) final;
    void unmap(cl::CommandQueue &q) final;
};

/** OpenCL SVM memory region interface */
class ICLSVMMemoryRegion : public ICLMemoryRegion
{
protected:
    /** Constructor
     *
     * @param[in] flags     Memory flags
     * @param[in] size      Region size
     * @param[in] alignment Alignment
     */
    ICLSVMMemoryRegion(cl_mem_flags flags, size_t size, size_t alignment);
    /** Destructor */
    virtual ~ICLSVMMemoryRegion();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ICLSVMMemoryRegion(const ICLSVMMemoryRegion &) = delete;
    /** Default move constructor */
    ICLSVMMemoryRegion(ICLSVMMemoryRegion &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ICLSVMMemoryRegion &operator=(const ICLSVMMemoryRegion &) = delete;
    /** Default move assignment operator */
    ICLSVMMemoryRegion &operator=(ICLSVMMemoryRegion &&) = default;

    // Inherited methods overridden :
    void *ptr() override;

protected:
    void *_ptr;
};

/** OpenCL coarse-grain SVM memory region implementation */
class CLCoarseSVMMemoryRegion final : public ICLSVMMemoryRegion
{
public:
    /** Constructor
     *
     * @param[in] flags     Memory flags
     * @param[in] size      Region size
     * @param[in] alignment Alignment
     */
    CLCoarseSVMMemoryRegion(cl_mem_flags flags, size_t size, size_t alignment);

    // Inherited methods overridden :
    void *map(cl::CommandQueue &q, bool blocking) final;
    void unmap(cl::CommandQueue &q) final;
};

/** OpenCL fine-grain SVM memory region implementation */
class CLFineSVMMemoryRegion final : public ICLSVMMemoryRegion
{
public:
    /** Constructor
     *
     * @param[in] flags     Memory flags
     * @param[in] size      Region size
     * @param[in] alignment Alignment
     */
    CLFineSVMMemoryRegion(cl_mem_flags flags, size_t size, size_t alignment);

    // Inherited methods overridden :
    void *map(cl::CommandQueue &q, bool blocking) final;
    void unmap(cl::CommandQueue &q) final;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_RUNTIME_CL_CL_MEMORY_REGION_H */
