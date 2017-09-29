/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLTENSORALLOCATOR_H__
#define __ARM_COMPUTE_CLTENSORALLOCATOR_H__

#include "arm_compute/runtime/ITensorAllocator.h"

#include "arm_compute/core/CL/OpenCL.h"

#include <cstdint>

namespace arm_compute
{
class CLTensor;
template <typename>
class MemoryGroupBase;
using CLMemoryGroup = MemoryGroupBase<CLTensor>;

/** Basic implementation of a CL memory tensor allocator. */
class CLTensorAllocator : public ITensorAllocator
{
public:
    /** Default constructor. */
    CLTensorAllocator(CLTensor *owner = nullptr);
    /** Default destructor */
    ~CLTensorAllocator();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLTensorAllocator(const CLTensorAllocator &) = delete;
    /** Prevent instances of this class from being copy assigned (As this class contains pointers). */
    CLTensorAllocator &operator=(const CLTensorAllocator &) = delete;
    /** Allow instances of this class to be moved */
    CLTensorAllocator(CLTensorAllocator &&) = default;
    /** Allow instances of this class to be moved */
    CLTensorAllocator &operator=(CLTensorAllocator &&) = default;

    /** Interface to be implemented by the child class to return the pointer to the mapped data. */
    uint8_t *data();
    /** Interface to be implemented by the child class to return the pointer to the CL data. */
    const cl::Buffer &cl_data() const;
    /** Enqueue a map operation of the allocated buffer on the given queue.
     *
     * @param[in,out] q        The CL command queue to use for the mapping operation.
     * @param[in]     blocking If true, then the mapping will be ready to use by the time
     *                         this method returns, else it is the caller's responsibility
     *                         to flush the queue and wait for the mapping operation to have completed before using the returned mapping pointer.
     *
     * @return The mapping address.
     */
    uint8_t *map(cl::CommandQueue &q, bool blocking);
    /** Enqueue an unmap operation of the allocated buffer on the given queue.
     *
     * @note This method simply enqueue the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     *
     * @param[in,out] q       The CL command queue to use for the mapping operation.
     * @param[in]     mapping The cpu mapping to unmap.
     */
    void unmap(cl::CommandQueue &q, uint8_t *mapping);

    /** Allocate size specified by TensorInfo of OpenCL memory.
     *
     * @note: The tensor must not already be allocated when calling this function.
     *
     */
    void allocate() override;

    /** Free allocated OpenCL memory.
     *
     * @note The tensor must have been allocated when calling this function.
     *
     */
    void free() override;
    /** Associates the tensor with a memory group
     *
     * @param[in] associated_memory_group Memory group to associate the tensor with
     */
    void set_associated_memory_group(CLMemoryGroup *associated_memory_group);

protected:
    /** Call map() on the OpenCL buffer.
     *
     * @return A pointer to the beginning of the tensor's allocation.
     */
    uint8_t *lock() override;
    /** Call unmap() on the OpenCL buffer. */
    void unlock() override;

private:
    CLMemoryGroup *_associated_memory_group; /**< Registered memory manager */
    cl::Buffer     _buffer;                  /**< OpenCL buffer containing the tensor data. */
    uint8_t       *_mapping;                 /**< Pointer to the CPU mapping of the OpenCL buffer. */
    CLTensor      *_owner;                   /**< Owner of the allocator */
};
}
#endif /* __ARM_COMPUTE_CLTENSORALLOCATOR_H__ */
