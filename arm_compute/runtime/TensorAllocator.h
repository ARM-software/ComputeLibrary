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
#ifndef __ARM_COMPUTE_TENSORALLOCATOR_H__
#define __ARM_COMPUTE_TENSORALLOCATOR_H__

#include "arm_compute/runtime/ITensorAllocator.h"
#include "arm_compute/runtime/Memory.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace arm_compute
{
class Coordinates;
class TensorInfo;
class Tensor;
template <typename>
class MemoryGroupBase;
using MemoryGroup = MemoryGroupBase<Tensor>;

/** Basic implementation of a CPU memory tensor allocator. */
class TensorAllocator : public ITensorAllocator
{
public:
    /** Default constructor. */
    TensorAllocator(Tensor *owner = nullptr);
    /** Default destructor */
    ~TensorAllocator();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    TensorAllocator(const TensorAllocator &) = delete;
    /** Prevent instances of this class from being copy assigned (As this class contains pointers). */
    TensorAllocator &operator=(const TensorAllocator &) = delete;
    /** Allow instances of this class to be moved */
    TensorAllocator(TensorAllocator &&) noexcept;
    /** Allow instances of this class to be moved */
    TensorAllocator &operator=(TensorAllocator &&) noexcept;

    /** Make ITensorAllocator's init methods available */
    using ITensorAllocator::init;

    /** Shares the same backing memory with another tensor allocator, while the tensor info might be different.
     *  In other words this can be used to create a sub-tensor from another tensor while sharing the same memory.
     *
     * @note TensorAllocator have to be of the same specialized type.
     *
     * @param[in] allocator The allocator that owns the backing memory to be shared. Ownership becomes shared afterwards.
     * @param[in] coords    The starting coordinates of the new tensor inside the parent tensor.
     * @param[in] sub_info  The new tensor information (e.g. shape etc)
     */
    void init(const TensorAllocator &allocator, const Coordinates &coords, TensorInfo sub_info);

    /** Returns the pointer to the allocated data. */
    uint8_t *data() const;

    /** Allocate size specified by TensorInfo of CPU memory.
     *
     * @note The tensor must not already be allocated when calling this function.
     *
     */
    void allocate() override;

    /** Free allocated CPU memory.
     *
     * @note The tensor must have been allocated when calling this function.
     *
     */
    void free() override;
    /** Import an existing memory as a tensor's backing memory
     *
     * @warning If the tensor is flagged to be managed by a memory manager,
     *          this call will lead to an error.
     * @warning Ownership of memory depends on the way the @ref Memory object was constructed
     * @note    Calling free on a tensor with imported memory will just clear
     *          the internal pointer value.
     *
     * @param[in] memory Memory to import
     *
     * @return error status
     */
    arm_compute::Status import_memory(Memory memory);
    /** Associates the tensor with a memory group
     *
     * @param[in] associated_memory_group Memory group to associate the tensor with
     */
    void set_associated_memory_group(MemoryGroup *associated_memory_group);

protected:
    /** No-op for CPU memory
     *
     * @return A pointer to the beginning of the tensor's allocation.
     */
    uint8_t *lock() override;

    /** No-op for CPU memory. */
    void unlock() override;

private:
    MemoryGroup *_associated_memory_group; /**< Registered memory manager */
    Memory       _memory;                  /**< CPU memory */
    Tensor      *_owner;                   /**< Owner of the allocator */
};
}
#endif /* __ARM_COMPUTE_TENSORALLOCATOR_H__ */
