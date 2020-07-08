/*
 * Copyright (c) 2016-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_TENSORALLOCATOR_H
#define ARM_COMPUTE_TENSORALLOCATOR_H
#include "arm_compute/runtime/ITensorAllocator.h"

#include "arm_compute/runtime/Memory.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace arm_compute
{
// Forward declaration
class Coordinates;
class TensorInfo;

/** Basic implementation of a CPU memory tensor allocator. */
class TensorAllocator : public ITensorAllocator
{
public:
    /** Default constructor.
     *
     * @param[in] owner Memory manageable owner
     */
    TensorAllocator(IMemoryManageable *owner);
    /** Default destructor */
    ~TensorAllocator();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    TensorAllocator(const TensorAllocator &) = delete;
    /** Prevent instances of this class from being copy assigned (As this class contains pointers) */
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
    void init(const TensorAllocator &allocator, const Coordinates &coords, TensorInfo &sub_info);

    /** Returns the pointer to the allocated data.
     *
     * @return a pointer to the allocated data.
     */
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
     * @warning size is expected to be compliant with total_size reported by ITensorInfo.
     * @warning ownership of memory is not transferred.
     * @warning tensor shouldn't be memory managed.
     * @warning padding should be accounted by the client code.
     * @warning memory must be writable in case of in-place operations
     * @note buffer alignment will be checked to be compliant with alignment reported by ITensorInfo.
     *
     * @param[in] memory Raw memory pointer to be used as backing memory
     *
     * @return An error status
     */
    Status import_memory(void *memory);
    /** Associates the tensor with a memory group
     *
     * @param[in] associated_memory_group Memory group to associate the tensor with
     */
    void set_associated_memory_group(IMemoryGroup *associated_memory_group);

protected:
    /** No-op for CPU memory
     *
     * @return A pointer to the beginning of the tensor's allocation.
     */
    uint8_t *lock() override;

    /** No-op for CPU memory. */
    void unlock() override;

private:
    IMemoryManageable *_owner;                   /**< Memory manageable object that owns the allocator */
    IMemoryGroup      *_associated_memory_group; /**< Registered memory manager */
    Memory             _memory;                  /**< CPU memory */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_TENSORALLOCATOR_H */
