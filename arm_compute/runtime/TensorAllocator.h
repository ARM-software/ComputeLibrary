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

#include <cstdint>
#include <memory>
#include <vector>

namespace arm_compute
{
class Coordinates;
class TensorInfo;

/** Basic implementation of a CPU memory tensor allocator. */
class TensorAllocator : public ITensorAllocator
{
public:
    /** Default constructor. */
    TensorAllocator();

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

protected:
    /** No-op for CPU memory
     *
     * @return A pointer to the beginning of the tensor's allocation.
     */
    uint8_t *lock() override;

    /** No-op for CPU memory. */
    void unlock() override;

private:
    std::shared_ptr<std::vector<uint8_t>> _buffer; /**< CPU memory allocation. */
};
}
#endif /* __ARM_COMPUTE_TENSORALLOCATOR_H__ */
