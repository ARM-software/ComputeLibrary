/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_ITENSORALLOCATOR_H__
#define __ARM_COMPUTE_ITENSORALLOCATOR_H__

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
/** Interface to allocate tensors */
class ITensorAllocator
{
public:
    /** Default constructor. */
    ITensorAllocator();
    /** Allow instances of this class to be copy constructed */
    ITensorAllocator(const ITensorAllocator &) = default;
    /** Allow instances of this class to be copied */
    ITensorAllocator &operator=(const ITensorAllocator &) = default;
    /** Allow instances of this class to be move constructed */
    ITensorAllocator(ITensorAllocator &&) = default;
    /** Allow instances of this class to be moved */
    ITensorAllocator &operator=(ITensorAllocator &&) = default;
    /** Default virtual destructor. */
    virtual ~ITensorAllocator() = default;

    /** Initialize a tensor based on the passed @ref TensorInfo.
     *
     * @param[in] input     TensorInfo object containing the description of the tensor to initialize.
     * @param[in] alignment Alignment in bytes that the underlying base pointer should comply with.
     */
    void init(const TensorInfo &input, size_t alignment = 0);
    /** Return a reference to the tensor's metadata
     *
     * @return Reference to the tensor's metadata.
     */
    TensorInfo &info();
    /** Return a constant reference to the tensor's metadata
     *
     * @return Constant reference to the tensor's metadata.
     */
    const TensorInfo &info() const;
    /** Return underlying's tensor buffer alignment
     *
     * @return Tensor buffer alignment
     */
    size_t alignment() const;

    /** Interface to be implemented by the child class to allocate the tensor.
     *
     * @note The child is expected to use the TensorInfo to get the size of the memory allocation.
     * @warning The tensor must not already be allocated. Otherwise calling the function will fail.
     */
    virtual void allocate() = 0;

    /** Interface to be implemented by the child class to free the allocated tensor.
     *
     * @warning The tensor must have been allocated previously. Otherwise calling the function will fail.
     */
    virtual void free() = 0;

protected:
    /** Interface to be implemented by the child class to lock the memory allocation for the CPU to access.
     *
     * @return Pointer to a CPU mapping of the memory
     */
    virtual uint8_t *lock() = 0;
    /** Interface to be implemented by the child class to unlock the memory allocation after the CPU is done accessing it. */
    virtual void unlock() = 0;

private:
    TensorInfo _info;      /**< Tensor's metadata. */
    size_t     _alignment; /**< Tensor's alignment in bytes */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_ITENSORALLOCATOR_H__ */
