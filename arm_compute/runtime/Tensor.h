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
#ifndef ARM_COMPUTE_TENSOR_H
#define ARM_COMPUTE_TENSOR_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include <cstdint>

namespace arm_compute
{
class ITensorInfo;
class IRuntimeContext;
/** Basic implementation of the tensor interface */
class Tensor : public ITensor, public IMemoryManageable
{
public:
    /** Constructor
     *
     * @param[in] ctx (Optional) Pointer to the runtime context.
     *
     */
    Tensor(IRuntimeContext *ctx = nullptr);
    /** Destructor: free the tensor's memory */
    ~Tensor() = default;
    /** Allow instances of this class to be move constructed */
    Tensor(Tensor &&) = default;
    /** Allow instances of this class to be moved */
    Tensor &operator=(Tensor &&) = default;
    /** Return a pointer to the tensor's allocator
     *
     * @return A pointer to the tensor's allocator
     */
    TensorAllocator *allocator();

    // Inherited methods overridden:
    ITensorInfo *info() const override;
    ITensorInfo *info() override;
    uint8_t     *buffer() const override;
    void associate_memory_group(IMemoryGroup *memory_group) override;

private:
    mutable TensorAllocator _allocator; /**< Instance of the basic CPU allocator.*/
};

/** Image */
using Image = Tensor;
} // namespace arm_compute
#endif /*ARM_COMPUTE_TENSOR_H */
