/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_OPERATORTENSOR_H
#define ARM_COMPUTE_OPERATORTENSOR_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/Types.h"
#include "arm_compute/runtime/experimental/Types.h"

#include <cstdint>

namespace arm_compute
{
class TensorInfo;
class IRuntimeContext;
class IMemory;
namespace experimental
{
/** Basic implementation of the tensor interface */
class OperatorTensor : public ITensor
{
public:
    /** Constructor
     *
     * @param[in] info   Pointer to the tensor info.
     * @param[in] memory Pointer to the memory info.
     *
     */
    OperatorTensor(ITensorInfo *info, IMemory *memory);
    /** Destructor: free the tensor's memory */
    ~OperatorTensor() = default;
    /** Allow instances of this class to be move constructed */
    OperatorTensor(OperatorTensor &&) = default;
    /** Allow instances of this class to be moved */
    OperatorTensor &operator=(OperatorTensor &&) = default;
    /** Prevent instances of this class to be copy assigned */
    OperatorTensor &operator=(const OperatorTensor &) = delete;
    /** Prevent instances of this class to be copy constructed */
    OperatorTensor(const OperatorTensor &) = delete;

    // Inherited methods overridden:
    arm_compute::ITensorInfo *info() const override;
    arm_compute::ITensorInfo *info() override;
    uint8_t                  *buffer() const override;

private:
    arm_compute::ITensorInfo *_info;
    arm_compute::IMemory     *_memory;
    MemoryType                _mem_type;
};
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_OPERATORTENSOR_H */
