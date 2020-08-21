/*
 * Copyright (c) 2017-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_SUBTENSOR_H
#define ARM_COMPUTE_SUBTENSOR_H

#include "arm_compute/core/SubTensorInfo.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>

namespace arm_compute
{
class ITensorInfo;

/** Basic implementation of the sub-tensor interface */
class SubTensor : public ITensor
{
public:
    /** Default Constructor */
    SubTensor();
    /** Constructor
     *
     * @param[in] parent        Parent tensor
     * @param[in] tensor_shape  Shape of the subtensor
     * @param[in] coords        Coordinates of the first subtensor element inside the parent tensor.
     * @param[in] extend_parent (Optional) Extend parent with subtensor shape if subtensor indexes out of bounds
     */
    SubTensor(ITensor *parent, const TensorShape &tensor_shape, const Coordinates &coords, bool extend_parent = false);
    /** Destructor: free the tensor's memory */
    ~SubTensor() = default;
    /** Restrict instances of this class to be copy constructed */
    SubTensor(const SubTensor &) = delete;
    /** Restrict instances of this class to be copied */
    SubTensor &operator=(const SubTensor &) = delete;
    /** Allow instances of this class to be move constructed */
    SubTensor(SubTensor &&) = default;
    /** Allow instances of this class to be moved */
    SubTensor &operator=(SubTensor &&) = default;
    /** Return the parent tensor of the subtensor
     *
     * @return Parent tensor
     */
    ITensor *parent();

    // Inherited methods overridden:
    ITensorInfo *info() const override;
    ITensorInfo *info() override;
    uint8_t     *buffer() const override;

private:
    ITensor              *_parent;
    mutable SubTensorInfo _info;
};
}
#endif /*ARM_COMPUTE_SUBTENSOR_H */
