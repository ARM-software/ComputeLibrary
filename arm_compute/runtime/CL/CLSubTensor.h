/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLSUBTENSOR_H__
#define __ARM_COMPUTE_CLSUBTENSOR_H__

#include "arm_compute/core/SubTensorInfo.h"
#include "arm_compute/runtime/CL/CLTensor.h"

#include <cstdint>

namespace arm_compute
{
class ITensorInfo;

/** Basic implementation of the OpenCL sub-tensor interface */
class CLSubTensor : public ICLTensor
{
public:
    /** Default Constructor */
    CLSubTensor();
    /** Constructor
     *
     * @param[in] parent        Parent tensor
     * @param[in] tensor_shape  Shape of the subtensor
     * @param[in] coords        Coordinates of the first subtensor element inside the parent tensor.
     * @param[in] extend_parent (Optional) Extend parent with subtensor shape if subtensor indexes out of bounds
     */
    CLSubTensor(ICLTensor *parent, const TensorShape &tensor_shape, const Coordinates &coords, bool extend_parent = false);
    /** Destructor: free the tensor's memory */
    ~CLSubTensor() = default;
    /** Restrict instances of this class to be copy constructed */
    CLSubTensor(const CLSubTensor &) = delete;
    /** Restrict instances of this class to be copied */
    CLSubTensor &operator=(const CLSubTensor &) = delete;
    /** Allow instances of this class to be move constructed */
    CLSubTensor(CLSubTensor &&) = default;
    /** Allow instances of this class to be moved */
    CLSubTensor &operator=(CLSubTensor &&) = default;

    /** Enqueue a map operation of the allocated buffer.
     *
     * @note Mapping a subtensor will lead to the mapping of the whole parent tensor for now.
     *
     * @param[in] blocking If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed.
     */
    void map(bool blocking = true);
    using ICLTensor::map;
    /** Enqueue an unmap operation of the allocated and mapped buffer.
     *
     * @note Unmapping a subtensor will lead to the unmapping of the whole parent tensor for now.
     *
     * @note This method simply enqueues the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     */
    void unmap();
    using ICLTensor::unmap;

    /** Return the parent tensor of the subtensor
     *
     * @return Parent tensor
     */
    ICLTensor *parent();

    // Inherited methods overridden:
    ITensorInfo      *info() const override;
    ITensorInfo      *info() override;
    const cl::Buffer &cl_buffer() const override;
    CLQuantization    quantization() const override;

protected:
    // Inherited methods overridden:
    uint8_t *do_map(cl::CommandQueue &q, bool blocking) override;
    void do_unmap(cl::CommandQueue &q) override;

private:
    ICLTensor            *_parent;
    mutable SubTensorInfo _info;
};
}
#endif /*__ARM_COMPUTE_CLSUBTENSOR_H__ */
