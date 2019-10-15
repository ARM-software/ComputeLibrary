/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLARGMINMAXLAYER_H__
#define __ARM_COMPUTE_CLARGMINMAXLAYER_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
class ITensorInfo;
class ICLTensor;
class CLReductionOperation;

/** Function to calculate the index of the minimum or maximum values in a
 *  tensor based on an axis.
 *
 * @note The indices are computed in unsigned 32-bit (U32). It is the user's
 *       responsibility to check that the results do not overflow in case the
 *       output data type is set to signed 32-bit integer (S32).
 */
class CLArgMinMaxLayer : public IFunction
{
public:
    /** Default Constructor.
     *
     * @param[in] memory_manager (Optional) Memory manager.
     */
    CLArgMinMaxLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input  Input source tensor, this could be written if @ref CLReductionOperation
     *                    manipulates its border for better performance. Data types supported: F16/F32.
     * @param[in]  axis   Axis to find max/min index.
     * @param[out] output Output source tensor. Data types supported: U32/S32.
     * @param[in]  op     Operation to perform: min or max
     */
    void configure(ICLTensor *input, int axis, ICLTensor *output, const ReductionOperation &op);
    /** Static function to check if given info will lead to a valid configuration of @ref CLArgMinMaxLayer
     *
     * @param[in] input  Input source tensor info. Data types supported: F16/F32.
     * @param[in] axis   Axis to find max/min index.
     * @param[in] output Output source tensor info. Data types supported: U32/S32.
     * @param[in] op     Operation to perform: min or max
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, int axis, const ITensorInfo *output, const ReductionOperation &op);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<CLReductionOperation> _reduction_function;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLARGMINMAXLAYER_H__ */
