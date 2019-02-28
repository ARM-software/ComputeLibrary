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
#ifndef __ARM_COMPUTE_NEARGMINMAXLAYER_H__
#define __ARM_COMPUTE_NEARGMINMAXLAYER_H__

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEReductionOperationKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/INESimpleFunction.h"

namespace arm_compute
{
class IsTensor;

/** Function to calculate the index of the minimum or maximum values in a tensor based on an axis.
 *  This function calls the following NEON kernels:
 *
 * -# @ref NEReductionOperationKernel
 * -# @ref NEFillBorderKernel
 *
 */
class NEArgMinMaxLayer : public IFunction
{
public:
    /** Constructor */
    NEArgMinMaxLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input  Input source tensor. Data types supported: F16/F32.
     * @param[in]  axis   Axis to find max/min index.
     * @param[out] output Output source tensor. Data types supported: U32.
     * @param[in]  op     Operation to perform: min or max
     */
    void configure(ITensor *input, int axis, ITensor *output, const ReductionOperation &op);
    /** Static function to check if given info will lead to a valid configuration of @ref NEArgMinMaxLayer
     *
     * @param[in] input  Input source tensor info. Data types supported: F16/F32.
     * @param[in] axis   Axis to find max/min index.
     * @param[in] output Output source tensor info. Data types supported: U32.
     * @param[in] op     Operation to perform: min or max
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, int axis, const ITensorInfo *output, const ReductionOperation &op);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                _memory_group;
    NEReductionOperationKernel _reduction_kernel;
    NEFillBorderKernel         _fill_border_kernel;
    bool                       _run_fill_border;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEARGMINMAXLAYER_H__ */
