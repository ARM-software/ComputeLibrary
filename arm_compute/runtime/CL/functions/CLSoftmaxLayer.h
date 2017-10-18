/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLSOFTMAXLAYER_H__
#define __ARM_COMPUTE_CLSOFTMAXLAYER_H__

#include "arm_compute/core/CL/kernels/CLSoftmaxLayerKernel.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Basic function to compute a SoftmaxLayer.
 *
 * Softmax is calculated by :
 * @f[ out = exp(x - max(x)) / sum(exp(x - max(x))) @f]
 *
 * This function runs the following kernels:
 * -# @ref CLLogits1DMaxKernel
 * -# @ref CLLogits1DShiftExpSumKernel
 * -# @ref CLLogits1DNormKernel
 */
class CLSoftmaxLayer : public IFunction
{
public:
    /** Constructor */
    CLSoftmaxLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QS8/QS16/F16/F32
     * @param[in]  beta   A scaling factor for the exponent. QS8/QS16/F16 only support a beta value of 1.
     * @param[out] output Destination tensor. Data types supported: same as @p input
     */
    void configure(const ICLTensor *input, ICLTensor *output, float beta = 1.0f);

    // Inherited methods overridden:
    void run() override;

private:
    CLMemoryGroup               _memory_group;
    CLLogits1DMaxKernel         _max_kernel;
    CLLogits1DShiftExpSumKernel _shift_exp_sum_kernel;
    CLLogits1DNormKernel        _norm_kernel;
    CLTensor                    _max;
    CLTensor                    _sum;
    CLTensor                    _tmp;
};
}
#endif /* __ARM_COMPUTE_CLSOFTMAXLAYER_H__ */
