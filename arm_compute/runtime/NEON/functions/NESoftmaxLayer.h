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
#ifndef __ARM_COMPUTE_NESOFTMAXLAYER_H__
#define __ARM_COMPUTE_NESOFTMAXLAYER_H__

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NESoftmaxLayerKernel.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
class ITensor;

/** Basic function to compute a SoftmaxLayer.
 *
 * Softmax is calculated by :
 * @f[ out = \frac{e^{x - max(x)}}{\sum{e^{x - max(x)}}} @f]
 *
 * This function runs the following kernels:
 * -# @ref NELogits1DMaxKernel
 * -# @ref NELogits1DShiftExpSumKernel
 * -# @ref NELogits1DNormKernel
 */
class NESoftmaxLayer : public IFunction
{
public:
    /** Constructor */
    NESoftmaxLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QS8/QS16/F16/F32.
     * @param[out] output Destination tensor. Data types supported: same as @p input.
     * @param[in]  beta   (Optional) A scaling factor for the exponent. QS8/QS16 only support a beta value of 1.
     */
    void configure(ITensor *input, ITensor *output, float beta = 1.0f);
    /** Static function to check if given info will lead to a valid configuration of @ref NESoftmaxLayer
     *
     * @param[in] input  Source tensor. Data types supported: QS8/QS16/F16/F32
     * @param[in] output Destination tensor. Data types supported: same as @p input
     * @param[in] beta   (Optional) A scaling factor for the exponent. QS8/QS16 only support a beta value of 1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, float beta = 1.0f);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                 _memory_group;
    NELogits1DMaxKernel         _max_kernel;
    NELogits1DShiftExpSumKernel _shift_exp_sum_kernel;
    NELogits1DNormKernel        _norm_kernel;
    NEFillBorderKernel          _fill_border_kernel;
    Tensor                      _max;
    Tensor                      _sum;
    Tensor                      _tmp;
};
}
#endif /* __ARM_COMPUTE_NESOFTMAXLAYER_H__ */
