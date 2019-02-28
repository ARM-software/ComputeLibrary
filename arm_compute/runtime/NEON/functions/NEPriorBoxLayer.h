/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEPRIORBOXLAYER_H__
#define __ARM_COMPUTE_NEPRIORBOXLAYER_H__

#include "arm_compute/core/NEON/kernels/NEPriorBoxLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEPriorBoxLayerKernel. */
class NEPriorBoxLayer : public INESimpleFunctionNoBorder
{
public:
    /** Set the input and output tensors.
     *
     * @param[in]  input1 First source tensor. Data types supported: F32. Data layouts supported: NCHW/NHWC.
     * @param[in]  input2 Second source tensor. Data types and layouts supported: same as @p input1
     * @param[out] output Destination tensor. Output dimensions are [W * H * num_priors * 4, 2]. Data type supported: same as @p input
     * @param[in]  info   Prior box layer info.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output, const PriorBoxLayerInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPriorBoxLayer
     *
     * @param[in] input1 First source tensor info. Data types supported: F32. Data layouts supported: NCHW/NHWC.
     * @param[in] input2 Second source tensor info. Data types and layouts supported: same as @p input1
     * @param[in] output Destination tensor info. Output dimensions are [W * H * num_priors * 4, 2]. Data type supported: same as @p input
     * @param[in] info   Prior box layer info.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const PriorBoxLayerInfo &info);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEPRIORBOXLAYER_H__ */
