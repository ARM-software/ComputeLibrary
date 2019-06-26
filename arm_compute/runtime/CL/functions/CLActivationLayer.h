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
#ifndef __ARM_COMPUTE_CLACTIVATIONLAYER_H__
#define __ARM_COMPUTE_CLACTIVATIONLAYER_H__

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLActivationLayerKernel
 *
 * @note The function simulates an activation layer with the specified activation function.
 */
class CLActivationLayer : public ICLSimpleFunction
{
public:
    /** Set the input and output tensor.
     *
     * @note If the output tensor is a nullptr or is equal to the input, the activation function will be performed in-place
     *
     * @param[in, out] input    Source tensor. In case of @p output tensor = nullptr, this tensor will store the result
     *                          of the activation function. Data types supported: QASYMM8/QSYMM16/F16/F32.
     * @param[out]     output   Destination tensor. Data type supported: same as @p input
     * @param[in]      act_info Activation layer parameters.
     */
    void configure(ICLTensor *input, ICLTensor *output, ActivationLayerInfo act_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLActivationLayer
     *
     * @param[in] input    Source tensor info. In case of @p output tensor info = nullptr, this tensor will store the result
     *                     of the activation function. Data types supported: QASYMM8/QSYMM16/F16/F32.
     * @param[in] output   Destination tensor info. Data type supported: same as @p input
     * @param[in] act_info Activation layer information.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info);
};
}
#endif /* __ARM_COMPUTE_CLACTIVATIONLAYER_H__ */
