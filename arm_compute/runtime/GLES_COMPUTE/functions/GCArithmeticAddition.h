/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_GCARITHMETICADDITION_H__
#define __ARM_COMPUTE_GCARITHMETICADDITION_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/IGCSimpleFunction.h"

namespace arm_compute
{
class IGCTensor;

/** Basic function to run @ref GCArithmeticAdditionKernel
 *
 * @note The tensor data type for the inputs must be F16.
 * @note The function performs an arithmetic addition between two tensors.
 */
class GCArithmeticAddition : public IGCSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and convertion policy.
     *
     * @param[in]  input1 First tensor input. Data types supported: F16.
     * @param[in]  input2 Second tensor input. Data types supported: F16.
     * @param[out] output Output tensor. Data types supported: F16.
     * @param[in]  policy Policy to use to handle overflow.
     */
    void configure(const IGCTensor *input1, const IGCTensor *input2, IGCTensor *output, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref GCArithmeticAddition
     *
     * @param[in] input1 First tensor input info. Data types supported: F16.
     * @param[in] input2 Second tensor input info. Data types supported: F16.
     * @param[in] output Output tensor info. Data types supported: F16.
     * @param[in] policy Policy to use to handle overflow.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy);
};
}
#endif /* __ARM_COMPUTE_GCARITHMETICADDITION_H__ */
