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

#ifndef __ARM_COMPUTE_GCABSOLUTEDIFFERENCE_H__
#define __ARM_COMPUTE_GCABSOLUTEDIFFERENCE_H__

#include "arm_compute/runtime/GLES_COMPUTE/IGCSimpleFunction.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref GCAbsoluteDifferenceKernel
 *
 * @note The tensor data types for the inputs must be U8.
 * @note The function calculates the absolute difference also when the 2 inputs have different tensor data types.
 */
class GCAbsoluteDifference : public IGCSimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in]  input1 First input tensor. Data types supported: U8
     * @param[in]  input2 Second input tensor. Data types supported: U8
     * @param[out] output Output tensor. Data types supported: U8
     */
    void configure(const IGCTensor *input1, const IGCTensor *input2, IGCTensor *output);
};
}

#endif /* __ARM_COMPUTE_GCABSOLUTEDIFFERENCE_H__ */
