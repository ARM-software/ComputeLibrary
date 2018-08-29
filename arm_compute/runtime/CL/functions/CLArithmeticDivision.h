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
#ifndef __ARM_COMPUTE_CLARITHMETICDIVISION_H__
#define __ARM_COMPUTE_CLARITHMETICDIVISION_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLArithmeticDivisionKernel
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs an arithmetic division between two tensors.
 */
class CLArithmeticDivision : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output.
     *
     * @param[in, out] input1 First tensor input. Data types supported: F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2 Second tensor input. Same as @p input1.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output Output tensor. Data types supported: Same as @p input1.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticDivision
     *
     * @param[in] input1 First tensor input info. Data types supported: F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};
}
#endif /* __ARM_COMPUTE_CLARITHMETICDIVISION_H__ */
