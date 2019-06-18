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
#ifndef __ARM_COMPUTE_CLELEMENTWISEOPERATIONS_H__
#define __ARM_COMPUTE_CLELEMENTWISEOPERATIONS_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLSaturatedArithmeticOperationKernel for addition
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
 * @note The function performs an arithmetic addition between two tensors.
 */
class CLArithmeticAddition : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2 Second tensor input. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), QSYMM16 (only if @p input1 is QSYMM16), S16/F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output Output tensor. Data types supported: U8 (Only if both inputs are U8), QASYMM8 (only if both inputs are QASYMM8), QSYMM16 (only if both inputs is QSYMM16), S16/F16/F32.
     * @param[in]      policy Policy to use to handle overflow.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSaturatedArithmeticOperationKernel for addition
     *
     * @param[in] input1 First tensor input info. Data types supported: U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), QSYMM16 (only if @p input1 is QSYMM16), S16/F16/F32.
     * @param[in] output Output tensor info. Data types supported: U8 (Only if both inputs are U8), QASYMM8 (only if both inputs are QASYMM8), QSYMM16 (only if both inputs is QSYMM16), S16/F16/F32.
     * @param[in] policy Policy to use to handle overflow.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy);
};

/** Basic function to run @ref CLSaturatedArithmeticOperationKernel for subtraction
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/S32/U32/F16/F32.
 * @note The function performs an arithmetic subtraction between two tensors.
 */
class CLArithmeticSubtraction : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: U8/QASYMM8/S16/S32/U32/F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2 Second tensor input. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), S16/F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output Output tensor. Data types supported: U8 (Only if both inputs are U8), QASYMM8 (only if both inputs are QASYMM8), S16/F16/F32.
     * @param[in]      policy Policy to use to handle overflow.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSaturatedArithmeticOperationKernel for subtraction
     *
     * @param[in] input1 First tensor input info. Data types supported: U8/QASYMM8/S16/S32/U32/F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), S16/F16/F32.
     * @param[in] output Output tensor info. Data types supported: U8 (Only if both inputs are U8), QASYMM8 ( only if both inputs are QASYMM8), S16/F16/F32.
     * @param[in] policy Policy to use to handle overflow.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy);
};

/** Basic function to run @ref CLSaturatedArithmeticOperationKernel for division
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

/** Basic function to run @ref CLArithmeticOperationKernel for max
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
 * @note The function performs a max operation between two tensors.
 */
class CLElementwiseMax : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2 Second tensor input. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), S16, QSYMM16 (only if @p input1 is QSYMM16), F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output Output tensor. Data types supported: U8 (Only if both inputs are U8), QASYMM8 (only if both inputs are QASYMM8), S16, QSYMM16 (only if both inputs are QSYMM16), F16/F32.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticOperationKernel for max
     *
     * @param[in] input1 First tensor input info. Data types supported: U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), S16, QSYMM16 (only if @p input1 is QSYMM16), F16/F32.
     * @param[in] output Output tensor info. Data types supported: U8 (Only if both inputs are U8), QASYMM8 ( only if both inputs are QASYMM8), S16, QSYMM16 (only if both inputs are QSYMM16), F16/F32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref CLArithmeticOperationKernel for min
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
 * @note The function performs a max operation between two tensors.
 */
class CLElementwiseMin : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2 Second tensor input. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), S16, QSYMM16 (only if @p input1 is QSYMM16), F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output Output tensor. Data types supported: U8 (Only if both inputs are U8), QASYMM8 (only if both inputs are QASYMM8), S16, QSYMM16 (only if both inputs are QSYMM16), F16/F32.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticOperationKernel for min
     *
     * @param[in] input1 First tensor input info. Data types supported: U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), S16, QSYMM16 (only if @p input1 is QSYMM16), F16/F32.
     * @param[in] output Output tensor info. Data types supported: U8 (Only if both inputs are U8), QASYMM8 ( only if both inputs are QASYMM8), S16, QSYMM16 (only if both inputs are QSYMM16), F16/F32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref CLArithmeticOperationKernel for squared difference
 *
 * @note The tensor data type for the inputs must be QASYMM8/U8/S16/QSYMM16/F16/F32.
 * @note The function performs a squared different operation between two tensors (i.e., out[i] = (in1[i] - in2[i])^2
 */
class CLElementwiseSquaredDiff : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: U8/QASYMM8/S16/QSYMM16/F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2 Second tensor input. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), S16, QSYMM16 (only if @p input1 is QSYMM16), F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output Output tensor. Data types supported: U8 (Only if both inputs are U8), QASYMM8 (only if both inputs are QASYMM8), S16, QSYMM16 (only if both inputs are QSYMM16), F16/F32.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticOperationKernel for squared difference
     *
     * @param[in] input1 First tensor input info. Data types supported: U8/QASYMM8/S16/QSYMM16/F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), S16, QSYMM16 (only if @p input1 is QSYMM16), F16/F32.
     * @param[in] output Output tensor info. Data types supported: U8 (Only if both inputs are U8), QASYMM8 ( only if both inputs are QASYMM8), S16, QSYMM16 (only if both inputs are QSYMM16), F16/F32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref CLArithmeticOperationKernel for power
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs an elementwise power of in1 to in2 (i.e., out[i] = in1[i] ^ in2[i])
 */
class CLElementwisePower : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2 Second tensor input. Data types supported: F16/F32.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output Output tensor. Data types supported:F16/F32.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticOperationKernel for power
     *
     * @param[in] input1 First tensor input info. Data types supported: F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: F16/F32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLELEMENTWISEOPERATIONS_H__ */
