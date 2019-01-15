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
#ifndef __ARM_COMPUTE_NEELEMENTWISEOPERATIONS_H__
#define __ARM_COMPUTE_NEELEMENTWISEOPERATIONS_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunction.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEArithmeticOperationKernel for max
 *
 * @note The tensor data type for the inputs must be QASYMM8/S16/F16/S32/F32.
 * @note The function performs a max operation between two tensors.
 */
class NEElementwiseMax : public INESimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor. Data types supported: Same as @p input1.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticOperationKernel for max
     *
     * @param[in] input1 First tensor input info. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref NEArithmeticOperationKernel for min
 *
 * @note The tensor data type for the inputs must be QASYMM8/S16/F16/S32/F32.
 * @note The function performs a max operation between two tensors.
 */
class NEElementwiseMin : public INESimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor. Data types supported: Same as @p input1.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticOperationKernel for min
     *
     * @param[in] input1 First tensor input info. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref NEArithmeticOperationKernel for squared difference
 *
 * @note The tensor data type for the inputs must be QASYMM8/S16/F16/S32/F32.
 * @note The function performs a squared different operation between two tensors (i.e., out[i] = (in1[i] - in2[i])^2
 */
class NEElementwiseSquaredDiff : public INESimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor. Data types supported: Same as @p input1.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticOperationKernel for squared difference
     *
     * @param[in] input1 First tensor input info. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref NEArithmeticOperationKernel for division
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs a squared different operation between two tensors (i.e., out[i] = in1[i] / in2[i])
 */
class NEElementwiseDivision : public INESimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: F16/F32.
     * @param[in, out] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor. Data types supported: Same as @p input1.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticOperationKernel for division
     *
     * @param[in] input1 First tensor input info. Data types supported: F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref NEComparisonOperationKernel.
 *
 * @note The tensor data type for the inputs must be QASYMM8/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
class NEElementwiseComparison : public INESimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor. Data types supported: U16/U32.
     * @param[in]      op     Comparison Operation to be performed.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output, ComparisonOperation op);
    /** Static function to check if given info will lead to a valid configuration of @ref NEComparisonOperationKernel
     *
     * @param[in] input1 First tensor input info. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: U16/U32.
     * @param[in] op     Comparison Operation to be performed.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ComparisonOperation op);
};

/** Basic function to run @ref NEComparisonOperationKernel
 *
 * @note The tensor data type for the inputs must be QASYMM8/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
template <ComparisonOperation op>
class NEElementwiseComparisonStatic : public INESimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor. Data types supported: U16/U32.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEComparisonOperationKernel
     *
     * @param[in] input1 First tensor input info. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: U16/U32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run equal comparison. */
using NEEqual = NEElementwiseComparisonStatic<ComparisonOperation::Equal>;
/** Basic function to run not equal comparison. */
using NENotEqual = NEElementwiseComparisonStatic<ComparisonOperation::NotEqual>;
/** Basic function to run greater comparison. */
using NEGreater = NEElementwiseComparisonStatic<ComparisonOperation::Greater>;
/** Basic function to run greater-equal comparison. */
using NEGreaterEqual = NEElementwiseComparisonStatic<ComparisonOperation::GreaterEqual>;
/** Basic function to run less comparison. */
using NELess = NEElementwiseComparisonStatic<ComparisonOperation::Less>;
/** Basic function to run less-equal comparison. */
using NELessEqual = NEElementwiseComparisonStatic<ComparisonOperation::LessEqual>;
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEELEMENTWISEOPERATIONS_H__ */
