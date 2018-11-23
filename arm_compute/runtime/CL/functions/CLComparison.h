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
#ifndef __ARM_COMPUTE_CLCOMPARISON_H__
#define __ARM_COMPUTE_CLCOMPARISON_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Basic function to run @ref CLComparisonKernel */
class CLComparison : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs and outputs.
     *
     * @param[in]  input1    Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     *                       The input1 tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in]  input2    Source tensor. Data types supported: Same as @p input1.
     *                       The input2 tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out] output    Destination tensor. Data types supported: U8.
     * @param[out] operation Comparison operation to be used.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, ComparisonOperation operation);
    /** Static function to check if given info will lead to a valid configuration of @ref CLComparison
     *
     * @param[in]  input1    Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in]  input2    Source tensor. Data types supported: Same as @p input1.
     * @param[in]  output    Destination tensor. Data types supported: U8.
     * @param[out] operation Comparison operation to be used.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ComparisonOperation operation);
};

/** Basic function to run @ref CLComparisonKernel */
template <ComparisonOperation COP>
class CLComparisonStatic : public ICLSimpleFunction
{
public:
    static constexpr ComparisonOperation operation = COP; /** Comparison operations used by the class */

public:
    /** Initialise the kernel's inputs and outputs.
     *
     * @param[in]  input1 Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     *                    The input1 tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in]  input2 Source tensor. Data types supported: Same as @p input1.
     *                    The input2 tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out] output Destination tensor. Data types supported: U8.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLComparison
     *
     * @param[in] input1 Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in] input2 Source tensor. Data types supported: Same as @p input1.
     * @param[in] output Destination tensor. Data types supported: U8.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run equal comparison. */
using CLEqual = CLComparisonStatic<ComparisonOperation::Equal>;
/** Basic function to run not equal comparison. */
using CLNotEqual = CLComparisonStatic<ComparisonOperation::NotEqual>;
/** Basic function to run greater comparison. */
using CLGreater = CLComparisonStatic<ComparisonOperation::Greater>;
/** Basic function to run greater-equal comparison. */
using CLGreaterEqual = CLComparisonStatic<ComparisonOperation::GreaterEqual>;
/** Basic function to run less comparison. */
using CLLess = CLComparisonStatic<ComparisonOperation::Less>;
/** Basic function to run less-equal comparison. */
using CLLessEqual = CLComparisonStatic<ComparisonOperation::LessEqual>;
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLCOMPARISON_H__ */
