/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_ELEMENTWISE_H
#define ARM_COMPUTE_CPU_ELEMENTWISE_H

#include "src/runtime/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for max
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a max operation between two tensors.
 */
class CpuElementwiseMax : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor info. Data types supported: Same as @p input1.
     */
    void configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuArithmeticKernel for max
     *
     * @param[in] input1 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for min
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a min operation between two tensors.
 */
class CpuElementwiseMin : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor info. Data types supported: Same as @p input1.
     */
    void configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuArithmeticKernel for min
     *
     * @param[in] input1 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for squared difference
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a squared different operation between two tensors (i.e., out[i] = (in1[i] - in2[i])^2
 */
class CpuElementwiseSquaredDiff : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor info. Data types supported: Same as @p input1.
     */
    void configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuArithmeticKernel for squared difference
     *
     * @param[in] input1 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for division
 *
 * @note The tensor data type for the inputs must be S32/F16/F32.
 * @note The function performs a division operation between two tensors (i.e., out[i] = in1[i] / in2[i])
 */
class CpuElementwiseDivision : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input info. Data types supported: S32/F16/F32.
     * @param[in, out] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor info. Data types supported: Same as @p input1.
     */
    void configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuArithmeticKernel for division
     *
     * @param[in] input1 First tensor input info. Data types supported: S32/F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for power
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs a elementwise power of in1 to in2 (i.e., out[i] = in1[i] ^ in2[i])
 * @note For an exponent that is a float, this function will only work with a positive base.
 */
class CpuElementwisePower : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input info. Data types supported: F16/F32.
     * @param[in, out] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor info. Data types supported: Same as @p input1.
     */
    void configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuArithmeticKernel for power
     *
     * @param[in] input1 First tensor input info. Data types supported: F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run @ref cpu::kernels::CpuComparisonKernel.
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
class CpuElementwiseComparison : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor info. Data types supported: U16/U32.
     * @param[in]      op     Comparison Operation to be performed.
     */
    void configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, ComparisonOperation op);
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuComparisonKernel
     *
     * @param[in] input1 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: U16/U32.
     * @param[in] op     Comparison Operation to be performed.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ComparisonOperation op);
};

/** Basic function to run @ref cpu::kernels::CpuComparisonKernel
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
template <ComparisonOperation op>
class CpuElementwiseComparisonStatic : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor info. Data types supported: U16/U32.
     */
    void configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuComparisonKernel
     *
     * @param[in] input1 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: U16/U32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};

/** Basic function to run equal comparison. */
using NEEqual = CpuElementwiseComparisonStatic<ComparisonOperation::Equal>;
/** Basic function to run not equal comparison. */
using NENotEqual = CpuElementwiseComparisonStatic<ComparisonOperation::NotEqual>;
/** Basic function to run greater comparison. */
using NEGreater = CpuElementwiseComparisonStatic<ComparisonOperation::Greater>;
/** Basic function to run greater-equal comparison. */
using NEGreaterEqual = CpuElementwiseComparisonStatic<ComparisonOperation::GreaterEqual>;
/** Basic function to run less comparison. */
using NELess = CpuElementwiseComparisonStatic<ComparisonOperation::Less>;
/** Basic function to run less-equal comparison. */
using NELessEqual = CpuElementwiseComparisonStatic<ComparisonOperation::LessEqual>;
} // namespace cpu
} // namespace arm_compute

#endif /* ARM_COMPUTE_CPU_ELEMENTWISE_H */