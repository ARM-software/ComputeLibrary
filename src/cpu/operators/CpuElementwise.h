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

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
class CpuElementwiseBase : public ICpuOperator
{
public:
    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
/** Class to run @ref cpu::kernels::CpuArithmeticKernel except for division and power
 *
 * @note Max/Min/Squared difference supports input data type of QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32
 * @note PRelu supports inpute data type of QASYMM8/QASYMM8_SIGNED/F16/F32.
 */
template <ArithmeticOperation op>
class CpuElementwiseArithmetic : public CpuElementwiseBase
{
public:
    /** Configure the operator
     *
     * @param[in]  src0 The first source tensor information.
     * @param[in]  src1 The second source tensor information. With PRelu, this is used as alpha tensor.
     * @param[out] dst  The output tensor information.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuElementwiseArithmetic::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);
};

/** Class to run @ref cpu::kernels::CpuArithmeticKernel except for maximum operation */
using CpuElementwiseMax = CpuElementwiseArithmetic<ArithmeticOperation::MAX>;
/** Class to run @ref cpu::kernels::CpuArithmeticKernel except for minimum operation */
using CpuElementwiseMin = CpuElementwiseArithmetic<ArithmeticOperation::MIN>;
/** Class to run @ref cpu::kernels::CpuArithmeticKernel except for squared difference operation */
using CpuElementwiseSquaredDiff = CpuElementwiseArithmetic<ArithmeticOperation::SQUARED_DIFF>;

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for division
 *
 * @note The tensor data type for the inputs must be S32/F16/F32.
 * @note The function performs a division operation between two tensors (i.e., out[i] = in1[i] / in2[i])
 */
class CpuElementwiseDivision : public CpuElementwiseBase
{
public:
    /** Initialise the kernel's inputs, dst and conversion policy.
     *
     * @param[in, out] src0 First tensor input info. Data types supported: S32/F16/F32.
     * @param[in, out] src1 Second tensor input info. Data types supported: Same as @p src0.
     * @param[out]     dst  Output tensor info. Data types supported: Same as @p src0.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuElementwiseDivision::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);
};

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for power
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs a elementwise power of in1 to in2 (i.e., out[i] = in1[i] ^ in2[i])
 * @note For an exponent that is a float, this function will only work with a positive base.
 */
class CpuElementwisePower : public CpuElementwiseBase
{
public:
    /** Initialise the kernel's inputs, dst and conversion policy.
     *
     * @param[in, out] src0 First tensor input info. Data types supported: F16/F32.
     * @param[in, out] src1 Second tensor input info. Data types supported: Same as @p src0.
     * @param[out]     dst  Output tensor info. Data types supported: Same as @p src0.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuElementwisePower::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);
};

/** Basic function to run @ref cpu::kernels::CpuComparisonKernel.
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
class CpuElementwiseComparison : public CpuElementwiseBase
{
public:
    /** Initialise the kernel's inputs, dst and conversion policy.
     *
     * @param[in, out] src0 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] src1 Second tensor input info. Data types supported: Same as @p src0.
     * @param[out]     dst  Output tensor info. Data types supported: U16/U32.
     * @param[in]      op   Comparison Operation to be performed.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst, ComparisonOperation op);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuElementwiseComparison::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, ComparisonOperation op);
};

/** Basic function to run @ref cpu::kernels::CpuComparisonKernel
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
template <ComparisonOperation op>
class CpuElementwiseComparisonStatic : public CpuElementwiseBase
{
public:
    /** Initialise the kernel's inputs, dst and conversion policy.
     *
     * @param[in, out] src0 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] src1 Second tensor input info. Data types supported: Same as @p src0.
     * @param[out]     dst  Output tensor info. Data types supported: U16/U32.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuElementwiseComparisonStatic::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);
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