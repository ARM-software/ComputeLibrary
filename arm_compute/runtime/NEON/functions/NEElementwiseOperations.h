/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEELEMENTWISEOPERATIONS_H
#define ARM_COMPUTE_NEELEMENTWISEOPERATIONS_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for max
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a max operation between two tensors.
 */
class NEElementwiseMax : public IFunction
{
public:
    /** Default Constructor */
    NEElementwiseMax();
    /** Default Destructor */
    ~NEElementwiseMax();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseMax(const NEElementwiseMax &) = delete;
    /** Default move constructor */
    NEElementwiseMax(NEElementwiseMax &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseMax &operator=(const NEElementwiseMax &) = delete;
    /** Default move assignment operator */
    NEElementwiseMax &operator=(NEElementwiseMax &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |S32            |S32            |S32            |
     * |S16            |S16            |S16            |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] input2   Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuArithmeticKernel for max
     *
     * @param[in] input1   First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for min
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a min operation between two tensors.
 */
class NEElementwiseMin : public IFunction
{
public:
    /** Default Constructor */
    NEElementwiseMin();
    /** Default Destructor */
    ~NEElementwiseMin();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseMin(const NEElementwiseMin &) = delete;
    /** Default move constructor */
    NEElementwiseMin(NEElementwiseMin &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseMin &operator=(const NEElementwiseMin &) = delete;
    /** Default move assignment operator */
    NEElementwiseMin &operator=(NEElementwiseMin &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |S32            |S32            |S32            |
     * |S16            |S16            |S16            |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] input2   Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuArithmeticKernel for min
     *
     * @param[in] input1   First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for squared difference
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a squared different operation between two tensors (i.e., out[i] = (in1[i] - in2[i])^2
 */
class NEElementwiseSquaredDiff : public IFunction
{
public:
    /** Default Constructor */
    NEElementwiseSquaredDiff();
    /** Default Destructor */
    ~NEElementwiseSquaredDiff();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseSquaredDiff(const NEElementwiseSquaredDiff &) = delete;
    /** Default move constructor */
    NEElementwiseSquaredDiff(NEElementwiseSquaredDiff &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseSquaredDiff &operator=(const NEElementwiseSquaredDiff &) = delete;
    /** Default move assignment operator */
    NEElementwiseSquaredDiff &operator=(NEElementwiseSquaredDiff &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |S32            |S32            |S32            |
     * |S16            |S16            |S16            |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] input2   Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuArithmeticKernel for squared difference
     *
     * @param[in] input1   First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for division
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs a squared different operation between two tensors (i.e., out[i] = in1[i] / in2[i])
 */
class NEElementwiseDivision : public IFunction
{
public:
    /** Default Constructor */
    NEElementwiseDivision();
    /** Default Destructor */
    ~NEElementwiseDivision();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseDivision(const NEElementwiseDivision &) = delete;
    /** Default move constructor */
    NEElementwiseDivision(NEElementwiseDivision &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseDivision &operator=(const NEElementwiseDivision &) = delete;
    /** Default move assignment operator */
    NEElementwiseDivision &operator=(NEElementwiseDivision &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: F16/F32.
     * @param[in, out] input2   Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuArithmeticKernel for division
     *
     * @param[in] input1   First tensor input info. Data types supported: F16/F32.
     * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for power
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs a elementwise power of in1 to in2 (i.e., out[i] = in1[i] ^ in2[i])
 * @note For an exponent that is a float, this function will only work with a positive base.
 */
class NEElementwisePower : public IFunction
{
public:
    /** Default Constructor */
    NEElementwisePower();
    /** Default Destructor */
    ~NEElementwisePower();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwisePower(const NEElementwisePower &) = delete;
    /** Default move constructor */
    NEElementwisePower(NEElementwisePower &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwisePower &operator=(const NEElementwisePower &) = delete;
    /** Default move assignment operator */
    NEElementwisePower &operator=(NEElementwisePower &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: F16/F32.
     * @param[in, out] input2   Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuArithmeticKernel for power
     *
     * @param[in] input1   First tensor input info. Data types supported: F16/F32.
     * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref cpu::kernels::CpuComparisonKernel.
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
class NEElementwiseComparison : public IFunction
{
public:
    /** Default Constructor */
    NEElementwiseComparison();
    /** Default Destructor */
    ~NEElementwiseComparison();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseComparison(const NEElementwiseComparison &) = delete;
    /** Default move constructor */
    NEElementwiseComparison(NEElementwiseComparison &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseComparison &operator=(const NEElementwiseComparison &) = delete;
    /** Default move assignment operator */
    NEElementwiseComparison &operator=(NEElementwiseComparison &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst   |
     * |:--------------|:--------------|:-----|
     * |QASYMM8        |QASYMM8        |U8    |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |U8    |
     * |S32            |S32            |U8    |
     * |U8             |U8             |U8    |
     * |S16            |S16            |U8    |
     * |F16            |F16            |U8    |
     * |F32            |F32            |U8    |
     *
     * @param[in, out] input1 First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor. Data types supported: U8.
     * @param[in]      op     Comparison Operation to be performed.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output, ComparisonOperation op);
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuComparisonKernel
     *
     * @param[in] input1 First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: U8.
     * @param[in] op     Comparison Operation to be performed.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ComparisonOperation op);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref cpu::kernels::CpuComparisonKernel
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
template <ComparisonOperation op>
class NEElementwiseComparisonStatic : public IFunction
{
public:
    /** Default Constructor */
    NEElementwiseComparisonStatic();
    /** Default Destructor */
    ~NEElementwiseComparisonStatic();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseComparisonStatic(const NEElementwiseComparisonStatic &) = delete;
    /** Default move constructor */
    NEElementwiseComparisonStatic(NEElementwiseComparisonStatic &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseComparisonStatic &operator=(const NEElementwiseComparisonStatic &) = delete;
    /** Default move assignment operator */
    NEElementwiseComparisonStatic &operator=(NEElementwiseComparisonStatic &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in, out] input1 First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in, out] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[out]     output Output tensor. Data types supported: U16/U32.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref cpu::kernels::CpuComparisonKernel
     *
     * @param[in] input1 First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: U16/U32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
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
#endif /* ARM_COMPUTE_NEELEMENTWISEOPERATIONS_H */
