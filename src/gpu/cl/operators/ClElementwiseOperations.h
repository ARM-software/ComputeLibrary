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
#ifndef ARM_COMPUTE_CL_ELEMENTWISE_OPERATIONS_H
#define ARM_COMPUTE_CL_ELEMENTWISE_OPERATIONS_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{
/** Basic function to run @ref opencl::kernels::ClArithmeticKernel for division
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs an arithmetic division between two tensors.
 */
class ClElementwiseDivision : public IClOperator
{
public:
    /** Configure function for a given list of arguments.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src1            First source tensor info. Data types supported: F16/F32.
     * @param[in]  src2            Second source tensor info. same as @p src1.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src1.
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClElementwiseDivision::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
};

/** Basic function to run @ref opencl::kernels::ClArithmeticKernel for max
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
 * @note The function performs a max operation between two tensors.
 */
class ClElementwiseMax : public IClOperator
{
public:
    /** Configure function for a given list of arguments.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src1            First source tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/U32/F16/F32.
     * @param[in]  src2            Second source tensor info. Data types supported: same as @p src1.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src1.
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClElementwiseMax::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
};

/** Basic function to run @ref opencl::kernels::ClArithmeticKernel for min
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
 * @note The function performs a max operation between two tensors.
 */
class ClElementwiseMin : public IClOperator
{
public:
    /** Configure function for a given list of arguments.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src1            First source tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/U32/F16/F32.
     * @param[in]  src2            Second source tensor info. Data types supported: same as @p src1.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src1.
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClElementwiseMin::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
};

/** Basic function to run @ref opencl::kernels::ClArithmeticKernel for squared difference
 *
 * @note The tensor data type for the inputs must be QASYMM8/U8/S16/QSYMM16/F16/F32.
 * @note The function performs a squared different operation between two tensors (i.e., out[i] = (in1[i] - in2[i])^2
 */
class ClElementwiseSquaredDiff : public IClOperator
{
public:
    /** Configure function for a given list of arguments.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src1            First source tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[in]  src2            Second source tensor info. Data types supported: same as @p src1.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src1.
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClElementwiseSquaredDiff::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
};

/** Basic function to run @ref opencl::kernels::ClArithmeticKernel for power
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs an elementwise power of in1 to in2 (i.e., out[i] = in1[i] ^ in2[i])
 */
class ClElementwisePower : public IClOperator
{
public:
    /** Configure function for a given list of arguments.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src1            First source tensor info. Data types supported: F16/F32.
     * @param[in]  src2            Second source tensor info. Data types supported: F16/F32.
     * @param[out] dst             Destination tensor info. Data types supported:F16/F32.
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClElementwisePower::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_ELEMENTWISE_OPERATIONS_H */
