/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_MUL_H
#define ARM_COMPUTE_CPU_MUL_H

#include "arm_compute/core/ITensorInfo.h"
#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref kernels::CpuMulKernel */
class CpuMul : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs, dst and convertion policy.
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *
     * @param[in, out] src1            First input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     *                                 This input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] src2            Second input tensor info. Data types supported: U8, QASYMM8 (only if @p src1 is QASYMM8), QASYMM8_SIGNED (only if @p src1 is QASYMM8_SIGNED), S16, S32, QSYMM16 (only if @p src1 is QSYMM16), F16 (only if @p src1 is F16), F32 (only if @p src1 is F32).
     *                                 This input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     dst             dst tensor info. Data types supported:
     *                                 - U8, only if both inputs are U8.
     *                                 - QASYMM8, only if both inputs are QASYMM8.
     *                                 - QASYMM8_SIGNED, only if @p src1 is QASYMM8_SIGNED.
     *                                 - S16.
     *                                 - QSYMM16, only if both inputs are QSYMM16.
     *                                 - S32, only if both inputs are S32 or both are QSYMM16.
     *                                 - F16, only if @p src1 is F16.
     *                                 - F32, only if both inputs are F32.
     * @param[in]      scale           Scale to apply after multiplication.
     *                                 Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     *                                 If both @p src1, @p src2 and @p dst are of datatype S32, scale cannot be 1/255
     * @param[in]      overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the inputs is of quantized datatype
     * @param[in]      rounding_policy Rounding policy.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuMul::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};

/** Basic function to run @ref kernels::CpuComplexMulKernel */
class CpuComplexMul : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs, dst.
     *
     * @param[in, out] src1     First input tensor. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] src2     Second input tensor. Data types supported: same as @p src1. Number of channels supported: same as @p src1.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     dst      The dst tensor. Data types supported: same as @p src1. Number of channels: same as @p src1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuComplexMul::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_MUL_H */