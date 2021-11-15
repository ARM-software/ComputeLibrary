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
#ifndef ARM_COMPUTE_CL_SUB_H
#define ARM_COMPUTE_CL_SUB_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{
/** Basic function to run arithmetic subtraction
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
 * @note The function performs an arithmetic subtraction between two tensors.
 */
class ClSub : public IClOperator
{
public:
    /** Configure function for a given list of arguments.
     *
     * Valid configurations (src1,src2) -> dst :
     *
     *   - (U8,U8)           -> U8
     *   - (U8,U8)           -> S16
     *   - (S16,U8)          -> S16
     *   - (U8,S16)          -> S16
     *   - (S16,S16)         -> S16
     *   - (S32,S32)         -> S32
     *   - (F16,F16)         -> F16
     *   - (F32,F32)         -> F32
     *   - (QASYMM8,QASYMM8) -> QASYMM8
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (QSYMM16,QSYMM16) -> QSYMM16
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] src1            First source tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     *                                 The source tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] src2            Second source tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     *                                 The source tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     dst             Destination tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in]      policy          Policy to use to handle overflow.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, ConvertPolicy policy,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClSub::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, ConvertPolicy policy,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_SUB_H */
