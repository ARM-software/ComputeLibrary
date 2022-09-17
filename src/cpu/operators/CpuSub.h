/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_SUB_H
#define ARM_COMPUTE_CPU_SUB_H

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref kernels::CpuSubKernel */
class CpuSub : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs, dst and conversion policy.
     *
     * Valid configurations (src0,src1) -> dst :
     *
     *   - (U8,U8)                          -> U8
     *   - (QASYMM8, QASYMM8)               -> QASYMM8
     *   - (QASYMM8_SIGNED, QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (S16,S16)                        -> S16
     *   - (S32,S32)                        -> S32
     *   - (F16,F16)                        -> F16
     *   - (F32,F32)                        -> F32
     *
     * @param[in]  src0     First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[in]  src1     Second tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[out] dst      Output tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[in]  policy   Policy to use to handle overflow. Convert policy cannot be WRAP if datatype is quantized.
     * @param[in]  act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuSub::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SUB_H */