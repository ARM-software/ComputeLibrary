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
#ifndef ARM_COMPUTE_NEARITHMETICADDITION_H
#define ARM_COMPUTE_NEARITHMETICADDITION_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to run @ref cpu::kernels::CpuAddKernel */
class NEArithmeticAddition : public IFunction
{
public:
    /** Default Constructor */
    NEArithmeticAddition();
    /** Default Destructor */
    ~NEArithmeticAddition();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEArithmeticAddition(const NEArithmeticAddition &) = delete;
    /** Default move constructor */
    NEArithmeticAddition(NEArithmeticAddition &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEArithmeticAddition &operator=(const NEArithmeticAddition &) = delete;
    /** Default move assignment operator */
    NEArithmeticAddition &operator=(NEArithmeticAddition &&);
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
     * |QSYMM16        |QSYMM16        |QASYMM16       |
     * |QSYMM16        |QSYMM16        |S32            |
     * |U8             |U8             |U8             |
     * |S16            |S16            |S16            |
     * |S32            |S32            |S32            |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in]  input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
     * @param[in]  input2   Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
     * @param[out] output   Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
     * @param[in]  policy   Policy to use to handle overflow.
     * @param[in]  act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticAddition
     *
     * @param[in] input1   First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
     * @param[in] input2   Second tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
     * @param[in] output   Output tensor info. Data types supported: U8/SQASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
     * @param[in] policy   Policy to use to handle overflow
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEARITHMETICADDITION_H */
