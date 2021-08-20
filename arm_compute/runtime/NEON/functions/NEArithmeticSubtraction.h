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
#ifndef ARM_COMPUTE_NEARITHMETICSUBTRACTION_H
#define ARM_COMPUTE_NEARITHMETICSUBTRACTION_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref cpu::kernels::CpuSubKernel
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/S32/F16/F32.
 * @note The function performs an arithmetic subtraction between two tensors.
 *
 *  This function calls the following kernels:
 * -# @ref cpu::kernels::CpuSubKernel
 */
class NEArithmeticSubtraction : public IFunction
{
public:
    /** Default Constructor */
    NEArithmeticSubtraction();
    /** Default Destructor */
    ~NEArithmeticSubtraction();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEArithmeticSubtraction(const NEArithmeticSubtraction &) = delete;
    /** Default move constructor */
    NEArithmeticSubtraction(NEArithmeticSubtraction &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEArithmeticSubtraction &operator=(const NEArithmeticSubtraction &) = delete;
    /** Default move assignment operator */
    NEArithmeticSubtraction &operator=(NEArithmeticSubtraction &&);
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
     * @param[in]  input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[in]  input2   Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[out] output   Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[in]  policy   Policy to use to handle overflow. Convert policy cannot be WRAP if datatype is quantized.
     * @param[in]  act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticSubtraction
     *
     * @param[in] input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/F16/F32
     * @param[in] input2   Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/F16/F32
     * @param[in] output   Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/F16/F32
     * @param[in] policy   Policy to use to handle overflow. Convert policy cannot be WRAP if datatype is quantized.
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
#endif /* ARM_COMPUTE_NEARITHMETICSUBTRACTION_H */
