/*
 * Copyright (c) 2016-2020 ARM Limited.
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
#include "arm_compute/runtime/NEON/INESimpleFunction.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEArithmeticSubtractionKernel
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/F16/F32.
 * @note The function performs an arithmetic subtraction between two tensors.
 *
 *  This function calls the following kernels:
 * -# @ref NEFillBorderKernel (In case of broadcasting, in the input being broadcasted)
 * -# @ref NEArithmeticSubtractionKernel
 */
class NEArithmeticSubtraction : public INESimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in]  input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32
     * @param[in]  input2   Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32
     * @param[out] output   Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32
     * @param[in]  policy   Policy to use to handle overflow. Convert policy cannot be WRAP if datatype is quantized.
     * @param[in]  act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticSubtraction
     *
     * @param[in] input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/F32
     * @param[in] input2   Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/F32
     * @param[in] output   Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/F32
     * @param[in] policy   Policy to use to handle overflow. Convert policy cannot be WRAP if datatype is quantized.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());
};
}
#endif /* ARM_COMPUTE_NEARITHMETICSUBTRACTION_H */
