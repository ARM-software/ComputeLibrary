/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEPIXELWISEMULTIPLICATION_H__
#define __ARM_COMPUTE_NEPIXELWISEMULTIPLICATION_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunction.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEPixelWiseMultiplicationKernel */
class NEPixelWiseMultiplication : public INESimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and convertion policy.
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *
     * @param[in, out] input1          An input tensor. Data types supported: U8/QASYMM8/S16/QSYMM16/F16/F32
     *                                 This input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          An input tensor. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), S16, QSYMM16 (only if @p input1 is QSYMM16), F16 (only if @p input1 is F16), F32 (only if @p input1 is F32).
     *                                 This input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported: U8 (Only if both inputs are U8), QASYMM8 (only if both inputs are QASYMM8), S16, QSYMM16 (only if both inputs are QSYMM16), F16 (only if @p input1 is F16), F32 (only if both inputs are F32).
     * @param[in]      scale           Scale to apply after multiplication.
     *                                 Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     * @param[in]      overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if datatype is QASYMM8 or QSYMM16.
     * @param[in]      rounding_policy Rounding policy.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPixelWiseMultiplication
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *
     * @param[in] input1          An input tensor info. Data types supported: U8/QASYMM8/S16/QSYMM16/F16/F32
     * @param[in] input2          An input tensor info. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), S16, QSYMM16 (only if both inputs are QSYMM16), F16 (only if @p input1 is F16), F32 (only if @p input1 is F32).
     * @param[in] output          Output tensor info. Data types supported: U8 (Only if both inputs are U8), QASYMM8 (only if both inputs are QASYMM8), S16, QSYMM16 (only if both inputs are QSYMM16), F16 (only if @p input1 is F16), F32 (only if both inputs are F32).
     * @param[in] scale           Scale to apply after multiplication.
     *                            Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     * @param[in] overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if datatype is QASYMM8 or QSYMM16.
     * @param[in] rounding_policy Rounding policy.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);
};

/** Basic function to run @ref NEComplexPixelWiseMultiplicationKernel. */
class NEComplexPixelWiseMultiplication : public INESimpleFunction
{
public:
    /** Initialise the kernel's inputs, output.
     *
     * @param[in, out] input1 An input tensor. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2 An input tensor. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output The output tensor. Data types supported: same as @p input1. Number of channels: same as @p input1.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEComplexPixelWiseMultiplication
     *
     * @param[in] input1 An input tensor info. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     * @param[in] input2 An input tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[in] output The output tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};
}
#endif /*__ARM_COMPUTE_NEPIXELWISEMULTIPLICATION_H__ */
