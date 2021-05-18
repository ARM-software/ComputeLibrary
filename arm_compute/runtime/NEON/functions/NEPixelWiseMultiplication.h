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
#ifndef ARM_COMPUTE_NEPIXELWISEMULTIPLICATION_H
#define ARM_COMPUTE_NEPIXELWISEMULTIPLICATION_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to run @ref cpu::CpuMul */
class NEPixelWiseMultiplication : public IFunction
{
public:
    /** Default Constructor */
    NEPixelWiseMultiplication();
    /** Default Destructor */
    ~NEPixelWiseMultiplication();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPixelWiseMultiplication(const NEPixelWiseMultiplication &) = delete;
    /** Default move constructor */
    NEPixelWiseMultiplication(NEPixelWiseMultiplication &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPixelWiseMultiplication &operator=(const NEPixelWiseMultiplication &) = delete;
    /** Default move assignment operator */
    NEPixelWiseMultiplication &operator=(NEPixelWiseMultiplication &&) = default;
    /** Initialise the kernel's inputs, output and convertion policy.
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
     * |U8             |U8             |S16            |
     * |U8             |S16            |S16            |
     * |S16            |U8             |S16            |
     * |S16            |S16            |S16            |
     * |F16            |F16            |F16            |
     * |F32            |S32            |F32            |
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *
     * @param[in, out] input1          An input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     *                                 This input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          An input tensor. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), QASYMM8_SIGNED (only if @p input1 is QASYMM8_SIGNED), S16, S32, QSYMM16 (only if @p input1 is QSYMM16), F16 (only if @p input1 is F16), F32 (only if @p input1 is F32).
     *                                 This input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported:
     *                                 - U8, only if both inputs are U8.
     *                                 - QASYMM8, only if both inputs are QASYMM8.
     *                                 - QASYMM8_SIGNED, only if @p input1 is QASYMM8_SIGNED.
     *                                 - S16.
     *                                 - QSYMM16, only if both inputs are QSYMM16.
     *                                 - S32, only if both inputs are S32 or both are QSYMM16.
     *                                 - F16, only if @p input1 is F16.
     *                                 - F32, only if both inputs are F32.
     * @param[in]      scale           Scale to apply after multiplication.
     *                                 Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     *                                 If both @p input1, @p input2 and @p output are of datatype S32, scale cannot be 1/255
     * @param[in]      overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the inputs is of quantized datatype
     * @param[in]      rounding_policy Rounding policy.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEPixelWiseMultiplication
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *
     * @param[in] input1          An input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in] input2          An input tensor info. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), QASYMM8_SIGNED (only if @p input1 is QASYMM8_SIGNED), S16, S32, QSYMM16 (only if both inputs are QSYMM16), F16 (only if @p input1 is F16), F32 (only if @p input1 is F32).
     * @param[in] output          Output tensor info. Data types supported:
     *                            - U8, only if both inputs are U8.
     *                            - QASYMM8, only if both inputs are QASYMM8.
     *                            - QASYMM8_SIGNED, only if @p input1 is QASYMM8_SIGNED.
     *                            - S16.
     *                            - QSYMM16, only if both inputs are QSYMM16.
     *                            - S32, only if both inputs are S32 or both are QSYMM16.
     *                            - F16, only if @p input1 is F16.
     *                            - F32, only if both inputs are F32.
     * @param[in] scale           Scale to apply after multiplication.
     *                            Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     *                            If both @p input1, @p input2 and @p output are of datatype S32, scale cannot be 1/255
     * @param[in] overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the inputs is of quantized datatype
     * @param[in] rounding_policy Rounding policy.
     * @param[in] act_info        (Optional) Activation layer information in case of a fused activation. Currently not supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref cpu::CpuComplexMul. */
class NEComplexPixelWiseMultiplication : public IFunction
{
public:
    /** Default Constructor */
    NEComplexPixelWiseMultiplication();
    /** Default Destructor */
    ~NEComplexPixelWiseMultiplication();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEComplexPixelWiseMultiplication(const NEComplexPixelWiseMultiplication &) = delete;
    /** Default move constructor */
    NEComplexPixelWiseMultiplication(NEComplexPixelWiseMultiplication &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEComplexPixelWiseMultiplication &operator=(const NEComplexPixelWiseMultiplication &) = delete;
    /** Default move assignment operator */
    NEComplexPixelWiseMultiplication &operator=(NEComplexPixelWiseMultiplication &&) = default;
    /** Initialise the kernel's inputs, output.
     *
     * @param[in, out] input1   An input tensor. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2   An input tensor. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output   The output tensor. Data types supported: same as @p input1. Number of channels: same as @p input1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEComplexPixelWiseMultiplication
     *
     * @param[in] input1   An input tensor info. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     * @param[in] input2   An input tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[in] output   The output tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEPIXELWISEMULTIPLICATION_H */
