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
#ifndef __ARM_COMPUTE_CLPIXELWISEMULTIPLICATION_H__
#define __ARM_COMPUTE_CLPIXELWISEMULTIPLICATION_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
// Forward declaration
class ICLTensor;

/** Basic function to run @ref CLPixelWiseMultiplicationKernel. */
class CLPixelWiseMultiplication : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output and convertion policy.
     *
     * @param[in, out] input1          An input tensor. Data types supported: U8/S16/F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          An input tensor. Data types supported: same as @p input1.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          The output tensor, Data types supported: same as @p input1. Note: U8 requires both inputs to be U8.
     * @param[in]      scale           Scale to apply after multiplication.
     *                                 Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     * @param[in]      overflow_policy Overflow policy. Supported overflow policies: Wrap, Saturate
     * @param[in]      rounding_policy Rounding policy. Supported rounding modes: to zero, to nearest even.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, float scale,
                   ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);
    /** Static function to check if given info will lead to a valid configuration of @ref CLPixelWiseMultiplication
     *
     * @param[in] input1          An input tensor info. Data types supported: U8/S16/F16/F32.
     * @param[in] input2          An input tensor info. Data types supported: same as @p input1.
     * @param[in] output          The output tensor info, Data types supported: same as @p input1. Note: U8 requires both inputs to be U8.
     * @param[in] scale           Scale to apply after multiplication.
     *                            Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     * @param[in] overflow_policy Overflow policy. Supported overflow policies: Wrap, Saturate
     * @param[in] rounding_policy Rounding policy. Supported rounding modes: to zero, to nearest even.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale,
                           ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);
};

/** Basic function to run @ref CLComplexPixelWiseMultiplicationKernel. */
class CLComplexPixelWiseMultiplication : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output.
     *
     * @param[in, out] input1 An input tensor. Data types supported: F32. Number of channels supported: 2.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2 An input tensor. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     *                        The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output The output tensor, Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLComplexPixelWiseMultiplication
     *
     * @param[in] input1 An input tensor info. Data types supported: F32. Number of channels supported: 2.
     * @param[in] input2 An input tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[in] output The output tensor info, Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLPIXELWISEMULTIPLICATION_H__ */
