/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLPIXELWISEMULTIPLICATIONKERNEL_H__
#define __ARM_COMPUTE_CLPIXELWISEMULTIPLICATIONKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the pixelwise multiplication kernel.
 *
 */
class CLPixelWiseMultiplicationKernel : public ICLKernel
{
public:
    /** Default constructor.*/
    CLPixelWiseMultiplicationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLPixelWiseMultiplicationKernel(const CLPixelWiseMultiplicationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLPixelWiseMultiplicationKernel &operator=(const CLPixelWiseMultiplicationKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLPixelWiseMultiplicationKernel(CLPixelWiseMultiplicationKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLPixelWiseMultiplicationKernel &operator=(CLPixelWiseMultiplicationKernel &&) = default;
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input1          An input tensor. Data types supported: U8/QS8/QS16/S16/F16/F32.
     * @param[in]  input2          An input tensor. Data types supported: same as @p input1.
     * @param[out] output          The output tensor, Data types supported: same as @p input1. Note: U8 (QS8, QS16) requires both inputs to be U8 (QS8, QS16).
     * @param[in]  scale           Scale to apply after multiplication.
     *                             Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15. For QS8 and QS16 scale must be 1.
     * @param[in]  overflow_policy Overflow policy. Supported overflow policies: Wrap, Saturate
     * @param[in]  rounding_policy Rounding policy. Supported rounding modes: to zero, to nearest even.
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, float scale,
                   ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);
    /** Static function to check if given info will lead to a valid configuration of @ref CLPixelWiseMultiplicationKernel
     *
     * @param[in] input1          An input tensor info. Data types supported: U8/QS8/QS16/S16/F16/F32.
     * @param[in] input2          An input tensor info. Data types supported: same as @p input1.
     * @param[in] output          The output tensor info, Data types supported: same as @p input1. Note: U8 (QS8, QS16) requires both inputs to be U8 (QS8, QS16).
     * @param[in] scale           Scale to apply after multiplication.
     *                            Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15. For QS8 and QS16 scale must be 1.
     * @param[in] overflow_policy Overflow policy. Supported overflow policies: Wrap, Saturate
     * @param[in] rounding_policy Rounding policy. Supported rounding modes: to zero, to nearest even.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale,
                           ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input1;
    const ICLTensor *_input2;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLPIXELWISEMULTIPLICATIONKERNEL_H__ */
