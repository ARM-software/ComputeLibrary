/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEIM2COL_H__
#define __ARM_COMPUTE_NEIM2COL_H__

#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEIm2ColKernel */
class NEIm2Col : public IFunction
{
public:
    /** Default constructor */
    NEIm2Col();
    /** Configure the im2col NEON kernel
     *
     * @param[in]  input       The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                         while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/F16/F32
     *                         Note: QASYMM8 works only for has_bias = false
     * @param[out] output      The output tensor. Data types supported: Same as @p input
     * @param[in]  kernel_dims The kernel dimensions (width and height).
     * @param[in]  conv_info   Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  has_bias    In case biases are provided expands the matrix with 1.
     * @param[in]  dilation    (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  num_groups  (Optional) Number of groups when performing a grouped convolution
     */
    void configure(const ITensor *input, ITensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation = Size2D(1U, 1U),
                   unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref NEIm2Col
     *
     * @param[in] input       The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                        while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/F16/F32
     *                        Note: QASYMM8 works only for has_bias = false
     * @param[in] output      The output tensor. Data types supported: Same as @p input
     * @param[in] kernel_dims The kernel dimensions (width and height).
     * @param[in] conv_info   Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] has_bias    In case biases are provided expands the matrix with 1.
     * @param[in] dilation    (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] num_groups  (Optional) Number of groups when performing a grouped convolution
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation = Size2D(1U, 1U),
                           unsigned int num_groups = 1);

    // Inherited methods overridden:
    void run() override;

private:
    NEIm2ColKernel _kernel;
    unsigned int   _y_dim;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEIM2COL_H__ */
