/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGEMMTRANSPOSE1XWKERNEL_H__
#define __ARM_COMPUTE_CLGEMMTRANSPOSE1XWKERNEL_H__

#include "arm_compute/core/CL/ICLSimple2DKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel which transposes the elements of a matrix in chunks of 1x4 if the input data type is F32 or in chunks of 1x8 if the input data type is F16.
 *
 * Following an example of how the transposition1xW works when the input data type is F32
 *
 * @f[
 * \left( \begin{array}{cccc}
 * a00 & a01 & a02 & a03 \\
 * a10 & a11 & a12 & a13 \\
 * a20 & a21 & a22 & a23 \\
 * a30 & a31 & a32 & a33 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{ccccccccccccccccc}
 * a00 & a01 & a02 & a03 & a10 & a11 & a12 & a13 & a20 & a21 & a22 & a23 & a30 & a31 & a32 & a33 \\
 * \end{array} \right)
 * @f]
 *
 * Following an example of how the transposition1xW works when the input data type is F16
 *
 * @f[
 * \left( \begin{array}{cccccccc}
 * a00 & a01 & a02 & a03 & a04 & a05 & a06 & a7 \\
 * a10 & a11 & a12 & a13 & a14 & a15 & a16 & 17 \\
 * a20 & a21 & a22 & a23 & a24 & a25 & a26 & 27 \\
 * a30 & a31 & a32 & a33 & a34 & a35 & a36 & 37 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc}
 * a00 & a01 & a02 & a03 & a04 & a05 & a06 & a07 & a10 & a11 & a12 & a13 & a14 & a15 & a16 & a17 & a20 & a21 & a22 & a23 & a24 & a25 & a26 & a27 & a30 & a31 & a32 & a33 & a34 & a35 & a36 & a37\\
 * \end{array} \right)
 * @f]
 *
 * @note If the input data type is F32, the output matrix will have the following shape: [ height * 4, width / 4 ]
 * @note If the input data type is F16, the output matrix will have the following shape: [ height * 8, width / 8 ]
 * @note If the input data type is U8, the output matrix will have the following shape: [ height * 16, width / 16 ]
 *
 */
class CLGEMMTranspose1xWKernel : public ICLSimple2DKernel
{
public:
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Input tensor. Data types supported: U8/F16/F32
     * @param[out] output Output tensor. Data type supported: same as @p input
     */
    void configure(const ICLTensor *input, ICLTensor *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
};
}
#endif /* __ARM_COMPUTE_CLGEMMTRANSPOSE1XWKERNEL_H__ */
