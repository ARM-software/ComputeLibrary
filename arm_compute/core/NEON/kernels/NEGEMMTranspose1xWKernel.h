/*
 * Copyright (c) 2016, 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMTRANSPOSE1xWKERNEL_H__
#define __ARM_COMPUTE_NEGEMMTRANSPOSE1xWKERNEL_H__

#include "arm_compute/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel which transposes the elements of a matrix in chunks of 1xW, where W is equal to (16 / element size of the tensor)
 *
 * Following an example of how the transposition1xW works when the input data is F32
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
 * a00 & a01 & a02 & a03 & a04 & a05 & a06 & a07 \\
 * a10 & a11 & a12 & a13 & a14 & a15 & a16 & a17 \\
 * a20 & a21 & a22 & a23 & a24 & a25 & a26 & a27 \\
 * a30 & a31 & a32 & a33 & a34 & a35 & a36 & a37 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc}
 * a00 & a01 & a02 & a03 & a04 & a05 & a06 & a07 & a10 & a11 & a12 & a13 & a14 & a15 & a16 & a17 & a20 & a21 & a22 & a23 & a24 & a25 & a26 & a27 & a30 & a31 & a32 & a33 & a34 & a35 & a36 & a37\\
 * \end{array} \right)
 * @f]
 *
 * @note The output matrix will have the following shape: [ height * W, ceil(width / W) ], where W = (16 / element size of the tensor)
 *
 */
class NEGEMMTranspose1xWKernel : public INESimpleKernel
{
public:
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Input tensor. Data types supported: U8/S8/QS8/QASYMM8/U16/S16/QS16/F16/U32/S32/F32
     * @param[out] output Output tensor. Data type supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMTranspose1xWKernel
     *
     * @param[in] input  Input tensor info. Data types supported: U8/S8/QS8/QASYMM8/U16/S16/QS16/F16/U32/S32/F32
     * @param[in] output Output tensor info. Data type supported: same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMTRANSPOSE1xWKERNEL_H__ */
