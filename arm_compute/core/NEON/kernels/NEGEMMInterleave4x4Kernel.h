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
#ifndef __ARM_COMPUTE_NEGEMMINTERLEAVE4x4KERNEL_H__
#define __ARM_COMPUTE_NEGEMMINTERLEAVE4x4KERNEL_H__

#include "arm_compute/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to interleave the elements of a matrix
 *
 * This function puts the values in a 4x4 block of Matrix A on the same row (Interleaved values)
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
 * a00 & a10 & a20 & a30 & a01 & a11 & a21 & a31 & a02 & a12 & a22 & a32 & a03 & a13 & a23 & a33 \\
 * \end{array} \right)
 * @f]
 *
 * After this operation, the output matrix will have the following shape: [ height * 4, ceil(width / 4.0f) ]
 */
class NEGEMMInterleave4x4Kernel : public INESimpleKernel
{
public:
    /* Constructor */
    NEGEMMInterleave4x4Kernel();
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Input tensor. Data types supported: U8/S8/QS8/QASYMM8/QS16/U16/S16/F16/U32/S32/F32
     * @param[out] output Output tensor which stores the interleaved matrix. Data type supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMInterleave4x4Kernel
     *
     * @param[in] input  Input tensor info. Data types supported: U8/S8/QS8/QASYMM8/QS16/U16/S16/F16/U32/S32/F32
     * @param[in] output Output tensor info which stores the interleaved matrix. Data type supported: same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the transpose functions
     *
     * @param[in]  input  An input tensor. Data types supported: U8/S8/QS8/QASYMM8/U16/S16/QS16/F16/U32/S32/F32
     * @param[out] output The output tensor. Data type supported: same as @p input
     * @param[in]  window Region on which to execute the kernel.
     */
    using GEMMInterleaveFunction = void(const ITensor *input, ITensor *output, const Window &window);

    GEMMInterleaveFunction *_func; /**< GEMM interleave function to use for the particular tensor types passed to configure() */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMINTERLEAVE4x4KERNEL_H__*/
