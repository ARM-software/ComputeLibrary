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
#ifndef __ARM_COMPUTE_CLGEMMINTERLEAVE4X4KERNEL_H__
#define __ARM_COMPUTE_CLGEMMINTERLEAVE4X4KERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel which interleaves the elements of a matrix A in chunk of 4x4
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
class CLGEMMInterleave4x4Kernel : public ICLKernel
{
public:
    /** Default constructor */
    CLGEMMInterleave4x4Kernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMInterleave4x4Kernel(const CLGEMMInterleave4x4Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMInterleave4x4Kernel &operator=(const CLGEMMInterleave4x4Kernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMInterleave4x4Kernel(CLGEMMInterleave4x4Kernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMInterleave4x4Kernel &operator=(CLGEMMInterleave4x4Kernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Input tensor. Data types supported: U8/S8/QS8/QASYMM8/U16/S16/QS16/F16/U32/S32/F32
     * @param[out] output Output tensor. Data type supported: same as @p input
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMInterleave4x4Kernel
     *
     * @param[in] input  Input tensor info. Data types supported: U8/S8/QS8/QASYMM8/U16/S16/QS16/F16/U32/S32/F32
     * @param[in] output Output tensor info which stores the interleaved matrix. Data type supported: same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLGEMMINTERLEAVE4X4KERNEL_H__ */
